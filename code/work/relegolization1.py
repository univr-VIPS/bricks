import random
import networkx as nx
import numpy as np
import heapq
import trimesh
from scipy.stats import gaussian_kde

import debug_utils as db
import utils

#######################################################################################################################
# DALL'ORIGINALE                                                                                                     #
#######################################################################################################################


# -- Projects the vertices of the triangles of the mesh and then calculates the area -- #
def projected_areas(mesh):
    # This is a no explicit loop optimized version of the original function (500x faster). Alas, np.einsum is quite obscure and to understand this function is better to check
    # the original not optimized version by Alberto. The math is based on the following links
    # https://stackoverflow.com/questions/8942950/how-do-i-find-the-orthogonal-projection-of-a-point-onto-a-plane
    # https://math.stackexchange.com/questions/738236/how-to-calculate-the-area-of-a-triangle-abc-when-given-three-position-vectors-a
    normals = mesh.face_normals
    faces = mesh.faces
    vertices = mesh.vertices

    planeNormals = np.copy(normals)
    planeNormals[:, 2] = 0
    planeNormals = planeNormals / np.expand_dims(np.linalg.norm(planeNormals, axis=1), axis=1)

    tris = vertices[faces[:]]
    planetrinormals = np.stack((planeNormals, planeNormals, planeNormals), axis=1)

    dots = np.einsum('ijk, ijk->ij', tris, planetrinormals)
    mults = np.einsum('ij,ik->ijk', dots, planeNormals)
    proj_tris = tris - mults

    diff1 = proj_tris[:, 1] - proj_tris[:, 0]
    diff2 = proj_tris[:, 1] - proj_tris[:, 2]
    crosses = np.cross(diff1, diff2)
    areas = np.linalg.norm(crosses, axis=1)
    # areas = np.where((planeNormals[:,0] == 0) & (planeNormals[:,1] == 0) , 0, areas)
    areas = np.nan_to_num(areas)
    return areas * 0.5


# -- Finds the best angle at which the model should be rotated (Z axis only) -- #
def optimal_rotation_angle(mesh, kde_discretize_samples=1000):
    normals = mesh.face_normals

    # -- Projects vertices of triangles and finds projected area -- #
    projectedAreas = projected_areas(mesh)

    # -- For continuity we consider the angles from x-90° to x+90° -- #
    anglesRad = np.arctan2(normals[:, 1], normals[:, 0]) % (np.pi * 0.5)
    angles = np.concatenate((anglesRad - np.pi * 0.5, anglesRad, anglesRad + np.pi * 0.5))
    areas = np.concatenate((projectedAreas, projectedAreas, projectedAreas))

    # -- Finding the max in the gaussian kde distribution -- #
    kde = gaussian_kde(angles, weights=areas)
    samples = np.linspace(0, np.pi * 0.5, kde_discretize_samples)
    distribution = kde.evaluate(samples)
    max_index = distribution.argmax()
    bestAngle = samples[max_index] % (np.pi * 0.5)

    return bestAngle


#######################################################################################################################


# BUILD_GRAPH:
# in    : voxels    Type: VoxelGrid
# out   : graph     Type: Graph

def build_graph(voxels):

    graph = nx.Graph()

    # voxel => nodo = blocco (nodo = [punto partenza, punto arrivo, flag se interno, numero di componente connessa])
    for point in voxels.sparse_indices:
        graph.add_node(tuple(point), end=tuple(point), inner=0, comp=0)

    z_l = 0

    for node in graph.nodes:
        if node[2] < z_l:
            z_l = node[2]

    # Individua i blocchi interni e crea gli edges del grafo
    for node in graph.nodes:

        x = node[0]
        y = node[1]
        z = node[2]

        # Se il blocco è interno, cambia il flag
        if ((x + 1, y, z) in graph.nodes and (x - 1, y, z) in graph.nodes and
            (x, y + 1, z) in graph.nodes and (x, y - 1, z) in graph.nodes and
            (x, y, z + 1) in graph.nodes and (x, y, z + 1) in graph.nodes):
            graph.nodes[node]["inner"] = 1

        # Creazione edges tra le coppie di nodi
        # - Peso 0 se vicini sullo stesso layer
        # - Peso 1 se vicini su layer diversi
        x1 = (x + 1, y, z)
        x2 = (x - 1, y, z)
        y1 = (x, y + 1, z)
        y2 = (x, y - 1, z)
        z1 = (x, y, z + 1)
        z2 = (x, y, z + 1)

        if x1 in graph.nodes:
            graph.add_edge(node, x1, weight=0)
        if x2 in graph.nodes:
            graph.add_edge(node, x2, weight=0)
        if y1 in graph.nodes:
            graph.add_edge(node, y1, weight=0)
        if y2 in graph.nodes:
            graph.add_edge(node, y2, weight=0)
        if z1 in graph.nodes:
            graph.add_edge(node, z1, weight=1)
        if z2 in graph.nodes:
            graph.add_edge(node, z2, weight=1)

    return graph


# COMPUTE_PRIORITY:
# in    : node, graph, use_random_priority
# out   : node's priority

def compute_priority(node, graph, use_random_priority):

    if use_random_priority:
        return random.randint(0,10000) + random.randint(5,1000) / 2000.5

    n_ver_nbr = 0   # Vicini su layer diversi
    n_lat_nbr = 0   # Vicini sullo stesso layer

    for nbr in graph.adj[node]:
        if graph.adj[node][nbr].get("weight") == 0:
            n_lat_nbr = n_lat_nbr + 1
        else:
            n_ver_nbr = n_ver_nbr + 1

    # Vogliamo considerare prima i nodi con basso valore di priorità
    # = Blocchi esterni, non supportati superiormente e / o inferiormente, con meno blocchi vicini sullo stesso layer
    p_border = graph.nodes[node]["inner"] * 100
    p_ver = n_ver_nbr * 5
    p_lateral = n_lat_nbr

    priority = p_border + p_ver + p_lateral

    return priority + random.randint(5,1000) / 20000.5


# GET_NODE_SHAPE:
# in    : graph, node
# out   : shape as string

def get_node_shape(node, graph):

    node_x = node[0]
    node_y = node[1]
    end_x = graph.nodes[node]["end"][0]
    end_y = graph.nodes[node]["end"][1]

    x_dim = abs(end_x - node_x) + 1
    y_dim = abs(end_y - node_y) + 1

    return str(x_dim) + "x" + str(y_dim)


# BUILD_LAYER_DIR: Eseguita solo 1 volta in blocks_merge_1x1
# in    : first node to be created by merge, graph
# out   : a dictionary containing the preferred direction for each layer

def build_layer_dir(voxels_in, graph):
    voxels = voxels_in.matrix
    maxarea = -1
    mazx = -1
    for i in range(voxels.shape[2]):
        if np.sum(voxels[:,:,i]) > maxarea:
            maxarea = np.sum(voxels[:,:,i])
            maxz = i
    
    rows = 0
    cols = 0
    for i in range(voxels.shape[0]):
        if np.sum(voxels[i,:,maxz]) > 0:
            rows += 1
    for i in range(voxels.shape[1]):
        if np.sum(voxels[:,i,maxz]) > 0:
            cols += 1    
    
    direction = "horizontal"
    if cols >= rows:
        direction = "vertical"
    
    layer_dir = dict()
    for i in range(voxels.shape[2]):
        if abs(i - maxz) % 2 == 0:
            layer_dir[i] = direction
        else:
            if direction == "horizontal":
                layer_dir[i] = "vertical"
            else:
                layer_dir[i] = "horizontal"

    return layer_dir


# CHECK_NODES_CONN:
# in    : nodea, nodeb, graph
# out   : boolean value that tells whether two nodes are compatible for a merge (same shape, side by side)

def check_nodes_conn(nodea, nodeb, graph):

    a_x = nodea[0]
    a_y = nodea[1]
    ae_x = graph.nodes[nodea]["end"][0]
    ae_y = graph.nodes[nodea]["end"][1]

    b_x = nodeb[0]
    b_y = nodeb[1]
    be_x = graph.nodes[nodeb]["end"][0]
    be_y = graph.nodes[nodeb]["end"][1]

    shapea = get_node_shape(nodea, graph)
    shapeb = get_node_shape(nodeb, graph)

    # Controlla che i blocchi siano affiancati correttamente e
    # che non si creino blocchi non disponibili come 1x12 o 1x16 o 4x3 o 4x4
    if (a_x == b_x and ae_x == be_x and not (shapea == shapeb and shapea in ["1x6", "1x8", "4x2", "3x2"])) or \
            (a_y == b_y and ae_y == be_y and not (shapea == shapeb and shapea in ["6x1", "8x1", "2x4", "2x3"])):
        return True
    else:
        return False


# FIND_NODE:
# in    : node, shape, pqueue, pqdict, graph, layer_dir
# out   : node chosen by shape, connectibility, priority, preferred direction

def find_node(node, shape, pqueue, pqdict, graph, layer_dir):

    # Trova i vicini adatti (vicini con archi che pesano 0, forma adatta, allineamento corretto)
    nbrs = list()
    for nbr in graph.adj[node]:
        if graph.adj[node][nbr].get("weight") == 0 and \
                ((get_node_shape(nbr, graph) == shape ) or (get_node_shape(nbr, graph) == shape[::-1]) )and \
                check_nodes_conn(node, nbr, graph):
            if shape == "1x1" and get_node_shape(node, graph) in ["1x1", "1x2", "2x1", "1x3", "3x1"] \
                    or (shape == "1x2" and get_node_shape(node, graph) in ["1x1", "1x2", "1x4", "1x6", "2x2", "3x2"]) \
                    or (shape == "2x1" and get_node_shape(node, graph) in ["1x1", "2x1", "4x1", "6x1", "2x2", "2x3"]) \
                    or (shape == "1x3" and get_node_shape(node, graph) in ["1x1", "1x3"]) \
                    or (shape == "3x1" and get_node_shape(node, graph) in ["1x1", "3x1"]) \
                    or (shape == "1x4" and get_node_shape(node, graph) in ["1x4", "1x2"]) \
                    or (shape == "4x1" and get_node_shape(node, graph) in ["4x1", "2x1"]) \
                    or (shape == "1x6" and get_node_shape(node, graph) in ["1x6", "1x2"]) \
                    or (shape == "6x1" and get_node_shape(node, graph) in ["6x1", "2x1"]) \
                    or (shape == "1x8" and get_node_shape(node, graph) in ["1x8"]) \
                    or (shape == "8x1" and get_node_shape(node, graph) in ["8x1"]) \
                    or (shape == "2x2" and get_node_shape(node, graph) in ["1x2", "2x1", "2x2", "6x2", "2x6", "4x2", "2x4"]) \
                    or (shape == "2x3" and get_node_shape(node, graph) in ["2x3", "2x1"]) \
                    or (shape == "3x2" and get_node_shape(node, graph) in ["3x2", "1x2"]) \
                    or (shape == "2x4" and get_node_shape(node, graph) in ["2x4", "2x2"]) \
                    or (shape == "4x2" and get_node_shape(node, graph) in ["4x2", "2x2"]):
                nbrs.append(nbr)
            else:
                shape = shape[::-1]
                if shape == "1x1" and get_node_shape(node, graph) in ["1x1", "1x2", "2x1", "1x3", "3x1"] \
                    or (shape == "1x2" and get_node_shape(node, graph) in ["1x1", "1x2", "1x4", "1x6", "2x2", "3x2"]) \
                    or (shape == "2x1" and get_node_shape(node, graph) in ["1x1", "2x1", "4x1", "6x1", "2x2", "2x3"]) \
                    or (shape == "1x3" and get_node_shape(node, graph) in ["1x1", "1x3"]) \
                    or (shape == "3x1" and get_node_shape(node, graph) in ["1x1", "3x1"]) \
                    or (shape == "1x4" and get_node_shape(node, graph) in ["1x4", "1x2"]) \
                    or (shape == "4x1" and get_node_shape(node, graph) in ["4x1", "2x1"]) \
                    or (shape == "1x6" and get_node_shape(node, graph) in ["1x6", "1x2"]) \
                    or (shape == "6x1" and get_node_shape(node, graph) in ["6x1", "2x1"]) \
                    or (shape == "1x8" and get_node_shape(node, graph) in ["1x8"]) \
                    or (shape == "8x1" and get_node_shape(node, graph) in ["8x1"]) \
                    or (shape == "2x2" and get_node_shape(node, graph) in ["1x2", "2x1", "2x2", "6x2", "2x6", "4x2", "2x4"]) \
                    or (shape == "2x3" and get_node_shape(node, graph) in ["2x3", "2x1"]) \
                    or (shape == "3x2" and get_node_shape(node, graph) in ["3x2", "1x2"]) \
                    or (shape == "2x4" and get_node_shape(node, graph) in ["2x4", "2x2"]) \
                    or (shape == "4x2" and get_node_shape(node, graph) in ["4x2", "2x2"]):
                    nbrs.append(nbr)

    # Prende priorità da pqueue
    chosen = list()         # Lista elementi papabili
    dont_matter = list()    # Lista elementi da rimettere in pqueue

    #while not pqueue.empty():
    #    maybe = pqueue.get()
    #    if maybe[1] in nbrs:
    #        chosen.append(maybe)
    #    else:
    #        dont_matter.append(maybe)
#
    ## Svuota dont_matter
    #if len(dont_matter) != 0:
    #    for el in dont_matter:
    #        pqueue.put(el)
    #dont_matter.clear()

    for n in nbrs:
        if n in pqdict:
            chosen.append((pqdict[n], n))
            #if not ( (pqdict[n], n) in pqueue):
            #    print(pqdict[n], n)
            #    print(pqueue)
            #    print(pqdict)
            #    assert(False)


    # Se non ci sono nodi adatti
    if len(chosen) == 0:
        return "Not found"

    # Tiene i vicini con priorità massima    -> termina se ne rimane solo 1

    # Trovo priorità massima
    max_priority = 0

    for el in chosen:
        if el[0] > max_priority:
            max_priority = int(el[0])

    # Tiene vicini con priorità massima
    for el in chosen:
        if int(el[0]) != max_priority:
            dont_matter.append(el)

    if len(dont_matter) != 0:
        for el in dont_matter:
            if el in chosen:
                chosen.remove(el)

    ## Svuota dont_matter
    #if len(dont_matter) != 0:
    #    for el in dont_matter:
    #        pqueue.put(el)
    dont_matter.clear()


    if len(chosen) == 1:
        # remove chosen from pqueue e pqdict
        utils.remove_from_heap(pqueue, chosen[0])
        del pqdict[chosen[0][1]] #this will crash if chosen[0][1] is not in pqdict, but it should always be
        return chosen[0][1]

    # Tiene i vicini che rispettano la direzione (se esiste layer_dir,
    # ossia se non siamo nella prima iterazione) -> termina se ne rimane solo 1
    # Se non ne rimane nessuno li tiene tutti
    if layer_dir is not None:
        z = node[2]
        layer_direction = layer_dir[z]
        for el in chosen:
            if (layer_direction == "horizontal" and node[0] == el[1][0]) or \
                    (layer_direction == "vertical" and node[1] == el[1][1]):
                dont_matter.append(el)

        if len(dont_matter) != 0:
            for el in dont_matter:
                if el in chosen:
                    chosen.remove(el)
    

    if len(chosen) == 1:
        #for el in dont_matter:
        #    pqueue.put(el)
        # remove chosen from pqueue and pqdict
        utils.remove_from_heap(pqueue, chosen[0])
        del pqdict[chosen[0][1]] #this will crash if chosen[0][1] is not in pqdict, but it should always be
        return chosen[0][1]
    elif len(chosen) == 0 and len(dont_matter) != 0:
        for el in dont_matter:
            chosen.append(el)
        for el in chosen:
            dont_matter.remove(el)

    # Svuota dont_matter
    #if len(dont_matter) != 0:
    #    for el in dont_matter:
    #        pqueue.put(el)
    dont_matter.clear()

    # Sceglie a caso tra i rimanenti
    chosen_node = chosen[random.randrange(len(chosen))]
    # print(f"Random: {chosen_node[1]}")
    #chosen.remove(chosen_node)
#
    ## Svuota chosen
    #if len(chosen) != 0:
    #    for el in chosen:
    #        pqueue.put(el)

    # remove chosen_node from pqueue and pqdict
    utils.remove_from_heap(pqueue, chosen_node)
    del pqdict[chosen_node[1]] #this will crash if chosen_node[1] is not in pqdict, but it should always be

    return chosen_node[1]


# NODES_MERGE:
# in    : graph, nodea, nodeb
# out   : resulting node

def nodes_merge(graph, nodea, nodeb):

    # L'end del blocco più in basso a dx diventa l'end dell'altro
    end = graph.nodes[nodeb]["end"]
    res_node = nodea
    elim_node = nodeb
    if graph.nodes[nodea]["end"][0] >= graph.nodes[nodeb]["end"][0] and \
            graph.nodes[nodea]["end"][1] >= graph.nodes[nodeb]["end"][1]:
        end = graph.nodes[nodea]["end"]
        res_node = nodeb
        elim_node = nodea

    # Se un nodo non è inner allora non lo sarà il loro merge
    inner = 1
    if graph.nodes[nodea]["inner"] == 0 or graph.nodes[nodeb]["inner"] == 0:
        inner = 0

    # Fa ereditare a res_node gli edge di elim_node
    for edge in graph.edges(elim_node):
        if edge[1] != res_node:
            graph.add_edge(res_node, edge[1], weight=graph.edges[edge]["weight"])

    # Elimina elim_node dal grafo
    graph.remove_node(elim_node)

    # Aggiusta le proprietà
    graph.nodes[res_node]["end"] = end
    graph.nodes[res_node]["inner"] = inner
    graph.nodes[res_node]["comp"] = 0

    return res_node


# BOUNDING_BOX_DISTANCE:
# in    : node, bounding box dimensions, direction
# out   : minimum distance of the element from the bounding box

def bounding_box_distance(node, bb_dims, direction):

    if direction == "horizontal":

        # Se direzione orizzontale, calcola la distanza dal lato sinistro o destro della bounding box (asse x)
        res = abs(node[0] - bb_dims[node[2]][0][0])

        if abs(node[0] - bb_dims[node[2]][0][0]) < res:
            res = abs(node[0] - bb_dims[node[2]][0][0])
        if abs(node[0] - bb_dims[node[2]][1][0]) < res:
            res = abs(node[0] - bb_dims[node[2]][1][0])

    elif direction == "vertical":

        # Se direzione verticale, calcola la distanza dal lato superiore o inferiore della bounding box (asse y)
        res = abs(node[0] - bb_dims[node[2]][0][1])

        if abs(node[1] - bb_dims[node[2]][0][1]) < res:
            res = abs(node[1] - bb_dims[node[2]][0][1])
        if abs(node[1] - bb_dims[node[2]][1][1]) < res:
            res = abs(node[1] - bb_dims[node[2]][1][1])

    else:

        # Altrimenti, calcola la distanza globalmente (questo caso viene usato solo per la prima iterazione)
        res = abs(node[0] - bb_dims[node[2]][0][0])

        if abs(node[0] - bb_dims[node[2]][0][0]) < res:
            res = abs(node[0] - bb_dims[node[2]][0][0])
        if abs(node[0] - bb_dims[node[2]][1][0]) < res:
            res = abs(node[0] - bb_dims[node[2]][1][0])

        if abs(node[1] - bb_dims[node[2]][0][1]) < res:
            res = abs(node[1] - bb_dims[node[2]][0][1])
        if abs(node[1] - bb_dims[node[2]][1][1]) < res:
            res = abs(node[1] - bb_dims[node[2]][1][1])

    return res


# CHOOSE_NODE:
# in    : priority queue, pqdict, bounding box dimensions
# out   : what node to use based on the distance from the bounding box

def choose_node(pqueue, pqdict, bb_dims, layer_dir):
    # initialize the list
    node_list = [pqueue[0]]
    node_indices = [0]
    priority = node_list[0][0]
    i=0
    # find all the nodes with equal priority
    while i < len(node_indices):
        ind = node_indices[i]
        if 2*ind+1 < len(pqueue) and pqueue[2*ind+1][0] == priority:
            node_list.append(pqueue[2*ind+1])
            node_indices.append(2*ind+1)
        if 2*ind+2 < len(pqueue) and pqueue[2*ind+2][0] == priority:
            node_list.append(pqueue[2*ind+2])
            node_indices.append(2*ind+2)
        i += 1
    #choose at random among them
    ind = random.randrange(0,len(node_indices))
    node = node_list[ind]
    #remove them form heap and dict
    #utils.remove_from_heap(pqueue, node)
    utils.remove_from_heap_by_index(pqueue, node_indices[ind])
    #if not node[1] in pqdict:
    #    print(pqdict)
    del pqdict[node[1]]
    return node
    
    ## Prende i primi elementi della priority queue con valore di priorità uguale
    #nodes_list = list()
    ##full_node = pqueue.get()
    #full_node = heapq.heappop(pqueue)
    #del pqdict[full_node[1]]
    #nodes_list.append(full_node)
    #pr = full_node[0]
    #while pqueue: # while pqueue is not empty
    #    if pqueue[0][0] == pr: # the priority of the next is the same
    #        candidate = heapq.heappop(pqueue)
    #        del pqdict[candidate[1]]
    #        nodes_list.append(candidate)
    #    else:
    #        break
    #
    ##while pqueue: # while pqueue is not empty
    ##    #candidate = pqueue.get()
    ##    candidate = heapq.heappop(pqueue)
    ##    del pqdict[candidate[1]]
    ##    if candidate[0] == pr:
    ##        nodes_list.append(candidate)
    ##    else:
    ##        #pqueue.put(candidate)
    ##        heapq.heappush(pqueue, candidate)
    ##        pqdict[candidate[1]] = candidate[0]
    ##        break
#
    ## Se ci sono più candidati, sceglie il nodo più vicino alla bounding box (in base alla direzione)
    ## Rimette tutti gli altri nella pqueue
    #if len(nodes_list) > 1:
    #    if layer_dir is not None:
    #        min_dist = bounding_box_distance(full_node[1], bb_dims, layer_dir[full_node[1][2]])
    #    else:
    #        min_dist = bounding_box_distance(full_node[1], bb_dims, "None")
    #    for node in nodes_list:
    #        if node != full_node:
    #            if layer_dir is not None:
    #                dist = bounding_box_distance(node[1], bb_dims, layer_dir[node[1][2]])
    #            else:
    #                dist = bounding_box_distance(node[1], bb_dims, "None")
    #            if dist < min_dist:
    #                #pqueue.put(full_node)
    #                heapq.heappush(pqueue, full_node)
    #                pqdict[full_node[1]] = full_node[0]
    #                min_dist = dist
    #                full_node = node
    #            else:
    #                #pqueue.put(node)
    #                heapq.heappush(pqueue, node)
    #                pqdict[node[1]] = node[0]
#
#
    ##del pqdict[full_node[1]] #this will crash if fullnode[1] is not in pqdict, but it should always be
    #return full_node


# BLOCKS_MERGE:
# in    : graph, pqueue, pqdict, node shape, preferred directions' dictionary,
#         bounding box dimensions, voxels (only for animation's sake), do random
# out   : new priority queue

def blocks_merge(graph, pqueue, pqdict, shape, layer_dir, bb_dims, voxels, use_random_priority):

    new_pqueue = [] #queue.PriorityQueue()
    new_pqdict = dict()
    
    # I nodi vengono aggiunti alla nuova coda se non ci sono altri nodi con cui unirli ossia se:
    # - find_node restituisce "Not found"
    # - la coda si svuota
    # Il ciclo termina quando la priority queue è vuota

    # PER ANIMAZIONE ##########################
    # db.save_levels(graph, voxels, None)
    ###########################################

    a = 1

    while True:

        # Sceglie tra i primi elementi della pqueue (con stesso valore di priorità)
        # quello più vicino alla bounding box del layer
        full_node = choose_node(pqueue, pqdict, bb_dims, layer_dir)

        if not pqueue:#  if pqueue is empty
            #new_pqueue.put(full_node)
            heapq.heappush(new_pqueue, full_node)
            new_pqdict[full_node[1]] = full_node[0]
            return new_pqueue, new_pqdict

        node = full_node[1]

        # Cerca il nodo con cui fare il merge
        chosen = find_node(node, shape, pqueue, pqdict, graph, layer_dir)

        if chosen == "Not found":
            #new_pqueue.put(full_node)
            heapq.heappush(new_pqueue, full_node)

            new_pqdict[full_node[1]] = full_node[0]
            continue

        # Unisce i due nodi nel grafo
        res_node = nodes_merge(graph, node, chosen)

        # Ricalcola la priorità
        priority = compute_priority(res_node, graph, use_random_priority)

        # Reinserisce il nodo nella pqueue
        #pqueue.put((priority, res_node))
        heapq.heappush(pqueue, (priority, res_node))
        pqdict[res_node] = priority
        
        # PER ANIMAZIONE ##############################
        # db.save_levels(graph, voxels, res_node)
        ###############################################

        # Se è la prima iterazione, costruisce layer_dir
        #if a == 1:
        #    layer_dir = build_layer_dir(node, graph)

        a += 1


# FROM_GRAPH_TO_MATRIX:
# in    : graph
# out   : matrix based on the given graph, dictionary listing the connected component for each node

def from_graph_to_matrix(graph, voxels):

    comp_dic = dict()   # Per ogni blocco, a quale compconn corrisponde

    matrix = np.zeros((voxels.shape[0], voxels.shape[1], voxels.shape[2]))
    block = 1

    for node in graph.nodes:

        x = node[0]
        y = node[1]
        z = node[2]

        x_e = graph.nodes[node]["end"][0]
        y_e = graph.nodes[node]["end"][1]

        for i in range(x, x_e + 1):
            for j in range(y, y_e + 1):
                matrix[i][j][z] = block

        comp_dic[block] = graph.nodes[node]["comp"]

        block += 1

    return matrix, comp_dic


# ADD_BASEPLATE:
# in    : graph
# out   : graph with baseplate

def add_baseplate(graph):

    # Aggiunge nodo baseplate
    graph.add_node("Baseplate")

    # Collega baseplate al primo layer
    for node in graph.nodes:
        if node != "Baseplate" and node[2] == 0:
            graph.add_edge(node, "Baseplate", weight=1)

    return graph


# BUILD:
# in    : mesh, voxels' dimensions, use_random_priority, input a previous graph
# out   : matrix, dictionary listing the connected component for each node, graph corresponding to the matrix

def build(mesh, single_lego_sizes=np.array([8, 8, 9.6]), use_random_priority=False, graph_in = None):

    voxels = mesh.voxelized(pitch=single_lego_sizes)
    # VoxelGrid object (stores 3D voxels) representing the current mesh discretized into voxels at the specified pitch

    voxels.fill()  # Mutates self by filling in the encoding according to morphology.fill - VoxelGrid type

    # Crea grafo
    if graph_in == None:
        graph = build_graph(voxels)
    else:
        graph = graph_in

    # Crea priority queue e calcola dimensioni della bounding box, per ogni layer
    pqueue = []
    pqdict = dict()
    bb_dims = dict()  # Per ogni layer [bb_min, bb_max] con bb_min = [x_min, y_min] e bb_max = [x_max, y_max]
    for node in graph.nodes:

        if node[2] not in bb_dims:
            bb_dims[node[2]] = [[node[0], node[1]], [node[0], node[1]]]
        else:
            # bb_min
            if node[0] < bb_dims[node[2]][0][0]:
                bb_dims[node[2]][0][0] = node[0]
            if node[1] < bb_dims[node[2]][0][1]:
                bb_dims[node[2]][0][1] = node[1]

            # bb_max
            if node[0] > bb_dims[node[2]][1][0]:
                bb_dims[node[2]][1][0] = node[0]
            if node[1] > bb_dims[node[2]][1][1]:
                bb_dims[node[2]][1][1] = node[1]

        priority = compute_priority(node, graph, use_random_priority)
        #pqueue.put((priority, node))
        heapq.heappush(pqueue, (priority, node))
        pqdict[node] = priority

    # DEBUG: Checkpoint before
    #print("Starting point:")
    db.see_specs(graph)
    #print("-----------------------\n")

    layer_dir = build_layer_dir(voxels, graph)

    for shape in ["1x1", "1x2", "2x1", "1x3", "3x1", "1x4", "4x1", "1x6", "6x1", "1x8", "8x1",
                  "2x2", "2x3", "3x2", "2x4", "4x2"]:
        #print(f"Merge {shape} blocks")
        pqueue, pqdict = blocks_merge(graph, pqueue, pqdict, shape, layer_dir, bb_dims, voxels, use_random_priority)

    # PER ANIMAZIONE ######
    # db.create_gif()
    #######################

    # DEBUG: Checkpoint after
    #print("\nResult: ")
    db.see_specs(graph)
    #print("-----------------------\n")

    # Aggiunge la baseplate per il calcolo delle componenti connesse
    # conncomp_graph = add_baseplate(graph.copy())
    conncomp_graph = graph.copy()

    # Mostra il numero di componenti connesse
    db.see_conn_comp(conncomp_graph)

    # Se il numero di componenti connesse è > 1
    if nx.number_connected_components(conncomp_graph) > 1:

        # Isola le componenti connesse
        conncomps = list()

        for c in nx.connected_components(conncomp_graph):
            conncomps.append(conncomp_graph.subgraph(c).copy())

        for a in range(0, len(conncomps)):
            for node in conncomps[a]:
                if node != "Baseplate":
                    graph.nodes[node]["comp"] = a * 2

        # Dal grafo alla matrice (per ricollegarsi agli step successivi nel main)
        lego_matrix, comp_dic = from_graph_to_matrix(graph, voxels)

        return lego_matrix, comp_dic, graph
    else:
        lego_matrix, comp_dic = from_graph_to_matrix(graph, voxels)
        return lego_matrix, None, graph


# DISASSEMBLE_REGION: remove all the lego bricks inside the given window and puts 1x1 bricks
# in: graph, lego_matrix, x,y,z of the window center, x,y,z dimension of the window
# out: graph
def disassemble_region(graph_in, lego_matrix_in, ix, iy, iz, window_x, window_y, window_z):
    graph = graph_in.copy()
    lego_matrix = lego_matrix_in.copy()
    id2node = [None,]
    for node in graph.nodes:
        id2node.append(node)
        graph.nodes[node]["comp"] = 0
    lego_mat = np.pad(lego_matrix, ((window_x,), (window_y,), (window_z,)), 'constant', constant_values=0)
    valid_id = len(id2node)
    # take all bricks inside a NxNxN windows
    window = lego_mat[ix-(window_x//2)+window_x:ix+(window_x//2)+1+window_x, 
                      iy-(window_y//2)+window_y:iy+(window_y//2)+1+window_y, 
                      iz-(window_z//2)+window_z:iz+(window_z//2)+1+window_z]
    #print(window)
    subnodes = set(id2node[round(n)] for n in window.flatten())
    #print("nodes to remove: ", subnodes)
    # add the individual 1x1x1
    added_nodes = []
    for node in subnodes:
        if node is None:
            continue
        n = graph.nodes[node]
        graph.remove_node(node)
        for i in range(node[0], n["end"][0]+1):
            for j in range(node[1], n["end"][1]+1):
                for k in range(node[2], n["end"][2]+1):
                    lego_matrix[i,j,k] = valid_id
                    valid_id += 1
                    graph.add_node((i,j,k), end=(i,j,k), inner=0, comp=0)
                    id2node.append((i,j,k))
                    added_nodes.append((i,j,k))
    #create the arcs for the new nodes
    for node in added_nodes:
        n = graph.nodes[node]
        i,j,k = node
        neibors_count = 0
        if i-1>=0 and lego_matrix[i-1,j,k] != 0:
            graph.add_edge(node, id2node[round(lego_matrix[i-1,j,k])], weight=0, fromnode=(i,j,k), tonode=(i-1,j,k))
            neibors_count += 1
        if i+1<lego_matrix.shape[0] and lego_matrix[i+1,j,k] != 0:
            graph.add_edge(node, id2node[round(lego_matrix[i+1,j,k])], weight=0, fromnode=(i,j,k), tonode=(i+1,j,k))
            neibors_count += 1
        if j-1>=0 and lego_matrix[i,j-1,k] != 0:
            graph.add_edge(node, id2node[round(lego_matrix[i,j-1,k])], weight=0, fromnode=(i,j,k), tonode=(i,j-1,k))
            neibors_count += 1
        if j+1<lego_matrix.shape[1] and lego_matrix[i,j+1,k] != 0:
            graph.add_edge(node, id2node[round(lego_matrix[i,j+1,k])], weight=0, fromnode=(i,j,k), tonode=(i,j+1,k))
            neibors_count += 1
        if k-1>=0 and lego_matrix[i,j,k-1] != 0:
            graph.add_edge(node, id2node[round(lego_matrix[i,j,k-1])], weight=1, fromnode=(i,j,k), tonode=(i,j,k-1))
            neibors_count += 1
        if k+1<lego_matrix.shape[2] and lego_matrix[i,j,k+1] != 0:
            graph.add_edge(node, id2node[round(lego_matrix[i,j,k+1])], weight=1, fromnode=(i,j,k), tonode=(i,j,k+1))
            neibors_count += 1
        if neibors_count == 6:
            graph.nodes[node]["inner"] = 1
    return graph


def check_heap_dict(h,d):
    assert(len(h) == len(set(h)))
    for (p,n) in h:
        assert(n in d)
        assert(d[n] == p)
    for k in d:
        assert((d[k], k) in h)
