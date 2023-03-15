import math
import os
import queue
import imageio
import numpy as np
import moviepy.editor as mp

import networkx as nx
import matplotlib.pyplot as plt


# Show graph
def g_show(graph):

    nx.draw(graph, with_labels=True, node_size=10, alpha=1, linewidths=1)
    plt.show()


# Get node shape (similar to the function in relegolization.py)
def get_a_node_shape(node, graph):

    node_x = node[0]
    node_y = node[1]
    end_x = graph.nodes[node]["end"][0]
    end_y = graph.nodes[node]["end"][1]

    x_dim = abs(end_x - node_x) + 1
    y_dim = abs(end_y - node_y) + 1

    return x_dim, y_dim, "" + str(x_dim) + "x" + str(y_dim)


# Save graph nodes in a file
def save_nodes(graph, filename):

    f = open(filename + ".txt", "w")
    for node in graph.nodes:
        if node != "Baseplate":
            f.write(f"{node} - {graph.nodes[node]}, shape: {get_a_node_shape(node, graph)[2]}\n")
        else:
            f.write(f"{node}\n")
    f.close()


# Save graph edges in a file
def save_edges(graph, filename):

    f = open(filename + ".txt", "w")
    for edge in graph.edges:
        f.write(f"{edge} - {graph.edges[edge]}\n")
    f.close()


# Print graph number of nodes and number of edges
def see_specs(graph):
    pass
    #print(f"Nodes: {graph.number_of_nodes()} - Edges: {graph.number_of_edges()}")


# Is an element a graph node?
def is_node(a_node, graph):

    print(f"Is {a_node} a node? {any([node for node in graph.nodes if node == a_node])}")


# Save pqueue elements in a file
def save_pqueue(pqueue, filename):

    pqueuea = queue.PriorityQueue()
    f = open(filename + ".txt", "w")
    while not pqueue.empty():
        a = pqueue.get()
        f.write(f"{a}\n")
        pqueuea.put(a)
    f.close()
    return pqueuea


# Save layer_dir in a file
def save_layer_dir(layer_dir):

    f = open("intermediate/layer_dir.txt", "w")
    f.write(str(layer_dir))
    f.close()


# See number of voxels
def see_voxels_number(graph):

    dim = 0

    for node in graph.nodes:

        x = get_a_node_shape(node, graph)[0]
        y = get_a_node_shape(node, graph)[1]

        dim = dim + x * y

    #print(f"Voxels: {dim}")


# Print number of connected components of the given graph
def see_conn_comp(graph):

    for edge in graph.edges:
        if graph.edges[edge]["weight"] == 0:
            graph.remove_edge(edge[0], edge[1])

    #print(f"Number of connected components: {nx.number_connected_components(graph)}")


# Set checkpoint: save nodes, edges, pqueue
def set_checkpoint(graph, pqueue, filename):

    os.chdir("intermediate/" + filename)

    save_nodes(graph, "nodes_" + filename)
    save_edges(graph, "edges_" + filename)
    pqueue = save_pqueue(pqueue, "pqueue_" + filename)

    os.chdir("../..")

    return pqueue


# From graph to matrix (same as the function in reutils.py)
def refrom_graph_to_matrix(graph, voxels):

    comp_dir = dict()   # Per ogni blocco, a quale compconn corrisponde

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

        comp_dir[block] = graph.nodes[node]["comp"]

        block += 1

    return matrix, comp_dir


# Lego to images for animation
def lego_to_images_for_animation(lego_mat, highlighted):

    xf, yf, bt = 10, 10, 1
    res = np.ones(np.array(lego_mat.shape)*np.array([xf, yf, 1]))
    used_labels = set()
    used_labels.add(0)
    for iz in range(lego_mat.shape[2]):
        for ix in range(lego_mat.shape[0]):
            for iy in range(lego_mat.shape[1]):
                if lego_mat[ix, iy, iz] in used_labels:
                    continue
                used_labels.add(lego_mat[ix, iy, iz])
                #let's find the lego dimensions
                dimX, dimY = 0, 0
                endX = lego_mat.shape[0]
                endY = lego_mat.shape[1]
                identifier = lego_mat[ix, iy, iz]
                for x in range(ix, endX):
                    if lego_mat[x][iy][iz] == identifier:
                        dimX += 1
                    else:
                        break
                for y in range(iy, endY):
                    if lego_mat[ix][y][iz] == identifier:
                        dimY += 1
                    else:
                        break

                res[ix * xf:(ix + dimX) * xf, iy * yf:(iy + dimY) * yf, iz] = 0

                if (ix, iy, iz) == highlighted:
                    res[ix * xf + bt:(ix + dimX) * xf - bt, iy * yf + bt:(iy + dimY) * yf - bt, iz] = .1
                else:
                    res[ix * xf + bt:(ix + dimX) * xf - bt, iy * yf + bt:(iy + dimY) * yf - bt, iz] = .5

    return res


# Creates layer images
def save_levels(graph, voxels, highlighted):

    if not os.path.isdir("tmp"):
        os.mkdir("tmp")

    lego_matrix, comp_dir = refrom_graph_to_matrix(graph, voxels)
    im = lego_to_images_for_animation(lego_matrix, highlighted)

    px = 1 / plt.rcParams['figure.dpi']

    plt.figure(figsize=(1024*px, 1024*px))

    plt.subplots_adjust(wspace=.1)
    plt.subplots_adjust(hspace=.1)

    for i in range(0, im.shape[2]):
        plt.subplot(math.ceil(im.shape[2] / 3), 3, i + 1)
        plt.imshow(im[:, :, i], interpolation="nearest")
        plt.axis('off')

    if len(os.listdir("tmp")) in range(0, 10):
        plt.savefig("tmp/00000" + str(len(os.listdir("tmp"))) + ".png", bbox_inches='tight')
    elif len(os.listdir("tmp")) in range(10, 100):
        plt.savefig("tmp/0000" + str(len(os.listdir("tmp"))) + ".png", bbox_inches='tight')
    elif len(os.listdir("tmp")) in range(100, 1000):
        plt.savefig("tmp/000" + str(len(os.listdir("tmp"))) + ".png", bbox_inches='tight')
    elif len(os.listdir("tmp")) in range(1000, 10000):
        plt.savefig("tmp/00" + str(len(os.listdir("tmp"))) + ".png", bbox_inches='tight')
    elif len(os.listdir("tmp")) in range(10000, 100000):
        plt.savefig("tmp/0" + str(len(os.listdir("tmp"))) + ".png", bbox_inches='tight')

    plt.close()


# Creates animation (gif + mp4)
def create_gif():

    files_list = list()

    for path, subdirs, files in os.walk('tmp'):
        for filename in files:
            files_list.append(filename)

    if not os.path.isdir("resulting_layers"):
        os.mkdir("resulting_layers")

    with imageio.get_writer("resulting_layers/result.gif", mode='I') as writer:
        for filename in files_list:
            image = imageio.imread("tmp/" + filename)
            for i in range(4):
                writer.append_data(image)
            os.remove("tmp/" + filename)

    print("\nCreazione gif...\n")
    clip = mp.VideoFileClip("resulting_layers/result.gif")
    clip.write_videofile("resulting_layers/result.mp4")
    os.remove("resulting_layers/result.gif")
