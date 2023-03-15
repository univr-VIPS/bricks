import numpy as np
import pymesh
import multiprocessing as pro_parallel
import multiprocessing.dummy as thr_parallel
import utils

def extract_outer_legos(lego_mat, mesh, single_lego_sizes):
    res = np.copy(lego_mat)
    out_labels = set()
    border = mesh.voxelized(pitch=single_lego_sizes).matrix #the voxels that have a triangle in them are the border
    #for every voxel check if its lego is on the border
    for iz in range(res.shape[2]):
        for ix in range(res.shape[0]):
            for iy in range(res.shape[1]):
                if border[ix,iy,iz] == True:
                    out_labels.add(res[ix,iy,iz])
    #now loop again and remove anything not outside
    for iz in range(res.shape[2]):
        for ix in range(res.shape[0]):
            for iy in range(res.shape[1]):
                if res[ix, iy, iz] not in out_labels:
                    res[ix, iy, iz] = 0
    return res

def extract_inner_legos(lego_mat, mesh, single_lego_sizes):
    res = np.copy(lego_mat)
    out_labels = set()
    border = mesh.voxelized(pitch=single_lego_sizes).matrix #the voxels that have a triangle in them are the border
    #for every voxel check if its lego is on the border
    for iz in range(res.shape[2]):
        for ix in range(res.shape[0]):
            for iy in range(res.shape[1]):
                if border[ix,iy,iz] == True:
                    out_labels.add(res[ix,iy,iz])
    #now loop again and remove anything on the outside
    for iz in range(res.shape[2]):
        for ix in range(res.shape[0]):
            for iy in range(res.shape[1]):
                if res[ix, iy, iz] in out_labels:
                    res[ix, iy, iz] = 0
    return res

def _load_lego(size):
    return pymesh.load_mesh(f'../meshes/lego_bricks/{size}.obj')


_bricksDict = {
    (1,1): _load_lego('1x1'),
    (1,2): _load_lego('1x2'),
    (1,3): _load_lego('1x3'),
    (1,4): _load_lego('1x4'),
    (1,6): _load_lego('1x6'),
    (1,8): _load_lego('1x8'),
    (2,2): _load_lego('2x2'),
    (2,3): _load_lego('2x3'),
    (2,4): _load_lego('2x4'),
    (2,6): _load_lego('2x6'),
    (2,8): _load_lego('2x8'),
    (2,1): _load_lego('2x1'),
    (3,1): _load_lego('3x1'),
    (4,1): _load_lego('4x1'),
    (6,1): _load_lego('6x1'),
    (8,1): _load_lego('8x1'),
    (3,2): _load_lego('3x2'),
    (4,2): _load_lego('4x2'),
    (6,2): _load_lego('6x2'),
    (8,2): _load_lego('8x2'),
}


def identifyBrick(mat,startX,startY,z):
    dimX, dimY = 0,0
    endX = mat.shape[0]
    endY = mat.shape[1]
    identifier = mat[startX,startY,z]
    for x in range(startX,endX):
        if mat[x][startY][z] == identifier:
            dimY += 1
        else:
            break  
    for y in range(startY,endY):
        if mat[startX][y][z] == identifier:
            dimX += 1
        else:
            break
    if (dimX,dimY) in _bricksDict:
        return _bricksDict[(dimX,dimY)]
    if (dimX-1,dimY) in _bricksDict:
        return _bricksDict[(dimX-1,dimY)]
    if (dimX-2,dimY) in _bricksDict:
        return _bricksDict[(dimX-2,dimY)]
    if (dimX,dimY-1) in _bricksDict:
        return _bricksDict[(dimX,dimY-1)]
    if (dimX,dimY-2) in _bricksDict:
        return _bricksDict[(dimX,dimY-2)]

def place_legos(lego_matrix, single_lego_sizes):
    
    res = []
    used_labels = set()
    used_labels.add(0)
    for iz in range(lego_matrix.shape[2]):
        for ix in range(lego_matrix.shape[0]):
            for iy in range(lego_matrix.shape[1]):
                if lego_matrix[ix,iy,iz] in used_labels: continue
                used_labels.add(lego_matrix[ix,iy,iz])
                brick = identifyBrick(lego_matrix, ix, iy, iz)
                offset = (np.array([ix,iy,iz]) * single_lego_sizes) - single_lego_sizes * .5
                res.append(pymesh.form_mesh(brick.vertices + offset, brick.faces))
    return res

def place_legos_ordered(lego_matrix, single_lego_sizes):
    
    res = []
    used_labels = set()
    used_labels.add(0)
    for iz in range(lego_matrix.shape[2]):
        for ix in range(lego_matrix.shape[0]):
            for iy in range(lego_matrix.shape[1]):
                if lego_matrix[ix,iy,iz] in used_labels: continue
                used_labels.add(lego_matrix[ix,iy,iz])
                brick = identifyBrick(lego_matrix, ix, iy, iz)
                offset = (np.array([ix,iy,iz]) * single_lego_sizes) - single_lego_sizes * .5
                res.append(pymesh.form_mesh(brick.vertices + offset, brick.faces))
    used_labels = set()
    used_labels.add(0)
    for iz in range(lego_matrix.shape[2]):
        for ix in range(lego_matrix.shape[0]):
            for iy in range(lego_matrix.shape[1]):
                if lego_matrix[ix,iy,iz] in used_labels: continue
                used_labels.add(lego_matrix[ix,iy,iz])
                brick = identifyBrick(lego_matrix, ix, iy, iz)
                offset = (np.array([ix,iy,iz]) * single_lego_sizes) - single_lego_sizes * .5
                res[int(lego_matrix[ix,iy,iz])-1] = (pymesh.form_mesh(brick.vertices + offset, brick.faces))

    return res


def lego_mesh_intersections(legos, mesh):
    #save to disk, pymesh meshes are not pickle-able
    for i,l in enumerate(legos):
        pymesh.save_mesh(f"/tmp/l{i}.ply", legos[i])
    pymesh.save_mesh(f"/tmp/base.ply", mesh)
    #parallel intersection ~7x faster
    pool = pro_parallel.Pool()
    pool.map(_intersection, range(len(legos)))
    #load the results
    res = []
    for i,l in enumerate(legos):
        res.append(pymesh.load_mesh(f"/tmp/l{i}.ply"))
    return res

"""
def repair_mesh(mesh,legos,custom_legos):
    print(mesh.bounds)
    for i in range(len(legos)):
        print(legos[i].bbox)
    return custom_legos
"""    
def merge_faulty_meshes(base_z,legos,custom_legos):
        
    lego_height = 9.6
    res = []
    i = 0
    #first_layer_legos = []
    first_layer = []
    
    merged_bricks = {b: False for b in legos}
    
    while i < len(legos):
        
        min_z_lego = legos[i].bbox[0][2]
        #first layer
        if (min_z_lego <= base_z + 1):
            
            trans_z = -(custom_legos[i].bbox[1][2] - custom_legos[i].bbox[0][2]) + 1.6
            tmp_lego = utils.trans_mesh(legos[i],0,0,trans_z)
            custom_legos[i] = pymesh.boolean(custom_legos[i],tmp_lego,"difference","igl")
            pymesh.save_mesh(f"../meshes/out/diff{i}.ply",custom_legos[i])
            #first_layer_legos.append(legos[i])
            first_layer.append(custom_legos[i])
            #res.append(custom_legos[i])
            merged_bricks[legos[i]] = True
            i+=1
            continue
        
        if merged_bricks[legos[i]] == True:
            #current brick has already been merged
            i+=1
            continue
            
        max_z_lego = legos[i].bbox[1][2]
        max_z_custom = custom_legos[i].bbox[1][2]
        min_z_custom = custom_legos[i].bbox[0][2]
        
        top_vertices_lego = legos[i].vertices[legos[i].vertices[:,2]==max_z_lego]
        top_vertices_custom = custom_legos[i].vertices[custom_legos[i].vertices[:,2]==max_z_lego]
        bottom_vertices_lego = [x for x in legos[i].vertices[:,2] if (min_z_lego+1.59)<=x<=(min_z_lego+1.61)]
        bottom_vertices_custom = [x for x in custom_legos[i].vertices[:,2] if (min_z_lego+1.59)<=x<=(min_z_lego+1.61)]
        
        #almost always, custom legos vertices are less than those of original pieces
        #let's say, if more than 60% of top vertices are present, there is no need to merge
        perc_top = len(top_vertices_custom)/len(top_vertices_lego)
        #print(f"lego {i} ha {len(bottom_vertices_lego)} stud inferiori")
        perc_bottom = len(bottom_vertices_custom)/len(bottom_vertices_lego)
        #perc_total = len(custom_legos[i].vertice)/len(legos[i].vertices)
        j = 0
        
        #if merged_bricks[legos[j]]
        if perc_top < 0.5 or perc_bottom < 0.5 or min_z_custom != min_z_lego:
            z_tmp = min_z_lego
            #j = i
            
            #search for the first, yet-to-be-merged brick
            for key,val in merged_bricks.items():
                if val == True:
                    j+=1
                else:
                    break
            
            #startz = min_z_lego
            #while checked meshes are on same layer as current mesh
            while merged_bricks[legos[i]] == False:
            
                
                #greedy, first option found is the one chosen
                #if any(point in tmp_lego.vertices for point in legos[i].vertices) and (merged_bricks[legos[j]] == False):
                if merged_bricks[legos[j]] == False and j!=i:
                    if check_adjacency(merged_bricks,legos[i],custom_legos[i],legos[j], custom_legos[j],i,j):
                        res.append(pymesh.merge_meshes([custom_legos[i],custom_legos[j]]))
                        merged_bricks[legos[j]] = True
                        merged_bricks[legos[i]] = True
                        
                        #print(merged_bricks)
                j+=1
                if j == len(legos):
                    #didn't find adjacent bricks...maybe was the last one of the layer
                    res.append(custom_legos[i])
                    merged_bricks[legos[i]] = True
                    
                #z_tmp = legos[j].bbox[0][2]
                #print(f"z_min_lego è {min_z_lego} e nuova z è {z_tmp}")
       
        i+=1
        
        """
        print("-------------------------------------------")
        """

    """?"""
    #first_layer = utils.trans_mesh(first_layer,0,0,9.6)
    res.append(pymesh.merge_meshes(first_layer))
   
    c = 0
    #add to res the bricks never used, if present
    for key,val in merged_bricks.items():
        if val == False:
            res.append(custom_legos[c])
        c+=1
        
    to_be_merged = []
    dict_cleanup = {el: [] for el in res}
    #links every problematic mesh to the one under it
    count = 0
  
    for el in res:
        height = el.bbox[1][2] - el.bbox[0][2]
        width = el.bbox[1][1] - el.bbox[0][1]
        length = el.bbox[1][0] - el.bbox[0][0]
        if (height < 4 or width < 4 or length < 4):
            piece_under = check_z(el,res,count)
            if piece_under != None:
                dict_cleanup[piece_under].append(el)
        count+=1
    
    #final merge
    
    #k = 0
    for key,val in dict_cleanup.items():
       
        count2=0
        if val: #list not empty
            res.remove(key)
            #print(f"key {k}... {key.bbox}")
            #pymesh.save_mesh(f"../meshes/out/debug{k}-{count2}.ply",key)
            #count2+=1
            for el in val:
                if el in res:
                    res.remove(el)
                #print(f"el {count2}...{el.bbox}")
                #pymesh.save_mesh(f"../meshes/out/debug{k}-{count2}.ply",el)
                count2+=1
            val.append(key)
            #print(f"{key}:{val}")
            
            res.append(pymesh.merge_meshes(val))
            #pymesh.save_mesh(f"../meshes/out/debug{count2}.ply",pymesh.merge_meshes(val))
            #count2+=1
            #k+=1
           
                       
    return res

def check_adjacency(merged_bricks,lego1,custom1,lego2,custom2,i,j):
    
    thresh = 0.1 
    lego1_xmax = lego1.bbox[1][0]
    lego1_xmin = lego1.bbox[0][0]
    lego1_ymax = lego1.bbox[1][1]
    lego1_ymin = lego1.bbox[0][1]
    lego1_zmin = lego1.bbox[0][2]
    
    custom1_xmax = custom1.bbox[1][0]
    custom1_xmin = custom1.bbox[0][0]
    custom1_ymax = custom1.bbox[1][1]
    custom1_ymin = custom1.bbox[0][1]
    custom1_zmin = custom1.bbox[0][2]
    
    lego2_xmax = lego2.bbox[1][0]
    lego2_xmin = lego2.bbox[0][0]
    lego2_ymax = lego2.bbox[1][1]
    lego2_ymin = lego2.bbox[0][1]
    lego2_zmin = lego2.bbox[0][2]
    lego2_zmax = lego2.bbox[1][2]
    
    custom2_xmax = custom2.bbox[1][0]
    custom2_xmin = custom2.bbox[0][0]
    custom2_ymax = custom2.bbox[1][1]
    custom2_ymin = custom2.bbox[0][1]
    custom2_zmin = custom2.bbox[0][2]
    
    if lego1_zmin != lego2_zmin:
        #meshes on different layers
        return False
    
    custom2_zmax = custom2.bbox[1][2]
    custom2_zmin = custom2.bbox[0][2]
    
    top_vertices_lego2 = lego2.vertices[lego2.vertices[:,2]==lego2_zmax]
    top_vertices_custom2 = custom2.vertices[custom2.vertices[:,2]==lego2_zmax]
    
    perc_top2 = len(top_vertices_custom2)/len(top_vertices_lego2)
    perc_total1 = len(custom1.vertices)/len(lego1.vertices)
    perc_total2 = len(custom2.vertices)/len(lego2.vertices)
    
    len2_y = lego2_ymax - lego2_ymin
    len2_x = lego2_xmax - lego2_xmin
    
    custom_len2_y = custom2_ymax - custom2_ymin
    custom_len2_x = custom2_xmax - custom2_xmin
   
    if ( (custom1_xmax+thresh >= custom2_xmin and custom1_xmin<custom2_xmin) or (custom1_xmin-thresh <= custom2_xmax and custom1_xmax>custom2_xmax) ) and (custom1_ymin - custom_len2_y <= custom2_ymin and custom1_ymax + custom_len2_y >= custom2_ymax):
        #adjacent left and right
        #print(f"blocco {j} merged con {i}")
        return True
    elif ( (custom1_ymax+thresh >= custom2_ymin and custom1_ymin<custom2_ymin) or (custom1_ymin-thresh <= custom2_ymax and custom1_ymax>custom2_ymax) ) and (custom1_xmin - custom_len2_x <= custom2_xmin and custom1_xmax + custom_len2_x >= custom2_xmax):
        #if adjacent vertically
        #print(f"blocco {j} merged con {i}")
        return True
    else:
        return False
    
def check_z(custom1,custom_list,count):
    #print(f"pezzo {count} controlla sotto di sè")
    thresh = 0.1
    count2 = 0
    for el in custom_list:
        if (el.bbox[1][2] - el.bbox[0][2] < 4) or (el.bbox[1][1] - el.bbox[0][1] < 4) or (el.bbox[1][0] - el.bbox[0][0] < 4):
            continue
        #if on layer below
        if (el.bbox[0][2] - thresh <= custom1.bbox[1][2] and el.bbox[0][2] > custom1.bbox[0][2]):
            center_y = (custom1.bbox[1][1]+custom1.bbox[0][1])/2
            center_x = (custom1.bbox[1][0]+custom1.bbox[0][0])/2
            if ( center_x <= el.bbox[1][0] and center_x >= el.bbox[0][0] ) and ( center_y <= el.bbox[1][1] and center_y >= el.bbox[0][1] ):
                """debug
                print(f"pezzo {count} ha trovato pezzo adiacente su asse z")
                print(custom1.bbox)
                print(el.bbox)
                """
                return el
       
        #if on same layer                                                                                                                                    
        elif custom1.bbox[0][2]>=el.bbox[0][2] and custom1.bbox[1][2] <= el.bbox[1][2]:
            center_x = (custom1.bbox[1][0] + custom1.bbox[0][0])/2
            center_y = (custom1.bbox[1][1] + custom1.bbox[0][1])/2
            if ( center_y >= el.bbox[0][1] and center_y <= el.bbox[1][1] ) and ( ( custom1.bbox[1][0] + thresh >= el.bbox[0][0] and custom1.bbox[0][0] <= el.bbox[0][0] ) or ( custom1.bbox[0][0] - thresh <= el.bbox[1][0] and custom1.bbox[1][0] >= el.bbox[1][0]) ):
                """debug
                print(f"pezzo {count} ha trovato pezzo adiacente su asse y, centro y = {center_y} >= {el.bbox[0][1]}, <= {el.bbox[1][1]}")
                print(custom1.bbox)
                print(el.bbox)
                """
                return el
            elif ( center_x >= el.bbox[0][0] and center_x <= el.bbox[1][0] ) and ( ( custom1.bbox[1][1] + thresh >= el.bbox[0][1] and custom1.bbox[0][1] <= el.bbox[0][1] ) or ( custom1.bbox[0][1] - thresh <= el.bbox[1][1] and custom1.bbox[1][1] >= el.bbox[1][1]) ):
                """debug
                print(f"pezzo {count} ha trovato pezzo adiacente su asse x,centro x = {center_x} >= {el.bbox[0][0]}, <= {el.bbox[1][0]}")
                print(custom1.bbox)
                print(el.bbox)
                """
                return el
                                                                                                                                               
        count2+=1
    return None
     
def fix_disconnected_components(mesh_list):
    
    #fixed_meshes = []
    #check every custom mesh
    for i in range(len(mesh_list)):
        
        disc_comp_list = pymesh.separate_mesh(mesh_list[i])
        
        if len(disc_comp_list)>1:# and mesh_list[i] not in fixed_meshes:

            disc_comp_len = [x.num_vertices for x in disc_comp_list]
            biggest_comp = disc_comp_list[disc_comp_len.index(max(disc_comp_len))]
            mesh_list[i] = biggest_comp
            disc_comp_list.remove(biggest_comp)
            #if more than one component, merge the disconnected ones with the mesh above them
            for comp in disc_comp_list:
                
                #box which contains current disconnected compontent
                comp_xmin = comp.bbox[0][0]
                comp_xmax = comp.bbox[1][0]
                comp_ymin = comp.bbox[0][1]
                comp_ymax = comp.bbox[1][1]
                comp_zmax = comp.bbox[1][2]
                
                for j in range(len(mesh_list)):
                    
                    if mesh_list[i] != mesh_list[j]:
                        
                        #box which contains current custom mesh
                        mesh_j_xmin = mesh_list[j].bbox[0][0]
                        mesh_j_xmax = mesh_list[j].bbox[1][0]
                        mesh_j_ymin = mesh_list[j].bbox[0][1]
                        mesh_j_ymax = mesh_list[j].bbox[1][1]
                        mesh_j_zmin = mesh_list[j].bbox[0][2]
                        mesh_j_zmax = mesh_list[j].bbox[1][2]
                        
                        if (comp_xmin >= mesh_j_xmin and comp_xmax <= mesh_j_xmax ) and (comp_ymin >= mesh_j_ymin and comp_ymax <= mesh_j_ymax ) and (comp_zmax <= mesh_j_zmax and comp_zmax >= mesh_j_zmin):
                            
                            #mesh_list[j] = pymesh.merge_meshes([comp,mesh_list[j]])
                            mesh_list[j] = pymesh.boolean(comp,mesh_list[j],"union")
                            #fixed_meshes.append(mesh_list[j])
                            #print(f"la nuova mesh {j} ha {len(pymesh.separate_mesh(mesh_list[j]))} comp connesse")
                            #pymesh.save_mesh(f"../meshes/out/fix{j}.ply", pymesh.merge_meshes([comp,mesh_list[j]]))
                            break
                
    return mesh_list

def _intersection(i):
        m = pymesh.load_mesh(f"/tmp/base.ply")
        l = pymesh.load_mesh(f"/tmp/l{i}.ply")
        n = pymesh.boolean(l, m, "intersection", "igl")
        pymesh.save_mesh(f"/tmp/l{i}.ply", n)
        #print(i)
        
def _union(i,j):
        m = pymesh.load_mesh(f"/tmp/base.ply")
        l = pymesh.load_mesh(f"/tmp/l{i}.ply")
        n = pymesh.boolean(l, m, "intersection", "igl")
        pymesh.save_mesh(f"/tmp/l{i}.ply", n)
