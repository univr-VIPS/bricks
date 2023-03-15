import trimesh
import numpy as np
import xlsxwriter
import pymesh
import bisect
import heapq

def trans_mesh(mesh,x,y,z):
    return pymesh.form_mesh(mesh.vertices + [[x,y,z]],mesh.faces)


def skew_mesh(mesh):
    res = mesh.copy()
    scale = 0.8/0.96
    scaleMatrix = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, scale, 0],
                   [0, 0, 0, 1]]
    res.apply_transform(scaleMatrix)
    return res


def rotate_mesh(mesh, angle, axis):
    res = mesh.copy()
    rm = trimesh.transformations.rotation_matrix(angle, axis)
    res.apply_transform(rm)
    return res


#scale and translate model to fit plate
def fit_mesh_to_plate(mesh, plate_size, single_lego_dimension = 8):
    res = mesh.copy()
    T = trimesh.transformations.translation_matrix(-mesh.bounding_box.bounds[0])
    side = max((mesh.bounding_box.bounds[1] - mesh.bounding_box.bounds[0])[0:2])
    base = plate_size * single_lego_dimension # 8 millimiter is the size of a single piece of lego
    S = trimesh.transformations.scale_matrix(base/side, [0,0,0])
    M = trimesh.transformations.concatenate_matrices(S, T)
    res.apply_transform(M)
    return res


def lego_to_images(lego_mat_in, comp_dir=None):
    lego_mat = np.rint(lego_mat_in)
    xf, yf, bt = 10, 10, 1
    res = np.full(np.array(lego_mat.shape)*np.array([xf, yf, 1]),1.1)
    if comp_dir is not None:
        max_comp = max(comp_dir.values())+2
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

                if comp_dir is not None:
                    res[ix * xf + bt:(ix + dimX) * xf - bt, iy * yf + bt:(iy + dimY) * yf - bt, iz] = comp_dir[identifier]/max_comp + 0.1
                else:
                    res[ix * xf + bt:(ix + dimX) * xf - bt, iy * yf + bt:(iy + dimY) * yf - bt, iz] = .5

    return res

    
# remove a value from a heap in O(log n), preserving heap-ness
# https://stackoverflow.com/a/10163422
def remove_from_heap(h, item):
    #h.sort()
    #i = find_in_sorted_list(h, item)
    i = h.index(item)
    if (i==-1):
        print(item)
        print(h)
        assert(False)
    h[i] = h[-1]
    h.pop()
    if i < len(h):
        heapq._siftup(h, i)
        heapq._siftdown(h, 0, i)


# remove a value from a heap given a index in O(log n), preserving heap-ness
# https://stackoverflow.com/a/10163422
def remove_from_heap_by_index(h, i):
    assert( 0 <= i < len(h))
    h[i] = h[-1]
    h.pop()
    if i < len(h):
        heapq._siftup(h, i)
        heapq._siftdown(h, 0, i)


# find a value in a sorted list
# https://stackoverflow.com/a/49281681
def find_in_sorted_list(sorted_list, elem):
    # https://docs.python.org/3/library/bisect.html
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(sorted_list, elem)
    if i != len(sorted_list) and sorted_list[i] == elem:
        return i
    return -1



#UNDER THIS IS ALBERTO'S ORIGINAL CODE FROM inout.py

# -- Import mesh from file -- #
def import_model(modelPath, modelName):
    # -- Loading mesh -- #
    mesh = trimesh.load(modelPath+modelName, force='mesh')
    scale = 0.8/0.96
    scaleMatrix = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, scale, 0],
                   [0, 0, 0, 1]]
    mesh.apply_transform(scaleMatrix)
    return mesh


# -- Export mesh to file -- #
def export_model(mesh, modelPath, modelName, legoBaseSize):
    mesh.export(modelPath + 'Legolized_'+str(legoBaseSize)+'Studs_' + modelName)


# -- Show mesh -- #
def show(mesh):
    mesh.show()


# -- Export brickMatrix, its size and the brickList -- #
def export_matrix(mergeMatrix, brickList, modelName):
    np.save("./"+modelName[:len(modelName)-4]+"_brickMatrix", mergeMatrix)
    np.save("./"+modelName[:len(modelName)-4]+"_brickList", brickList)

    text_file = open("./"+modelName[:len(modelName)-4]+"_sizes.txt", "w")
    text_file.write(str(np.shape(mergeMatrix)))
    text_file.close()


# -- Print the list of bricks used -- #
def print_bricks_list(bricksList):
    totalBricks = 0
    for i in range(1, 3):
        for j in range(1, 20):
            bricksCount = bricksList.count(str(i) + "x" + str(j))
            if (i == 2 and j == 1) or (j == 7) or (j == 5) or (bricksCount == 0):
                continue
            else:
                totalBricks += bricksCount
                print("[ " + str(i) + " x " + str(j) + " ] Bricks used: " + str(bricksCount))
    print("[ Corner ] Bricks used: " + str(bricksList.count("Corner")))
    print("--- Total Amount of Bricks: "+str(totalBricks))
    with open('list.txt', 'w') as f:
        for i in range(len(bricksList)):
            if bricksList[i] is None or bricksList[i] == 0:
                continue
            else:
                f.write(str(i) + ":\t" + str(bricksList[i]) + "\n")


def export_spreadsheet(mergeMatrix, modelName):
    workbook = xlsxwriter.Workbook(modelName[:-4]+'_instructions.xlsx')
    worksheet = workbook.add_worksheet()

    row = 0
    for i in range(mergeMatrix.shape[2]):
        array = mergeMatrix[:, :, i]

        for col, data in enumerate(array):
            worksheet.write_column(row, col, data)

        row += mergeMatrix.shape[1] + 2

    worksheet.set_column(0, 1000, 4)

    format1 = workbook.add_format({'bg_color': '#C6EFCE',
                                   'font_color': '#006100'})

    worksheet.conditional_format('A1:ZZ1000',  {'type': 'cell',
                                                'criteria': '>',
                                                'value': 0,
                                                'format': format1})
    workbook.close()
