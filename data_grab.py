import os

os.add_dll_directory('C:/Users/johnw/Downloads/openslide-win64-20220811/openslide-win64-20220811/bin')

import openslide
import cv2
from openslide.deepzoom import DeepZoomGenerator
from paquo.projects import QuPathProject
import numpy as np
import collections



def grab_tiles(zeta, tile_number, datatype, tile_sze, img, s):

    s = openslide.open_slide(s)

    x,y = img.hierarchy.annotations[zeta].roi.exterior.xy
    x = list(x)
    y = list(y)

    polygon = [[int(x_),int(y_)] for (x_,y_) in zip(x,y)]


    region_start = (polygon[0][0],polygon[0][1])
    size_region = (polygon[2][0] - polygon[1][0], polygon[1][1] - polygon[3][1])

    print(size_region)

    print(region_start[0])

    low_range = (105, 100, 100)
    high_range = (117, 255, 255)

    tiles = DeepZoomGenerator(s, tile_sze, 0)
    dx, dy = tiles.level_tiles[tiles.level_count - 1]

    for x in range(dx):
        count = 0
        for y in range(dy):

            change_blue = collections.defaultdict(int)
            change_white = []

            coords = tiles.get_tile_coordinates(tiles.level_count-1, (x,y))[0]

            if coords[0] >= region_start[0] and coords[0] <= region_start[0] + size_region[0]:

                if coords[1] >= region_start[1] and coords[1] <= region_start[1] + size_region[1]:

                    im = tiles.get_tile(tiles.level_count - 1, (x,y))
                    im = np.array(im, dtype='uint8')
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                    im2 = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                    im2 = cv2.inRange(im2, low_range, high_range)
                    # for i, row in enumerate(im3):
                    #     for j, val in enumerate(row):
                    #         if val == 255:
                    #             change_blue[(i,j)] = 1
                    im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
                    # for i, row in enumerate(im):
                    #     for j, val in enumerate(row):
                    #         lookupVal = change_blue[(i,j)]
                    #         if val[0] >= 235 and val[1] >= 235 and val[2] >= 235:
                    #             im2[i][j] = np.asarray([255, 255, 255])
                    #         elif lookupVal != 0:
                    #             im2[i][j] = np.asarray([212, 188, 113])
                    #         else:
                    #             im2[i][j] = np.asarray([0, 0, 0])
                    
                    catted = cv2.hconcat((im,im2))
                    im_name = f'{datatype}_tile_{tile_number}.png'
                    print(im_name)
                    cv2.imwrite(im_name, catted)
                    #cv2.waitKey()

                    tile_number += 1
    return tile_number


if __name__ == "__main__":

    qp = QuPathProject('./qupathproj', mode="r")
    tile_index = 0
    img_size = 256
    images_dir = './wholeSlideImages/'
    for i in range(len(qp.images)):
        for j in range(len(qp.images[i].hierarchy.annotations)):
            if j == len(qp.images[i].hierarchy.annotations) - 1:
                datatype = './val/val'
            else:
                datatype = './train/train'
            
            tile_index = grab_tiles(zeta=j, tile_number=tile_index, datatype=datatype, tile_sze=img_size, img=qp.images[i], s=f'{images_dir}{qp.images[i].image_name}')
    # datatype = './train/train'
    # i=0
    # j = 0
    # grab_tiles(zeta=j, tile_number=tile_index, datatype=datatype, tile_sze=img_size, img=qp.images[i], s=f'{images_dir}{qp.images[i].image_name}')