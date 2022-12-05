import cv2
import os
import math
import numpy as np

file_loc = './static/objc1_files/'

all_files = os.listdir(file_loc)

def write_new_level(loc, idex):

    all_files = os.listdir(f'{loc}{idex}/')

    x_values = []
    y_values = []

    for value in all_files:
        value = value.split('.')
        x_values.append(int(value[0].split('_')[0]))
        y_values.append(int(value[0].split('_')[1]))

    max_x = max(x_values)
    max_y = max(y_values)

    print(max_x)
    print(max_y)

    max_x_x = 1
    max_y_y = 1

    if max_x == 0:
        max_x = 1
        max_x_x = 0
    if max_y == 0:
        max_y = 1
        max_y_y = 0

    for dx in range(max_x):
        dx = dx*2
        if dx > max_x:
            break
        for dy in range(max_y):
            dy = dy*2

            if dy > max_y:
                break

            img_00 = cv2.imread(f'{loc}{idex}/{dx}_{dy}.jpeg', 1)
            img_10 = cv2.imread(f'{loc}{idex}/{dx+1}_{dy}.jpeg',1)
            img_01 = cv2.imread(f'{loc}{idex}/{dx}_{dy+1}.jpeg',1)
            img_11 = cv2.imread(f'{loc}{idex}/{dx+1}_{dy+1}.jpeg',1)

            if img_01 is None:
                if img_10 is None:
                    img_00 = cv2.imread(f'{loc}{idex}/{dx}_{dy}.jpeg', 1)
                    shape_0 = img_00.shape
                    # if max_x_x == 0 and max_y_y == 0:
                    #     new_shape = (shape_0[1], shape_0[0])
                    # else:
                    new_shape = (math.ceil(shape_0[1]/2), math.ceil(shape_0[0]/2))
                    img = cv2.resize(img_00, new_shape, interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(f'{loc}{idex-1}/{math.ceil(dx/2)}_{math.ceil(dy/2)}.jpeg', img)

                else:
                    print('None')

                    img_00 = cv2.imread(f'{loc}{idex}/{dx}_{dy}.jpeg', 1)
                    img_10 = cv2.imread(f'{loc}{idex}/{dx+1}_{dy}.jpeg',1)

                    shape_0 = img_00.shape
                    shape_1 = img_10.shape

                    new_shape = (math.ceil((shape_0[1] + shape_1[1])/2),math.ceil((shape_0[0] + shape_1[0])/4))

                    img = cv2.hconcat((img_00, img_10))
                    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

                    cv2.imwrite(f'{loc}{idex-1}/{math.ceil(dx/2)}_{math.ceil(dy/2)}.jpeg', img)

            elif img_10 is None:

                print('None')
                img_01 = cv2.imread(f'{loc}{idex}/{dx}_{dy}.jpeg',1)
                img_11 = cv2.imread(f'{loc}{idex}/{dx}_{dy+1}.jpeg',1)

                shape_2 = img_01.shape
                shape_3 = img_11.shape

                new_shape = (math.ceil((shape_2[1] + shape_3[1])/4),math.ceil((shape_2[0] + shape_3[0])/2))

                img = cv2.vconcat((img_01, img_11))
                img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(f'{loc}{idex-1}/{math.ceil(dx/2)}_{math.ceil(dy/2)}.jpeg', img)

            else:

                shape_0 = img_00.shape
                shape_1 = img_10.shape
                shape_2 = img_01.shape
                shape_3 = img_11.shape

                new_shape = (math.ceil((shape_0[1] + shape_1[1]  + shape_2[1] + shape_3[1])/4),math.ceil((shape_0[0] + shape_1[0]  + shape_2[0] + shape_3[0])/4))
                #new_shape = (new_shape[1], new_shape[0])

                img_top = cv2.hconcat((img_00, img_10))
                img_bottom = cv2.hconcat((img_01, img_11))

                img = cv2.vconcat((img_top, img_bottom))
                img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(f'{loc}{idex-1}/{math.ceil(dx/2)}_{math.ceil(dy/2)}.jpeg', img)

                    

            
            
            # if dy == 59*2:
            #         cv2.imshow(f'{loc}{idex-1}/{math.ceil(dx/2)}_{math.ceil(dy/2)}.jpeg', img)
            #         cv2.waitKey()
        #print(dx)


    


    return 0


if __name__ == "__main__":
    index = 12

    while index > 0:
        write_new_level(loc=file_loc, idex=index)
        print('gg')
        index = index - 1