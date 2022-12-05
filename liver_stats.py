import numpy as np
import os
import cv2
import math


def test_func(num_iter):

    for final in range(num_iter):

        h = 7


    return final

def raw_perc_fibrosis(image_path, raw_img_path):

    images = os.listdir(f'{image_path}/12/')
    raw_images = os.listdir(f'{image_path}/12/')
    print(images)
    num_fiber = 0
    num_non_fiber = 0
    num_useful_fiber = 0
    #count = 0

    for image in images:

        curr_img = cv2.imread(f'{image_path}/12/{image}', 1)
        raw_img = cv2.imread(f'{raw_img_path}/12/{image}', 1)

        for i, row in enumerate(curr_img):

            for j, pixel in enumerate(row):
                
                if image == '0_3.jpeg':
                    print(pixel)

                if pixel[0] >= 100:
                    if pixel[2] == 0:
                        num_useful_fiber += 1
                    num_fiber += 1
                elif raw_img[i][j][0] >=230 and raw_img[i][j][1] >=230 and raw_img[i][j][2]>=230:
                    continue
                else:
                    num_non_fiber +=1

    print(num_useful_fiber)
    print(num_fiber)


    return f'{100*round(num_fiber / (num_fiber + num_non_fiber), 3)} % Fibrotic Tissue', f'{100*round((num_fiber - num_useful_fiber) / (num_fiber + num_non_fiber), 3)} % Excess Fibrotic Tissue'


#print(raw_perc_fibrosis(image_path='./static/objC0_files', raw_img_path='./static/c0_vips_files'))
