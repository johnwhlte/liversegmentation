import cv2
import os
import math
import numpy as np
import collections

file_loc = './static/infc0_files/12/'
new_loc = './static/objc0_files/12/'


image_files = os.listdir(file_loc)

for img in image_files:

    img_00 = cv2.imread(f'{file_loc}{img}', 1)

    print(f'{file_loc}{img}')


    total_pixels = img_00.shape[0] * img_00.shape[1]

    new_img = np.copy(img_00)

    for i, row in enumerate(img_00):
        if i >= len(img_00) - 4:
            break
        for j, pixel in enumerate(row):
            if j >= len(row) - 4:
                break
            new_img_row = []
            channel_avg = 0

            for p in range(4):
                for q in range(4):
                    channel_avg += img_00[i+p][j+q][2]

            channel_avg = channel_avg / 16

            if channel_avg >= 100 and channel_avg <= 240:
                new_img[i][j][0] = 0
                new_img[i][j][1] = 255
                new_img[i][j][2] = 255
            else:
                continue

    #new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)


    #lr_gray_scale = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    



    # threshold = 220
    # _, background_mask = cv2.threshold(lr_gray_scale, threshold, 255, cv2.THRESH_BINARY_INV)
    # im_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # cv2.imshow('Original', img)
    # cv2.waitKey(0)
    
    # Convert to graycsale
    # img_gray = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)
    # # Blur the image for better edge detection
    # img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    
    # # Sobel Edge Detection
    # sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    # sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    # cv2.imshow('Sobel X', sobelx)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel Y', sobely)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    # cv2.waitKey(0)


            # if pixel[0] >= 10 and pixel[0] <=220 and pixel[1] >= 10 and pixel[1] <=220 and pixel[2] >= 10 and pixel[2] <= 220:

            #     recolor_dic[(i, j)] = np.asarray([0, 255, 255])

    #sobelxy = cv2.cvtColor(sobelxy, cv2.COLOR_GRAY2BGR)


    cv2.imwrite(f'{new_loc}{img}',new_img)

    # if img == '0_3.jpeg':
    #     img_test = cv2.imread(f'{new_loc}{img}', 1)

    #     for row in img_test:
    #         for pixel in row:
    #             if pixel[0] == 0 and pixel[1] == 255 and pixel[2] == 255:
    #                 print(pixel)