# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 19:24:31 2021

@author: rfkjh
"""

import numpy as np
import cv2
import math

def finger_print_blur(img_o, img):
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_o_g = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)

    result = (img_g != img_o_g).astype(np.uint8)
    blur_img = cv2.blur(img_o, (5, 5))
    result = cv2.merge([result, result, result])
    final = blur_img * result

    f_final = final + img

    return f_final

img = cv2.imread('./ellipes_hand.png')
img_o = cv2.imread('./hand_img3.jpg')

img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_o_g = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)

result = (img_g != img_o_g).astype(np.uint8)
blur_img = cv2.blur(img_o,(5,5))
result = cv2.merge([result,result, result])
final = blur_img * result

f_final = final + img

#f_final = finger_print_blur(img_o, img)

cv2.imshow('original',img)
cv2.imshow('jfjfj',img_o)
cv2.imshow('lll', final)
cv2.imshow('jjj', f_final)

cv2.imwrite(
        './blur_progress/' + '1' + '.png', cv2.flip(img_o, 1))
cv2.imwrite(
        './blur_progress/' + '2' + '.png', cv2.flip(img, 1))
cv2.imwrite(
        './blur_progress/' + '3' + '.png', cv2.flip(final, 1))
cv2.imwrite(
        './blur_progress/' + '4' + '.png', cv2.flip(f_final, 1))






cv2.waitKey(0)
cv2.destoryAllWindows()