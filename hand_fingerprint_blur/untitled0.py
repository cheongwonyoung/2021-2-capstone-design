# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 21:09:13 2021

@author: rfkjh
"""

import numpy as np
import cv2
import math

def len_line(coord):
    return np.sqrt(np.sum(coord*coord))

def draw_ellipse(image, coord_tip, coord_dip):
    red_color = (0,0,255)
    cood_med = tuple((np.array(coord_tip) + np.array(coord_dip))//2)
    len_l = int(len_line(np.array(coord_tip) - np.array(coord_dip))/2)
    len_2 = int(len_l//2)
    cood_temp= np.array(coord_tip) - np.array(coord_dip)
    cood_angle = int(np.arctan2(cood_temp[1],cood_temp[0]) * 180 / math.pi)
    
    image = cv2.ellipse(image,cood_med,(len_l,len_2),cood_angle,0,360,red_color,2)
    
    return image


blue_color = (255,0,0)
red_color = (0,0,255)
black_color = (0,0,0)
cood1 = (290, 70)
cood2 = (170,290)
co1 = (70,70 )
co2 = (280, 290)

img = np.zeros((384,384,3), np.uint8)
img += 255
img = cv2.ellipse(img,(290,70), (20,50), 0,0,360, blue_color,-1)
img = cv2.line(img,cood1, cood2, red_color, 5)


cood_med = tuple((np.array(cood1) + np.array(cood2))//2)
len_l = int(len_line(np.array(cood1) - np.array(cood2))/2)
len_2 = int(len_l//2)
cood_temp= np.array(cood1) - np.array(cood2)
cood_angle = int(np.arctan2(cood_temp[1],cood_temp[0]) * 180 / math.pi)

img = cv2.ellipse(img,cood_med,(len_l,len_2),cood_angle,0,360,black_color,-1)
#img = draw_ellipse(img, co1, co2)


cv2.imshow('ellipse',img)
cv2.waitKey(0)
cv2.destoryAllWindows()