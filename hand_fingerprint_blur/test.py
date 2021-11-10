# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:27:12 2021

@author: rfkjh
"""

import cv2
import mediapipe as mp
import math
import numpy as np
from google.protobuf.json_format import MessageToDict


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


finger_dip = [3,7,11,15,19] # 각 손가락의 끝부분을 리스트로 관리한다. (엄지, 검지, 중지, 약지, 소지)
palm_point = [0,2 ,9, 17]   # 손바닥을 판단하기위한 4개의 지점을 리스트로 관리한다.
thumb_folded = 0

# 두 수를 비교하여 작은 값과 큰 값을 구분한다. 

def comp_int(num1, num2):
    if (num1 < num2):
        return int(num1), int(num2)
    else :
        return int(num2), int(num1)
    
# 인자로 들어온 배열을 좌표로 하여 원정으로부터 길이를 계산한다. 넘파이 배열을 받으며 다차원 또한 가능하다. 

def len_line(coord):
    return np.sqrt(np.sum(coord*coord))

# 좌표의 회전이동 공식을 이용하여 현재 탐지한 손이 손바닥부분인지 손등 부분인지 확인한다 
# 손등인경우 0 손바닥인 경우 1을 반환한다.

def finger_print_blur(img_o, img):
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_o_g = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)

    cv2.waitKey(0)

    result = (img_g != img_o_g).astype(np.uint8)
    blur_img = cv2.blur(img_o, (5, 5))
    result = cv2.merge([result, result, result])
    final = blur_img * result

    f_final = final + img

    return f_final

def isPalm(coord, hand_):
    mov_zero = coord - coord[0]
    len_l = len_line(mov_zero[2])
    sin = -(mov_zero[2][1]/len_l)
    cos = mov_zero[2][0]/len_l
    y2 = coord[1][0]*sin + coord[1][1]*cos          #thumb_mcp
    y17 = coord[3][0]*sin + coord[3][1]*cos         #pinky_mcp
    # 4번 좌표를 받아 회전공식적용 y좌표를 y2와 비교 , 해당 손에 따라 사용
    print(hand_)
    if (y2 < y17):          # 엄지가 아래
        if (hand_ == 0):    # 왼손인 경우
            return 0
        else :              # 오른손인 경우
            return 1
    else :                  # 엄지가 위쪽
        if (hand_ == 0):    # 왼손인 경우
            return 1
        else :              # 오른손인 경우
            return 0
    return 0

def thumb_folded(coord, hand_, cood_thumb_tip, cood_thumb_dip):
    mov_zero = coord - coord[0]
    len_l = len_line(mov_zero[2])
    sin = -(mov_zero[2][1]/len_l)
    cos = mov_zero[2][0]/len_l
    y2 = coord[1][0]*sin + coord[1][1]*cos          #thumb_mcp
    # 4번 좌표를 받아 회전공식적용 y좌표를 y2와 비교 , 해당 손에 따라 사용
    y4 = cood_thumb_tip[0]*sin + cood_thumb_tip[1]*cos

    print(hand_)
    if (y2 < y4):          # 엄지가 관절보다 위
        if (hand_ == 0):    # 왼손인 경우
            return False        # 접혀 있음
        else :              # 오른손인 경우
            return True
    else :                  # 엄지가 관절보다 아래
        if (hand_ == 0):    # 왼손인 경우
            return True
        else :              # 오른손인 경우
            return False
    return False

# 손가락이 접혀있는지 판단하는 알고리즘이다.  !!! 현재 엄지의 손가락이 접혀있는지 판단할 수가 없다. 
# 인자로는 손가락 Tip과 Dip 좌표를 받는다. 
def foldedFinger(coord_tip, coord_dip, rest):
    coord_tip -= rest
    coord_dip -= rest
    folded = []
    for tip, dip in zip(coord_tip, coord_dip):
        if (len_line(tip) > len_line(dip)):
            folded.append(False)
        else:
            folded.append(True)
                
    return folded

 
def draw_ellipse(image, coord_tip, coord_dip, color):
    cood_med = tuple((np.array(coord_tip) + np.array(coord_dip))//2)
    len_l = int(len_line(np.array(coord_tip) - np.array(coord_dip))/2)
    len_2 = int(len_l//2)
    cood_temp= np.array(coord_tip) - np.array(coord_dip)
    cood_angle = int(np.arctan2(cood_temp[1],cood_temp[0]) * 180 / math.pi)
    
    return cv2.ellipse(image,cood_med,(len_l,len_2),cood_angle,0,360,color,-1)



# For static images:
IMAGE_FILES = ["./hand_img3.jpg","./hand_img4.jpg","./hand_img6.jpg","./hand_img5.jpg","./hand2_img.jpg","./hand2_img2.jpg"]
#IMAGE_FILES = ["./hand_img3.jpg"]
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    image_o = image.copy()
    img_test = image.copy()
    
    hand_cnt = 0
    for hand_landmarks in results.multi_hand_landmarks:

      # detecting finter_tips 
      finger_tip_coord = []
      finger_dip_coord = []
      for i in finger_dip:
        itx_ = hand_landmarks.landmark[i + 1].x * image_width
        ity_ = hand_landmarks.landmark[i + 1].y * image_height
        idx_ = hand_landmarks.landmark[i].x * image_width
        idy_ = hand_landmarks.landmark[i].y * image_height
        '''
        color = ( 255,0,0)
        img_test = cv2.line(img_test,(int(idx_),int(idy_)),(int(itx_),int(ity_)),color,5)
        '''
        # detection finger_fold
        finger_tip_coord.append([itx_,ity_])
        finger_dip_coord.append([idx_,idy_])
      
      
          
          
      # detectng palm
      palm_point_coordinate = [];
      for i in palm_point:
        temp = []
        temp.append(hand_landmarks.landmark[i].x * image_width)
        temp.append(hand_landmarks.landmark[i].y * image_height)
        palm_point_coordinate.append(temp)
          
      hand_ = MessageToDict(results.multi_handedness[hand_cnt])['classification'][0]['index']
      
      hand_cnt += 1
      print(palm_point_coordinate)
      
      front = isPalm(np.array(palm_point_coordinate), hand_)
      '''
      차후 손바닥 여부를 판단하여 해당 손을 blur처리 할 것인지 결정한다. 
      또한 손가락의 접힘 정도를 판단하여 blur처리를 하지 않아야할 손가락 또한 구별할 것이다. 
      '''
      if(front == 1):       # 손바닥 앞면 구분
        color = (255,102,165) # 손바닥인경우
        finger_folded = foldedFinger(np.array(finger_tip_coord), np.array(finger_dip_coord),np.array(palm_point_coordinate[0]))

        finger_folded[0] = thumb_folded(np.array(palm_point_coordinate), hand_, np.array(finger_tip_coord[0]),np.array(finger_dip_coord[0]))
        print(finger_folded)

        # 테스트용 이미지 그림
        '''
        img_test = cv2.line(img_test, (int(palm_point_coordinate[0][0]), int(palm_point_coordinate[0][1])),(int(palm_point_coordinate[1][0]), int(palm_point_coordinate[1][1])), color, 5)
        img_test = cv2.line(img_test, (int(palm_point_coordinate[2][0]), int(palm_point_coordinate[2][1])),(int(palm_point_coordinate[1][0]), int(palm_point_coordinate[1][1])), color, 5)
        img_test = cv2.line(img_test, (int(palm_point_coordinate[3][0]), int(palm_point_coordinate[3][1])),(int(palm_point_coordinate[0][0]), int(palm_point_coordinate[0][1])), color, 5)
        img_test = cv2.line(img_test, (int(palm_point_coordinate[3][0]), int(palm_point_coordinate[3][1])),(int(palm_point_coordinate[2][0]), int(palm_point_coordinate[2][1])), color, 5)
        '''
        temp_tip = np.array(finger_tip_coord, dtype=int)
        temp_dip = np.array(finger_dip_coord, dtype=int)
        # 손가락 굽힘 확인

        for i, ff in enumerate(finger_folded):
          if (ff):  # 손가락 굽힘
            color = (0, 153, 0)
          else:  # 안굽힘
            color = (0, 0, 0)
            img = draw_ellipse(img_test, (temp_tip[i][0], temp_tip[i][1]), (temp_dip[i][0], temp_dip[i][1]),color)

        final_img = finger_print_blur(image_o,img)

                # color = (0,0,0)

            #img_test = cv2.line(img_test, (temp_tip[i][0], temp_tip[i][1]), (temp_dip[i][0], temp_dip[i][1]), color, 5)

      '''
      cv2.imshow('test',img_test)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
      '''
    '''
    cv2.imwrite(
        './' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    '''
    cv2.imwrite(
        './img_res/' + str(idx)+'_final' + '.png', cv2.flip(final_img, 1))
    cv2.imwrite(
        './img_res/' + str(idx) + '_ellipes' + '.png', cv2.flip(img, 1))

    
