# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:27:12 2021

@author: rfkjh
"""

import cv2
import mediapipe as mp
import math
import numpy as np
from google.protobuf.json_format import MessageToDict       # 왼손 오른 손을 구분하기 위한 텍스트를 변형하기위한 도구


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 두 수를 비교하여 작은 값과 큰값을 구분한다. 
def comp_int(num1, num2):
    if (num1 < num2):
        return int(num1), int(num2)
    else :
        return int(num2), int(num1)
    
# 인자로 들어온 배열을 좌표로 하여 원점을로부터 길이를 구한다. 넘파이 배일을 받아 다차원 연산이 기능하다. 
def len_line(coord):
    return np.sqrt(np.sum(coord*coord))

# 좌표의 회전이동 공식을 이용하여 현재 탐색한 손이 손바닥 부분인지 손등 부분인지 확인한다.
# 손등의 경우 0 손바닥인 경우 1을 반환한다. 

def isPalm(coord, hand_):
    mov_zero = coord - coord[0]
    len_l = len_line(mov_zero[2])
    sin = -(mov_zero[2][1]/len_l)
    cos = mov_zero[2][0]/len_l
    y2 = coord[1][0]*sin + coord[1][1]*cos
    y17 = coord[3][0]*sin + coord[3][1]*cos
    
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

finger_tip = [3,7,11,15,19] # 각 손가락의 끝부분을 리스트로 관리한다. 
palm_point = [0,2 ,9, 17]   # 손바닥을 판단하기위한 4개의 좌표를 지정하여 리스트로 관리한다. 
# For static images:
IMAGE_FILES = ["./hand_img2.jpg","./hand_img3.jpg","./hand_img4.jpg","./hand_img5.jpg","./hand_img6.jpg","./hand2_img.jpg","./hand2_img2.jpg"]
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
    annotated_image = image.copy()
    img_test = image.copy()
    
    hand_cnt = 0
    for hand_landmarks in results.multi_hand_landmarks:
      '''
      print('hand_landmarks:', hand_landmarks)
      
      print(type(mp_hands.HandLandmark.INDEX_FINGER_TIP))
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      print(
          f'index finger dip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height})'
      )
      
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height})'
      )
      print(
          f'index finger dip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height})'
      )
      '''
      # detecting finter_tips 
      for i in finger_tip:
          idx_ = hand_landmarks.landmark[i + 1].x * image_width
          idy_ = hand_landmarks.landmark[i + 1].y * image_height
          itx_ = hand_landmarks.landmark[i].x * image_width
          ity_ = hand_landmarks.landmark[i].y * image_height
          color = ( 255,0,0)
          img_test = cv2.line(img_test,(int(idx_),int(idy_)),(int(itx_),int(ity_)),color,5)
          
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
      if(front == 0):       # 손바닥 앞면 구분
          color = (255,102,165)
      else :
          color = (61,61,204)
          
      img_test = cv2.line(img_test,(int(palm_point_coordinate[0][0]),int(palm_point_coordinate[0][1])),(int(palm_point_coordinate[1][0]),int(palm_point_coordinate[1][1])),color, 5)
      img_test = cv2.line(img_test,(int(palm_point_coordinate[2][0]),int(palm_point_coordinate[2][1])),(int(palm_point_coordinate[1][0]),int(palm_point_coordinate[1][1])),color, 5)
      img_test = cv2.line(img_test,(int(palm_point_coordinate[3][0]),int(palm_point_coordinate[3][1])),(int(palm_point_coordinate[0][0]),int(palm_point_coordinate[0][1])),color, 5)
      img_test = cv2.line(img_test,(int(palm_point_coordinate[3][0]),int(palm_point_coordinate[3][1])),(int(palm_point_coordinate[2][0]),int(palm_point_coordinate[2][1])),color, 5)
      
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
    cv2.imwrite(
        './' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    '''
    cv2.imwrite(
        './img_res/' + str(idx)+'_' + '.png', cv2.flip(img_test, 1))
    
    
