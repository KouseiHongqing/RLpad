'''
函数说明: 
Author: hongqing
Date: 2021-07-30 16:08:24
LastEditTime: 2021-07-30 17:44:48
'''
import logging
import cv2
import math
import time
import mediapipe as mp
import multiprocessing as mup
import threading as td
from env import Myenv
# from env import env

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def calAngle(pt1, pt2, pt3):
  #print('---------')
  a = math.sqrt(math.pow(pt2[1]-pt1[1],2)+math.pow(pt2[0]-pt1[0],2))
  b = math.sqrt(math.pow(pt3[1]-pt2[1],2)+math.pow(pt3[0]-pt2[0],2))
  c = math.sqrt(math.pow(pt1[1]-pt3[1],2)+math.pow(pt1[0]-pt3[0],2))
  #print(a,b,c)
  angle = math.acos((a*a + b*b -c*c)/(2*a*b))*180/math.pi
  #print(angle)
  return angle

def Normalize_landmarks(image, hand_landmarks):
  new_landmarks = []
  for i in range(0,len(hand_landmarks.landmark)):
    float_x = hand_landmarks.landmark[i].x
    float_y = hand_landmarks.landmark[i].y
    width = image.shape[1]
    height = image.shape[0]
    pt = mp_drawing._normalized_to_pixel_coordinates(float_x,float_y,width,height)
    new_landmarks.append(pt)
  return new_landmarks


def Draw_hand_points(image, normalized_hand_landmarks):
  cv2.circle(image,normalized_hand_landmarks[4],12,(255,0,255),-1, cv2.LINE_AA)
  cv2.circle(image,normalized_hand_landmarks[8],12,(255,0,255),-1, cv2.LINE_AA)
  cv2.line(image, normalized_hand_landmarks[4], normalized_hand_landmarks[8],(255,0,255),3)
  x1, y1 = normalized_hand_landmarks[4][0], normalized_hand_landmarks[4][1]
  x2, y2 = normalized_hand_landmarks[8][0], normalized_hand_landmarks[8][1]
  cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
  length = math.hypot(x2 - x1, y2 - y1)
  if length < 100:
      cv2.circle(image, (cx, cy),12,(0,255,0),cv2.FILLED)
  else:
      cv2.circle(image, (cx, cy),12,(255,0,255),cv2.FILLED)
  return image,length

def videoProcess(vz):
    hands = mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("camera frame is empty!")
            continue
    
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                normalized_landmarks = Normalize_landmarks(image, hand_landmarks)
        
                try:
                    image,length = Draw_hand_points(image, normalized_landmarks)
                    #print(length) #20~300
                    # cv2.rectangle(image, (50, 150), (85, 350), (255, 0, 0), 3)
                    if length > 200:
                        vz.value=1
                        status='turn left'
                    else:
                        vz.value=-1
                        status='turn right'
                    cv2.putText(image,status,(40,410),cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)
                        
                    # cv2.rectangle(image, (50, int(350-length)), (85, 350), (255, 0, 0), cv2.FILLED)
                    percent = int(length / 200.0 * 100)
                    #print(percent)
                    if percent > 100:
                        percent = 100
                    strRate = str(percent) + '%'
                    # cv2.putText(image,strRate,(40,410),cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)
                    cTime = time.time()
                    fps = 1 / (cTime - pTime)
                    pTime = cTime
                    cv2.putText(image, f'FPS: {int(fps)}',(20, 40),cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)
                except:
                    pass
        cv2.imshow('result', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    hands.close()
    cap.release()

def gameProcess(vs):
    env = Myenv()
    while True:
        env.step(vs.value*10)

if __name__=='__main__':
    ctx = mup.get_context("spawn")
    vs = ctx.Value('i', 0) 
    p1 = ctx.Process(target=videoProcess,args=(vs,))
    p2 = ctx.Process(target=gameProcess,args=(vs,))
    p1.start()
    env = Myenv()
    while True:
        env.step(True,vs.value)
    # p1.join()
    # env = Myenv()

    # hands = mp_hands.Hands(
    #     min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     success, image = cap.read()
    #     if not success:
    #         print("camera frame is empty!")
    #         continue
    
    #     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    #     image.flags.writeable = False
    #     results = hands.process(image)
        
    #     image.flags.writeable = True
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     if results.multi_hand_landmarks:
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    #             normalized_landmarks = Normalize_landmarks(image, hand_landmarks)
        
    #             try:
    #                 image,length = Draw_hand_points(image, normalized_landmarks)
    #                 #print(length) #20~300
    #                 # cv2.rectangle(image, (50, 150), (85, 350), (255, 0, 0), 3)
    #                 if length > 200:
    #                     vs.value=1
    #                     status='turn left'
    #                 else:
    #                     vs.value=-1
    #                     status='turn right'
    #                 cv2.putText(image,status,(40,410),cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)
    #                 print(vs.value)
    #                 # cv2.rectangle(image, (50, int(350-length)), (85, 350), (255, 0, 0), cv2.FILLED)
    #                 percent = int(length / 200.0 * 100)
    #                 #print(percent)
    #                 if percent > 100:
    #                     percent = 100
    #                 strRate = str(percent) + '%'
    #                 # cv2.putText(image,strRate,(40,410),cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)
    #                 cTime = time.time()
    #                 fps = 1 / (cTime - pTime)
    #                 pTime = cTime
    #                 cv2.putText(image, f'FPS: {int(fps)}',(20, 40),cv2.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),2)
    #             except:
    #                 pass
    #     cv2.imshow('result', image)
    #     if cv2.waitKey(5) & 0xFF == 27:
    #         break
    # cv2.destroyAllWindows()
    # hands.close()
    # cap.release()
    