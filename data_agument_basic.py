import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

#경로설정
path = "C:\\Users\\user\\Desktop\\Autism_data\\Train_new\\Negative"
path = path + '/'

#좌우반전
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR) #IMREAD_GRAYSCALE
    img_rev_leftright = cv2.flip(img_array, 1) # 좌우반전
    cv2.imwrite(os.path.join(path, img.split('.')[0]+"_rev1.jpg"), img_rev_leftright) 
    
#상하반전
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR) #IMREAD_GRAYSCALE
    img_rev_topdown = cv2.flip(img_array, 0) #상하반전
    cv2.imwrite(os.path.join(path, img.split('.')[0]+"_rev0.jpg"), img_rev_topdown)  
    
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img_array, None, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    h, w = img_resized.shape[:2] # 중앙 가로, 세로 점
    
    M45 = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1) #45도 회전
    img2 = cv2.warpAffine(img_resized, M45, (w, h))
    dst = img2.copy()
    dst = img2[int(w / 4) : int(w*3 / 4), int(h / 4) : int(h*3 / 4)]
    
    cv2.imwrite(os.path.join(path, img.split('.')[0]+"_rotate.jpg"), dst)
