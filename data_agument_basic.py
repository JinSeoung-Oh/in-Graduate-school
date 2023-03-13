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

#이미지 회전
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img_array, None, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    h, w = img_resized.shape[:2] # 중앙 가로, 세로 점
    
    M45 = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1) #45도 회전
    img2 = cv2.warpAffine(img_resized, M45, (w, h))
    dst = img2.copy()
    dst = img2[int(w / 4) : int(w*3 / 4), int(h / 4) : int(h*3 / 4)]
    
    cv2.imwrite(os.path.join(path, img.split('.')[0]+"_rotate.jpg"), dst)


#비디오 프레임 추출 후 저장
path = "C:\\Users\\user\\Desktop\\Video data"
save_image_path = "C:\\Users\\user\\Desktop\\Photos"
video_list = np.array(os.listdir(path))
for index in range(len(video_list)):
    count = 0
    videocap = cv2.VideoCapture(os.path.join(path,video_list[index]))
    success, image = videocap.read()
    success = True
    save_path = os.path.join(save_image_path, video_list[index].split('.')[0])
    while success:
        videocap.set(cv2.CAP_PROP_POS_MSEC, (count*1000))
        success, image = videocap.read()
        if(success):
            print('Read a new frame: ', success)
            if not (os.path.exists(save_path)):
                os.makedirs(save_path)
            cv2.imwrite(save_path + "/frame_%d.jpg" % count, image)
            count = count + 1
videocap.release()

#변경된 사이즈 담을 폴더 생성
def make_output_list(path, lists):
    output_lists = []
    for file_list in range(len(lists)):
        if not (os.path.exists(os.path.join(path, 'resize%d' % int(lists[file_list])))):
            os.makedirs(os.path.join(path, 'resize%d' % int(lists[file_list])))
            output_lists.append('resize%d' % int(lists[file_list]))
    return output_lists
resize_path = "D:/crack dataset/4_videos/extracted_images/"
input_lists = np.array(os.listdir(resize_path))
output_lists = make_output_list(resize_path, input_lists)

#이미지 사이즈 자르기
for in_list, out_list in zip(input_lists, output_lists):
    path_join = os.path.join(resize_path, in_list)
    for img in os.listdir(path_join + '/'):
        img_array = cv2.imread(os.path.join(path_join, img), cv2.IMREAD_COLOR)
        h, w = img_array.shape[:2]
        dst = img_array.copy()
        dst = img_array[:, 0:600]
        cv2.imwrite(resize_path + out_list + '/' + img, dst)
