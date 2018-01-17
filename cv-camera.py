#!/usr/bin/python

import os
import cv2
from PIL import Image,ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import math

maxImages=50
# lbpcascade: 0.35, haarcascade: 0.65
MAX_HIST_DIFF=0.65

CascadeClassifierPath="./data/lbpcascades/lbpcascade_frontalface_improved.xml"
# CascadeClassifierPath="./data/haarcascades/haarcascade_frontalface_default.xml"

OUT_PUT_DIR='./result/'
if not os.path.exists(OUT_PUT_DIR):
    os.makedirs(OUT_PUT_DIR)

cv2.namedWindow("test")
cap=cv2.VideoCapture(0)
success,frame=cap.read()
classifier=cv2.CascadeClassifier(CascadeClassifierPath)

num=0
imageNo=1

srcImgHist=[]
def isTheSameImg(inputImg):
    isTheSame=True
    global srcImgHist
    inputImgHist = cv2.calcHist([inputImg],[0],None,[256],[0,256])
    if len(srcImgHist) != 0:
        histDiff = cv2.compareHist(inputImgHist,srcImgHist,cv2.HISTCMP_CORREL)
        print('histr %f' % histDiff)
        if histDiff < MAX_HIST_DIFF:
            print('img change detected! histDiff is %d' % histDiff)
            # print('srcImgHist', srcImgHist)
            # print('inputImgHist', inputImgHist)
            isTheSame = False
            srcImgHist = inputImgHist
    else:
        srcImgHist = inputImgHist
    return isTheSame;


def saveImage(image): 
    # save image when new face detected
    global imageNo
    if(not isTheSameImg(image)):    
        if imageNo<=maxImages:
            out_put_file=OUT_PUT_DIR + '%s.jpg'%(str(imageNo))
            cv2.imwrite(out_put_file,frame)
            print('save new image - %s .' % out_put_file)
            imageNo = imageNo + 1


def detectFaces(image_name):
    img = cv2.imread(image_name)
    # 3 means not gray, 2 means gray
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    #1.3, 5 are the min and max detect frame
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result


#save the detected faces
def saveFaces(image_name):
    faces = detectFaces(image_name)
    if faces:
        #save the face image in output dir.
        save_dir = image_name.split('.jpg')[0]+"_faces/"
        #using py3's makedirs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        count = 1
        for (x1,y1,x2,y2) in faces:
            file_name = os.path.join(save_dir,'face-'+str(count)+".jpg")
            Image.open(image_name).crop((x1,y1,x2,y2)).save(file_name)
            count+=1
        return file_name

while success:
    success,frame=cap.read()

    # start to save teh image
    key=cv2.waitKey(100) # wait for the key input with delay time, -1: timeout, 0: null

    size=frame.shape[:2]
    image=np.zeros(size,dtype=np.float16)
    image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # keep the origin_image
    origin_image=image
    # update the image hist
    cv2.equalizeHist(image,image)

    divisor=8
    h,w=size
    minSize=(math.floor(w/divisor),math.floor(h/divisor))
    # minSize=(160, 90)
    faceRects=classifier.detectMultiScale(image,1.1,3,cv2.CASCADE_SCALE_IMAGE,minSize)
    if len(faceRects)>0:
        saveImage(origin_image)

        for faceRect in faceRects:
            x,y,w,h=faceRect
            print('x: %d, y: %d, w: %d, h: %d' % (x,y,w,h))
            cv2.circle(frame,(math.floor(x+w/2),math.floor(y+h/2)),min(math.floor(w/2),math.floor(h/2)),(255,0,0),3)
            cv2.circle(frame,(math.floor(x+w/2),math.floor(y+h/2)),min(math.floor(w/8),math.floor(h/8)),(0,122,0),3)
            cv2.circle(frame,(math.floor(x+w/4),math.floor(y+h/4)),min(math.floor(w/8),math.floor(h/8)),(0,0,255),3)
            cv2.circle(frame,(math.floor(x+3*w/4),math.floor(y+h/4)),min(math.floor(w/8),math.floor(h/8)),(0,0,255),3)

            cv2.rectangle(frame,(math.floor(x+3*w/8),math.floor(y+3*h/4)),(math.floor(x+5*w/8),math.floor(y+7*h/8)),(0,255,0),3)
    cv2.imshow("test",frame)

    if key==ord('c'):
        saveImage(origin_image)
    if key==ord('q'): #quit the loop when input 'q'
        break

if __name__ == '__main__':
    for i in range(1, maxImages):
        IMG_FILE = OUT_PUT_DIR + eval('str(i) + ".jpg"')

        if os.path.exists(IMG_FILE) and saveFaces(IMG_FILE):
            print("face image saved successfully. " + IMG_FILE)
        else:
            print("failed，it maybe not a face. " + IMG_FILE)

cap.release() #close the camera
cv2.destroyWindow("test")


