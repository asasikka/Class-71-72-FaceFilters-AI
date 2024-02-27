import math
import time

from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector

import cv2
import os
import cvzone

captureVideo=cv2.VideoCapture(0)

detector= HandDetector(detectionCon=0.8)

path="filters"

menuImages=[]

pathList=os.listdir(path)
print(pathList)

for i in pathList:
    # print(i)
    image= (cv2.imread(path+'/'+i,cv2.IMREAD_UNCHANGED))
    image=cv2.resize(image,(100,100))
    menuImages.append(image)

menuCount=len(menuImages)
print(menuCount)

menuChoice=-1
isImageSelected=False

faceDetector=FaceMeshDetector(maxFaces=2)

while True:
    success, cameraFeedImg = captureVideo.read()
    cameraFlipedImg = cv2.flip(cameraFeedImg, 1)

    wHeight, wWidth, wChannels = cameraFlipedImg.shape
    print(wHeight, wWidth, wChannels)

    x=0
    xIncrement=math.floor(wWidth/menuCount)
    print(xIncrement)

    handsDetector = detector.findHands(cameraFlipedImg, flipType=False)
    hands = handsDetector[0]
    cameraFlipedImg = handsDetector[1]

    try:
        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            indexFingerTop = lmList1[8]
            indexFingerBottom = lmList1[6]

            if(indexFingerTop[1]< xIncrement):
                 i=0 
                 while(xIncrement*i <= wWidth): 
                    if (indexFingerTop[0] <xIncrement*i): 
                        menuChoice=i-1 
                        isImageSelected=True 
                        break 
                    i=i+1 
            if(indexFingerTop[1] > indexFingerBottom[1]): 
                    isImageSelected=False 
            print(isImageSelected)

        cameraFlipedImg,faces=faceDetector.findFaceMesh(cameraFlipedImg,draw=False)
        try:
            for face in faces:
                xLoc=face[21][0]
                yLoc=face[21][1]

                if isImageSelected:
                    image=cv2.resize(menuImages[menuChoice],(100,100))
                    cameraFlipedImg=cvzone.overlayPNG(cameraFlipedImg,menuImages[menuChoice],[int(indexFingerTop[0]),int(indexFingerTop[1])])
                else:
                    #Face Width
                    distance = math.dist(face[21],face[251])

                    # Set the initial scale to 0
                    scale=90

                    #Creating dx,dy variables to fix the position of the filters
                    dx=0
                    dy=0

                    if (menuChoice==0):
                        scale=90
                        dx=5
                        dy=40
                    if (menuChoice==1):
                        scale=85
                        dx=5
                        dy=80
                    if (menuChoice==2):
                        scale=55
                        dx=20
                        dy=60
                    if (menuChoice==3):
                        scale=70
                        dx=15
                        dy=30
                    if (menuChoice==4):
                        scale=80
                        dx=10
                        dy=30

                    resizeFactor=distance/scale
                    xLoc=int(xLoc-(resizeFactor*dx))
                    yLoc=int(yLoc-(resizeFactor*dy))
                    filterImage=cv2.resize(menuImages[menuChoice],(100,100))
                    filterImage=cv2.resize(filterImage,(0,0), fx=resizeFactor, fy=resizeFactor)
                    cameraFlipedImg=cvzone.overlayPNG(cameraFlipedImg,filterImage,[xLoc,yLoc])
        except Exception as e:
            print(e)

        

    except Exception as e:
        print(e)

    try:
        for i in menuImages:
            margin=20
            image=cv2.resize(i,(xIncrement-margin,xIncrement-margin))
            cameraFlipedImg=cvzone.overlayPNG(cameraFlipedImg, image, [x,0])
            x=x+xIncrement
            
    except:
        print("Out of Bounds")

    cv2.imshow('My Fliped Picture', cameraFlipedImg)
    if cv2.waitKey(1) == 32:
        break

captureVideo.release()