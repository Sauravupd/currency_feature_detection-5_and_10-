import cv2
import numpy as np
import os


path = 'training'
orb = cv2.ORB_create(nfeatures=1000)

images = []
classname = []
mylist = os.listdir(path)
for image in mylist:
    currImg = cv2.imread(f'{path}/{image}',0)
    images.append(currImg)
    classname.append(os.path.splitext(image)[0])

print(classname)

def findDes(image):
    deslist=[]
    for i in images:
        kp,des = orb.detectAndCompute(i,None)
        deslist.append(des)
    return deslist

deslist = findDes(images)

def findID(img,deslist):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchlist = []
    finalval = -1
    for des in deslist:
        matches = bf.knnMatch(des,des2,k=2)
        accurate = []
        for m,n in matches:
           if m.distance < 0.75*n.distance:
               accurate.append([m])
        matchlist.append(len(accurate))
    finalval = matchlist.index(max(matchlist))
    
    return finalval

cap = cv2.VideoCapture(0)

while True:
    success,img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    id = findID(img2,deslist)
    

    if id!=-1:
        cv2.putText(imgOriginal,classname[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
    cv2.imshow('img2',imgOriginal)


    cv2.waitKey(1)



# img1 = cv2.imread('training/10.jpg',0)
# img2 = cv2.imread('query/rs10.jpg',0)

# orb = cv2.ORB_create(nfeatures=1000)

# kp1,des1 = orb.detectAndCompute(img1, None)
# kp2,des2 = orb.detectAndCompute(img2, None)

# imgKp1 = cv2.drawKeypoints(img1,kp1,None)
# imgKp2 = cv2.drawKeypoints(img2,kp2,None)

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)

# accurate = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         accurate.append([m])

# print(len(accurate))



# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)

# cv2.imshow('img1',img1)
# cv2.imshow('img2',img2)
# cv2.imshow('img3',img3)




# cv2.waitKey(0)