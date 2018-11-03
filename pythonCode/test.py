import os

import cv2
import numpy as np

PATH = 'img'
imglist = [] 
def getsize():
    listdir = os.listdir(PATH)
    for image in listdir:
        imglist.append(cv2.imread(PATH + '/' + image))

getsize()
res = []
Rimage = []
Gimage = []
Bimage = []
for i in range(len(imglist)):
    res.append(cv2.resize(imglist[i],(500,500)))
    Rimage.append(res[i][:,:,0])
    Gimage.append(res[i][:,:,1])
    Bimage.append(res[i][:,:,2])
# img2 = res[0][:,249:500]
# cv2.imshow('test',img2)
# cv2.imshow('ori',res[0])
# cv2.waitKey(0)
base = []
def dealwithSourceimage(img):
    for col in range(249,480):
        b1 = np.array(img[100:400,col],dtype=float)
        b2 = np.array(img[100:400,col+20],dtype=float)
        if 0 in b2:
            listindex = np.where(b2 == 0)
            for i in listindex:
                b2[i] = 0.1 
        base.append(b1/b2)

dealwithSourceimage(Gimage[8])
base2 = []
# dealwithSourceimage
# print(len(base[0]))
def dealwithTargetimage(img):
    for col in range(0,231):
        b1 = np.array(img[50:450,col],dtype=float)
        b2 = np.array(img[50:450,col+20],dtype=float)
        if 0 in b2:
            listindex = np.where(b2 == 0)
            for i in listindex:
                b2[i] = 0.1 
        base2.append(b1/b2)   
dealwithTargetimage(Gimage[6])
minReg = []
epsi = []
disReg = []
sum_eq = 0
for num in range(231):
    sum_eq = 0
    epsi = []
    for dis in range(101):
        for i in range(300):    
            sum_eq += (base2[num][i+dis] - base[220][i])*(base2[num][i+dis] - base[220][i])
        epsi.append(sum_eq)
        sum_eq = 0
    # if num == 0:
    #     print(epsi)
    disReg.append(epsi.index(min(epsi)))
    minReg.append(min(epsi))
minIndex = minReg.index(min(minReg))
disIndex = disReg[minIndex]
# print(disReg)
print(minIndex)
print(disIndex)
x_range = minIndex + 30
dis = 100 - 50 - disIndex
def merge(img1,img2):
    x_temp = 500 - x_range
    new_img = img1[0:482,0:(x_temp)]#(482,242)

    print(new_img.shape)
    # d = 0.1   
    # i = 0
    # newLine = d * img1[0:482,i+x_temp:i+x_temp+1] + (1 - d )*img2[18:,i:i+1] 
    # # newLine[0][0].astype(int)
    # newLine = img1[0:482,i+x_range:i+x_range+1]
    # print(newLine.shape)
    # # # print(newLine)
    # # newLine = newLine.astype(int)
    # new_img = np.hstack((new_img,newLine))
    # print(new_img.shape)
    # print(new_img[:,x_range])
    # new_img = new_img[:,0:x_range]
    # if (new_img == img1[0:482,0:x_range]):
    #     print('y')
    # else:
    #     print('n')
    for i in range(x_range):
        d = 1 - i/x_range
        d = d * d
        newLine = (d * img1[0:482,i+x_temp:i+x_temp+1] + (1 - d )*img2[18:,i:i+1]).astype(int)
        newLine = newLine.astype(int)
        new_img = np.hstack((new_img,newLine))
    print("loop_end"+str(new_img.shape))
    new_img = np.hstack((new_img,img2[18:,x_range:])) 
    return new_img
def mergetest(img1,img2):
    new1 = np.concatenate((img1,img2),axis=1)
    cv2.imshow('new',new1)
    cv2.waitKey(0)

new_img = merge(Gimage[8],Gimage[6])    
# print(new_img.shape)
cv2.imshow('origin1',Rimage[8])
cv2.imshow('origin2',Rimage[6])
cv2.imshow('test',new_img.astype(np.uint8))
cv2.imwrite('haha.jpg',new_img)
cv2.waitKey(0)
