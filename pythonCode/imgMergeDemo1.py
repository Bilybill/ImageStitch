import os
import cv2
import numpy as np
from GetHomography import Homography

PATH1 = "../img/1.jpg"
PATH2 = "../img/2.jpg"

class ImageMerge:
    def __init__(self):
        self.imglist = []
        self.Rimglist = []
        self.Gimglist = []
        self.Bimglist = []
        self.Limglist = []
        self.base = []
        self.Target = []
        self.x_dis = 0#水平重叠的像素点的个数
        self.y_dis = 0#垂直方向上的偏移
        # for i in range(len(self.listdir)):
        #     for image in self.listdir:
        #         if image[0] == str(i):
        self.imglist.append(cv2.resize(cv2.imread(PATH1),(512,512)))
        self.imglist.append(cv2.resize(cv2.imread(PATH2),(512,512)))
        for i in range(len(self.imglist)):
            self.Rimglist.append(self.imglist[i][:,:,0])
            self.Gimglist.append(self.imglist[i][:,:,1])
            self.Bimglist.append(self.imglist[i][:,:,2])
            self.Limglist.append(cv2.cvtColor(self.imglist[i],cv2.COLOR_BGR2GRAY))

    def dealWithSourceImage(self,img):
        col = 400
        b1 = np.array(img[50:450,col],dtype=float)
        b2 = np.array(img[50:450,col+20],dtype=float)
        if 0 in b2:
            listindex = np.where(b2 == 0)
            for i in listindex:
                b2[i] = 0.1 
        self.base = b1/b2

    def RotateImage(self,Homography,img):#使用反向映射+双线性插值——旋转图片
        Old_rows,Old_cols = img.shape
        H_INV = np.linalg.inv(Homography)
        Points = [(0,0),(Old_rows-1,0),(0,Old_cols-1),(Old_rows-1,Old_cols-1)]
        Pro_point = []
        for Value in Points:
            Value = np.array(Value+(1,)).reshape(3,1)
            temp = np.dot(Homography,Value)
            #print(temp)
            Pro_point.append((temp[0]/temp[2],temp[1]/temp[2]))
            map(tuple,Pro_point)
        X_max = 0
        Y_max = 0
        X_min = 0
        Y_min = 0
        for value in Pro_point:
            if X_max < value[0]:
                X_max = value[0]
            if Y_max < value[1]:
                Y_max = value[1]
            if X_min > value[0]:
                X_min = value[0]
            if Y_min > value[1]:
                Y_min = value[1]
        rows = np.int(np.rint(X_max - X_min))
        cols = np.int(np.rint(Y_max-Y_min))
        print(rows,cols)
        empty_img = np.zeros((rows,cols),dtype = np.uint8)
        num = 0
        for i in range(rows):
            for j in range(cols):
                temp_Point = (i+X_min,j+Y_min,1)
                Projection_point = np.dot(H_INV,temp_Point)
                Pro_x = Projection_point[0]/Projection_point[2]#反投影变换后的点
                Pro_y = Projection_point[1]/Projection_point[2]
                # print(Pro_x,Pro_y)
                num += 1
                INTER_x = np.int(np.ceil(Pro_x))
                INTER_y = np.int(np.ceil(Pro_y))
                FLOOR_x = np.int(INTER_x - 1)
                FLOOR_y = np.int(INTER_y - 1)
                if FLOOR_x >= Old_rows or FLOOR_x < 0 or FLOOR_y >= Old_cols or FLOOR_y < 0:
                    continue
                elif INTER_x >= Old_rows or INTER_x < 0 or INTER_y >= Old_cols or INTER_y < 0:
                    empty_img[i,j] = img[FLOOR_x,FLOOR_y]
                else:
                    Q11 = (FLOOR_x,FLOOR_y)
                    Q12 = (FLOOR_x,INTER_y)
                    Q21 = (INTER_x,FLOOR_y)
                    Q22 = (INTER_x,INTER_y)    
                    empty_img[i,j] = np.rint(img[Q11]*(INTER_x-Pro_x)*(INTER_y-Pro_y)+img[Q21]*(Pro_x-FLOOR_x)*(INTER_y-Pro_y) + img[Q12]*(INTER_x-Pro_x)*(Pro_y-FLOOR_y)+img[Q22]*(Pro_x-FLOOR_x)*(Pro_y-FLOOR_y))
        empty_img = cv2.resize(empty_img,(512,512))
        return empty_img
    
    def dealwithTargetImage(self,img):
        self.Target = []
        for col in range(0,450):
            b1 = np.array(img[0:500,col],dtype=float)
            b2 = np.array(img[0:500,col+20],dtype=float)
            if 0 in b2:
                listindex = np.where(b2 == 0)
                for i in listindex:
                    b2[i] = 0.001 
            self.Target.append(b1/b2)

    def getX_disAndY_dis(self):
        disReg = []
        minReg = []
        for num in range(450):
            sum_eq = 0
            epsi = []
            for dis in range(101):
                sum_eq = 0
                for i in range(400):    
                    sum_eq += np.square((self.Target[num][i+dis] - self.base[i]))
                epsi.append(sum_eq)
                sum_eq = 0
            disReg.append(epsi.index(min(epsi)))
            minReg.append(min(epsi))
        minIndex = minReg.index(min(minReg))
        disIndex = disReg[minIndex]
        self.x_dis = minIndex + 100
        print(self.x_dis)
        self.y_dis = disIndex
        if self.y_dis > 50:
            self.y_dis = 100 - self.y_dis
        elif self.y_dis > 20:
            self.y_dis = 6  
        # print("disIndex is "+str(disIndex))

    def MergeImage(self,img1,img2):
        Homo = np.array([[  1.45755222e+00   ,1.63962772e-02 , -3.22800863e+02],
 [  2.17501001e-01  , 1.32908655e+00 , -9.08129533e+01],
 [  8.59837661e-04  , 3.86790201e-05  , 1.00000000e+00]])
        img1 = self.RotateImage(Homo,img1)
        self.dealWithSourceImage(img1)
        self.dealwithTargetImage(img2)
        self.getX_disAndY_dis()
        print("y_dis is "+str(self.y_dis))
        x_temp = 512 -self.x_dis
        y_temp = 512 - self.y_dis
        new_img = img1[self.y_dis:,0:x_temp]
        # cv2.imshow('temp',new_img)
        for i in range(self.x_dis):
            # d = 1 - (i/self.x_dis)
            d = i/self.x_dis
            d = np.sqrt(1-d)
            # d = np.tan()
            # d = d*d
            # d = (i/self.x_dis)
            # d = 1 / (1 + np.exp((d - 1/2)*10))
            newLine = (d * img1[self.y_dis:,i+x_temp:i+x_temp+1] + (1 - d )*img2[0:y_temp:,i:i+1]).astype(int)
            new_img = np.hstack((new_img,newLine))
        new_img = np.hstack((new_img,img2[0:y_temp,self.x_dis:]))
        return new_img
 
if __name__ == '__main__':
    Merge1 = ImageMerge()
    newImg = Merge1.MergeImage(Merge1.Limglist[0],Merge1.Limglist[1])
    newImg = newImg.astype(np.uint8) 
    #newImg = cv2.equalizeHist(newImg)
    # blur = cv2.GaussianBlur(newImg, (5, 5), 2)
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # canny = cv2.Canny(blur,50,150)
    # newImg = cv2.blur(newImg,(3,3))
    cv2.imshow('test',newImg.astype(np.uint8))
    # cv2.imshow('res',res)
    # cv2.imshow('canny',canny)
    cv2.imshow('origin1',Merge1.imglist[0])
    cv2.imshow('origin2',Merge1.imglist[1])
    cv2.waitKey(0)
