import numpy as np
import cv2
import os
from GetHomography import Homography

class Merge:
    def __init__(self,PATH1,PATH2):
        self.Oriimg1 = cv2.resize(cv2.imread(PATH1),(512,512))
        self.Oriimg2 = cv2.resize(cv2.imread(PATH2),(512,512))
        self.img1 = cv2.cvtColor(cv2.resize(cv2.imread(PATH1),(512,512)), cv2.COLOR_BGR2GRAY)
        self.img2 = cv2.cvtColor(cv2.resize(cv2.imread(PATH2),(512,512)), cv2.COLOR_BGR2GRAY)

    def _getKeyPoint(self,img1,img2):
        SIFT = cv2.xfeatures2d.SIFT_create()
        keyPoints1 = SIFT.detect(img1,None)
        keyPoints2 = SIFT.detect(img2,None)
        keyPs1,Des1 = SIFT.compute(img1,keyPoints1)
        keyPs2,Des2 = SIFT.compute(img2,keyPoints2)
        return keyPs1,keyPs2,Des1,Des2
    def RotateImage(self,Homography,img):#使用反向映射+双线性插值——旋转图片
        Old_rows,Old_cols = img.shape
        H_INV = np.linalg.inv(Homography)
        Points = [(0,0),(Old_cols-1,0),(0,Old_rows-1),(Old_cols-1,Old_rows-1)]
        Pro_point = []
        for Value in Points:
            Value = np.array(Value+(1,)).reshape(3,1)
            temp = np.dot(Homography,Value)
            Pro_point.append(tuple(np.rint((temp[0]/temp[2],temp[1]/temp[2]))))
        X_max = 0
        Y_max = 0
        X_min = np.inf
        Y_min = np.inf
        print(Pro_point)
        for value in Pro_point:
            if X_max < value[0]:
                X_max = int(value[0])
            if Y_max < value[1]:
                Y_max = int(value[1])
            if X_min > value[0]:
                X_min = int(value[0])
            if Y_min > value[1]:
                Y_min = int(value[1])
        print(X_max,X_min,Y_max,Y_min)
        # X_max,X_min,Y_max,Y_min = map(int,(np.rint((X_max,X_min,Y_max,Y_min))))
        cols = np.int((X_max - X_min + 1))
        rows = np.int((Y_max - Y_min + 1))
        print("img1 shape")
        print(rows,cols)
        empty_img = np.zeros((rows,cols),dtype = np.uint8)
        num = 0
        for i in range(rows):
            for j in range(cols):
                temp_Point = (j+X_min,i+Y_min,1)
                Projection_point = np.dot(H_INV,temp_Point)
                Pro_x = Projection_point[0]/Projection_point[2]#反投影变换后的点
                Pro_y = Projection_point[1]/Projection_point[2]
                # print(Pro_x,Pro_y)
                INTER_x = np.int(np.ceil(Pro_x))#对坐标点进行上取整
                INTER_y = np.int(np.ceil(Pro_y))
                FLOOR_x = np.int(INTER_x - 1)#对坐标点进行下取整
                FLOOR_y = np.int(INTER_y - 1)
                
                if FLOOR_x >= Old_cols or FLOOR_x < 0 or FLOOR_y >= Old_rows or FLOOR_y < 0:
                    continue
                elif INTER_x >= Old_cols or INTER_x < 0 or INTER_y >= Old_rows or INTER_y < 0:
                    #empty_img[i,j] = img[FLOOR_x,FLOOR_y]
                    empty_img[i,j] = 0
                else:
                    Q11 = (FLOOR_y,FLOOR_x)
                    Q12 = (INTER_y,FLOOR_x)
                    Q21 = (FLOOR_y,INTER_x)
                    Q22 = (INTER_y,INTER_x)    
                    empty_img[i,j] = np.rint(img[Q11]*(INTER_x-Pro_x)*(INTER_y-Pro_y)+img[Q21]*(Pro_x-FLOOR_x)*(INTER_y-Pro_y) + img[Q12]*(INTER_x-Pro_x)*(Pro_y-FLOOR_y)+img[Q22]*(Pro_x-FLOOR_x)*(Pro_y-FLOOR_y))
        return Pro_point,(X_max,Y_max,X_min,Y_min),empty_img
    
    def getKeyPoint(self):
        return self._getKeyPoint(self.img1,self.img2)

    def Merge(self,img1,img2,point,Spe_point):
        rows_2,cols_2 = img2.shape#img2是未旋转的图像
        img2Point = [(0,0),(rows_2-1,0),(0,cols_2-1),(rows_2-1,cols_2-1)]
        TotalPoint = point + img2Point
        minX = np.inf
        minY = np.inf
        maxX = 0
        maxY = 0
        for valueX,valueY in TotalPoint:
            if minX > valueX:
                minX = valueX
            if minY > valueY:
                minY = valueY
            if maxX < valueX:
                maxX = valueX
            if maxY < valueY:
                maxY = valueY
        minX,minY,maxX,maxY = map(int,(minX,minY,maxX,maxY))
        print(minX,minY,maxX,maxY)
        shape_Rows = np.int(maxX - minX + 1)
        shape_Cols = np.int(maxY - minY + 1)
        print(shape_Rows,shape_Cols)
        new_Img = np.zeros((shape_Rows,shape_Cols),dtype = np.uint8)
        if shape_Rows > img1.shape[0]:
            k1 = 0
            k2 = 0
            if Spe_point[3] < 0:#ymin < 0 
                k1 = abs(Spe_point[3])
            if Spe_point[1] > rows_2 - 1:#ymax > rows_2 - 1
                k2 = abs(Spe_point[1] - rows_2 + 1)
            temp_img = np.zeros((shape_Rows,img1.shape[1]),dtype = np.uint8)
            temp_img[k1:shape_Rows-k2,:] = img1
            img1 = temp_img
        if shape_Rows > img2.shape[0]:
            k1 = 0
            k2 = 0
            if Spe_point[3] > 0:
                k1 = abs(Spe_point[2])
            if Spe_point[0] > rows_2 - 1:
                k2 = Spe_point[0] - rows_2 + 1
            temp_img = np.zeros((shape_Rows,img2.shape[1]),dtype = np.uint8)
            temp_img[k1:shape_Rows-k2] = img2
            img2 = temp_img
        new_Img[:,0:abs(minY)] = img1[:,0:abs(minY)]
        alpha = np.dot(np.ones((maxY+1,1)),np.linspace(0,maxY,maxY+1).reshape(1,maxY+1))/maxY
        new_Img[:,abs(minY):abs(minY)+maxY+1] = (1 - alpha) * img1[:,abs(minY):abs(minY)+maxY+1] + alpha * img2[:,0:maxY+1]
        new_Img[:,abs(minY)+maxY+1:] = img2[:,maxY + 1:]
        return new_Img
            
    def _MatchPoints(self,Kp1,Kp2,Des1,Des2):
        ratio = 0.5
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(Des1, Des2, 2)
        MatchRes = []
        for item in rawMatches:
            if len(item) == 2 and item[0].distance < ratio * item[1].distance:
                MatchRes.append((item[0].trainIdx,item[0].queryIdx))
        # if len(MatchRes) > 4:
        ptsA = [Kp1[i].pt for (_, i) in MatchRes]
        ptsB = [Kp2[i].pt for (i, _) in MatchRes]
        Find_Homography = Homography()
        Homo,in1,in2 = Find_Homography.GetHomography(ptsA,ptsB)
        (H,status) = cv2.findHomography(np.array(ptsA), np.array(ptsB), cv2.RANSAC,3.0)
        Homo = Homo / Homo[2,2]
        print(in1,in2)
        print(Homo)
        #print(H)
        Point,Spe_point,result_2 = self.RotateImage(H,self.img1)
        # # result = cv2.warpPerspective(self.img1, Homo,
        # # (cols,rows))
        newImg = self.Merge(result_2,self.img1,Point,Spe_point)
        cv2.imshow("ori",self.img1)
        # cv2.imshow("2",self.img2)
        # #cv2.imshow("Rotate",result)
        cv2.imshow("myRotate",result_2)
        # cv2.imshow("res",newImg)
        cv2.waitKey(0)

            
    def MatchPoints(self):
        keyPoints1,keyPoints2,Des1,Des2 = self.getKeyPoint()
        self._MatchPoints(keyPoints1,keyPoints2,Des1,Des2)

    


if __name__ == "__main__":
    Merge1 = Merge("../img/2.jpg","../img/1.jpg")
    Merge1.MatchPoints() 