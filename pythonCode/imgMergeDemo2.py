import numpy as np
import cv2
import os
from GetHomography import Homography
import pyramid as pyr

class Merge:
    def __init__(self,PATH_list):
        self.imglist = []
        # self.Rimglist = []
        # self.Gimglist = []
        # self.Bimglist = []
        # self.Limglist = []
        for filepath in PATH_list:
            self.imglist.append(cv2.resize(cv2.imread(filepath),(512,512)))
        # for i in range(len(self.imglist)):
        #     self.Rimglist.append(cv2.equalizeHist(self.imglist[i][:,:,0]))
        #     self.Gimglist.append(cv2.equalizeHist(self.imglist[i][:,:,1]))
        #     self.Bimglist.append(cv2.equalizeHist(self.imglist[i][:,:,2]))
        #     self.Limglist.append(cv2.cvtColor(self.imglist[i],cv2.COLOR_BGR2GRAY))

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
        X_max = -np.inf
        Y_max = -np.inf
        X_min = np.inf
        Y_min = np.inf
        for value in Pro_point:
            if X_max < value[0]:
                X_max = int(value[0])
            if Y_max < value[1]:
                Y_max = int(value[1])
            if X_min > value[0]: 
                X_min = int(value[0])
            if Y_min > value[1]:
                Y_min = int(value[1])
        # X_max,X_min,Y_max,Y_min = map(int,(np.rint((X_max,X_min,Y_max,Y_min))))
        cols = np.int((X_max - X_min + 1))
        rows = np.int((Y_max - Y_min + 1))
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

    def Merge(self,RotateImage,NotRotateImage,point,Spe_point):
        rows_2,cols_2 = NotRotateImage.shape#NotRotateImage是未旋转的图像
        NotRotateImagePoint = [(0,0),(cols_2-1,0),(0,rows_2-1),(cols_2-1,rows_2-1)]
        TotalPoint = point + NotRotateImagePoint
        minX = np.inf
        minY = np.inf
        maxX = -np.inf
        maxY = -np.inf
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
        shape_Rows = np.int(maxY - minY + 1)
        shape_Cols = np.int(maxX - minX + 1)
        new_Img = np.zeros((shape_Rows,shape_Cols),dtype = np.uint8)
        Temp_img1 = NotRotateImage.copy()
        Temp_img2 = RotateImage.copy()
        
        if shape_Rows > RotateImage.shape[0]:
            k1 = 0
            k2 = 0
            if Spe_point[3] > 0:#ymin > 0 
                k1 = abs(Spe_point[3])
            if Spe_point[1] < rows_2 - 1:#ymax < rows_2 - 1
                k2 = abs(Spe_point[1] - rows_2 + 1)
            temp_img = np.zeros((shape_Rows,RotateImage.shape[1]),dtype = np.uint8)
            temp_img[k1:shape_Rows-k2,:] = RotateImage
            Temp_img2 = temp_img
        if shape_Rows > NotRotateImage.shape[0]:
            k1 = 0
            k2 = 0
            if Spe_point[3] < 0: #ymin < 0
                k1 = abs(Spe_point[3])
            if Spe_point[1] > rows_2 - 1:#ymax > rows_2 - 1
                k2 = abs(rows_2 - 1 - Spe_point[1])
            temp_img = np.zeros((shape_Rows,NotRotateImage.shape[1]),dtype = np.uint8)
            temp_img[k1:shape_Rows-k2] = NotRotateImage
            Temp_img1 = temp_img
        #Temp_img1 是未旋转的图片
        #Temp_img2 是旋转的图片
        if Spe_point[2] != minX:
            new_Img[:,0:abs(Spe_point[2])] = Temp_img1[:,0:abs(Spe_point[2])]
            if Spe_point[0] == maxX:
                Len = Temp_img1.shape[1] - abs(Spe_point[2])
                alpha = np.dot(np.ones((shape_Rows,1)),np.linspace(0,Len-1,Len).reshape(1,Len))/(Len-1)
                alpha = (1 - np.power(alpha,5))/(np.power(alpha,5)+(1 - np.power(alpha,5)))
                matrix = (Temp_img1[:,abs(Spe_point[2]):abs(Spe_point[2])+Len] == 0) 
                beta = alpha * matrix
                beta = beta + (1-matrix)
                beta = (beta == 0) + beta
                alpha = alpha / beta
                matrix = (Temp_img2[:,0:Len] == 0)
                alpha = alpha * (1 - matrix)
                new_Img[:,abs(Spe_point[2]):abs(Spe_point[2])+Len] = (1 - alpha) * Temp_img1[:,abs(Spe_point[2]):abs(Spe_point[2])+Len] + (alpha) * Temp_img2[:,0:Len]
                new_Img[:,abs(Spe_point[2])+Len:] = Temp_img2[:,Len:]
                return new_Img
            else:
                Len = Spe_point[0] - abs(Spe_point[2]) + 1
                alpha = np.dot(np.ones((shape_Rows,1)),np.linspace(0,Len-1,Len).reshape(1,Len))/(Len-1)
                alpha = (1 - np.power(alpha,5))/(np.power(alpha,5)+(1 - np.power(alpha,5)))
                matrix = (Temp_img1[:,abs(Spe_point[2]):abs(Spe_point[2])+Len] == 0)
                beta = alpha * matrix
                beta = beta + (1-matrix)
                beta = (beta == 0) + beta
                alpha = alpha / beta
                matrix = (Temp_img2[:,0:Len] == 0)
                alpha = alpha * (1 - matrix)
                new_Img[:,abs(Spe_point[2]):abs(Spe_point[2])+Len] = (1 - alpha) * Temp_img1[:,abs(Spe_point[2]):abs(Spe_point[2])+Len] + (alpha) * Temp_img2[:,0:Len]
                new_Img[:,abs(Spe_point[2])+Len:] = Temp_img1[:,abs(Spe_point[2])+Len:]
                return new_Img
        elif Spe_point[2] == minX:
            new_Img[:,0:abs(Spe_point[2])] = Temp_img2[:,0:abs(Spe_point[2])]
            if Spe_point[0] != maxX:    
                Len = Temp_img2.shape[1] - abs(Spe_point[2])
                alpha = np.dot(np.ones((shape_Rows,1)),np.linspace(0,Len-1,Len).reshape(1,Len))/(Len-1)
                alpha = (1 - np.power(alpha,5))/(np.power(alpha,5)+(1 - np.power(alpha,5)))
                matrix = (Temp_img2[:,abs(Spe_point[2]):abs(Spe_point[2])+Len] == 0) 
                beta = alpha * matrix
                beta = beta + (1-matrix)
                beta = (beta == 0) + beta
                alpha = alpha / beta
                matrix = (Temp_img1[:,0:Len] == 0)
                alpha = alpha * (1 - matrix)
                new_Img[:,abs(Spe_point[2]):abs(Spe_point[2])+Len] = (1 - alpha) * Temp_img2[:,abs(Spe_point[2]):abs(Spe_point[2])+Len] + (alpha) * Temp_img1[:,0:Len]
                new_Img[:,abs(Spe_point[2])+Len:] = Temp_img1[:,Len:]
                return new_Img
            else:
                Len = Temp_img1.shape[1] - abs(Spe_point[2])
                alpha = np.dot(np.ones((shape_Rows,1)),np.linspace(0,Len-1,Len).reshape(1,Len))/(Len-1)
                alpha = (1 - np.power(alpha,5))/(np.power(alpha,5)+(1 - np.power(alpha,5)))
                matrix = (Temp_img2[:,abs(Spe_point[2]):abs(Spe_point[2])+Len] == 0)
                beta = alpha * matrix
                beta = beta + (1-matrix)
                beta = (beta == 0) + beta
                alpha = alpha / beta
                matrix = (Temp_img1[:,0:Len] == 0)
                alpha = alpha*(1-matrix)
                new_Img[:,abs(Spe_point[2]):abs(Spe_point[2])+Len] = (1 - alpha) * Temp_img2[:,abs(Spe_point[2]):abs(Spe_point[2])+Len] + (alpha) * Temp_img1[:,0:Len]
                new_Img[:,abs(Spe_point[2])+Len:] = Temp_img2[:,abs(Spe_point[2])+Len:]
                return new_Img 
    def MergeNewAlg(self,RotateImage,NotRotateImage,point,Spe_point):#最佳拼接线算法
        rows_2,cols_2 = NotRotateImage.shape#NotRotateImage是未旋转的图像
        NotRotateImagePoint = [(0,0),(cols_2-1,0),(0,rows_2-1),(cols_2-1,rows_2-1)]
        TotalPoint = point + NotRotateImagePoint
        minX = np.inf
        minY = np.inf
        maxX = -np.inf
        maxY = -np.inf
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
        shape_Rows = np.int(maxY - minY + 1)
        shape_Cols = np.int(maxX - minX + 1)
        Temp_img1 = NotRotateImage.copy()
        Temp_img2 = RotateImage.copy()
        #Temp_img1 是未旋转的图片
        #Temp_img2 是旋转的图片
        if shape_Rows > RotateImage.shape[0]:
            k1 = 0
            k2 = 0
            if Spe_point[3] > 0:#ymin > 0 
                k1 = abs(Spe_point[3])
            if Spe_point[1] < rows_2 - 1:#ymax < rows_2 - 1
                k2 = abs(Spe_point[1] - rows_2 + 1)
            temp_img = np.zeros((shape_Rows,RotateImage.shape[1]),dtype = np.uint8)
            temp_img[k1:shape_Rows-k2,:] = RotateImage
            Temp_img2 = temp_img
        if shape_Rows > NotRotateImage.shape[0]:
            k1 = 0
            k2 = 0
            if Spe_point[3] < 0: #ymin < 0
                k1 = abs(Spe_point[3])
            if Spe_point[1] > rows_2 - 1:#ymax > rows_2 - 1
                k2 = abs(rows_2 - 1 - Spe_point[1])
            temp_img = np.zeros((shape_Rows,NotRotateImage.shape[1]),dtype = np.uint8)
            temp_img[k1:shape_Rows-k2] = NotRotateImage
            Temp_img1 = temp_img
        start = 0
        end = 0
        if shape_Cols > Temp_img1.shape[1]:#未旋转图片
            temp_img = np.zeros((shape_Rows,shape_Cols),dtype = np.uint8)
            if Spe_point[2] != minX:
                temp_img[:,0:Temp_img1.shape[1]] = Temp_img1
                start = Spe_point[2]
                if Spe_point[0] != maxX:
                    end = Spe_point[0]+1
                else:
                    end = Temp_img1.shape[1]
            elif Spe_point[2] == minX and Spe_point[0] != maxX:
                temp_img[:,abs(minX):abs(minX) + Temp_img1.shape[1]] = Temp_img1
                start = 0
                end = Spe_point[0] + 1
            else:
                start = abs(Spe_point[2])
                end = abs(Spe_point[2])+Temp_img1.shape[1]
                temp_img[:,abs(Spe_point[2]):abs(Spe_point[2])+Temp_img1.shape[1]] = Temp_img1
            Temp_img1 = temp_img
        if shape_Cols > Temp_img2.shape[1]:
            temp_img = np.zeros((shape_Rows,shape_Cols),dtype = np.uint8)
            if Spe_point[2] != minX:
                temp_img[:,abs(Spe_point[2]):abs(Spe_point[2])+Temp_img2.shape[1]] = Temp_img2
                start = Spe_point[2]
                if Spe_point[0] != maxX:
                    end = Spe_point[0]+1
            elif Spe_point[2] == minX and Spe_point[0] != maxX:
                temp_img[:,0:Temp_img2.shape[1]] = Temp_img2
            else:
                temp_img[:,:] = Temp_img2
            Temp_img2 = temp_img
        if end - start > 50:
            start = start + 100
        mask1 = 1 - (Temp_img1 == 0)#有值的部分是1 无值的部分是0
        mask2 = 1 - (Temp_img2 == 0)
        Diff = (abs(Temp_img1.astype(np.int32)- Temp_img2.astype(np.int32))*mask1*mask2).astype(np.uint8)
        Diff = Diff[:,start:end]#截取到相交部分
        # cv2.imshow("DIff",Diff)
        # cv2.waitKey(0)
        Length = end - start
        Intensity = [int(Diff[0,i]) for i in range(Length)]
        Route = []
        for i in range(Length):
            Route.append([])
            Route[i].append(i)
        for i in range(Diff.shape[0] - 1):
            for j in range(Diff.shape[1]):
                m = 0
                if j == 0:
                    if Diff[i+1,j] < Diff[i+1,j+1]:
                        m = j+1
                    else:
                        m = j
                elif j == Diff.shape[1] - 1:
                    if Diff[i+1,j] < Diff[i+1,j-1]:
                        m = j - 1
                    else:
                        m = j
                else:
                    if Diff[i+1,j] < Diff[i+1,j+1]:
                        m = j + 1
                    else:
                        m = j
                    if Diff[i+1,m] < Diff[i+1,j-1]:
                        m = j - 1
                Intensity[j] = Intensity[j] + int(Diff[i+1,m])
                Route[j].append(m)
        #print(Intensity)
        minIndex = Intensity.index(min(Intensity))
        minRoute = np.array(Route[minIndex]) + start
        #print(minRoute)
        DiffGraph = np.zeros((shape_Rows,shape_Cols),dtype=np.uint8)
        for i in range(shape_Rows):
            DiffGraph[i,minRoute[i]:] = 1
        newImg = None
        H = None   
        if Spe_point[2] != minX and Spe_point[0] == maxX:#Temp_img1 在左边
            newImg = Temp_img1 * (1 - DiffGraph) + DiffGraph * Temp_img2 + ((mask1 | mask2) * DiffGraph * (1 - mask2) * Temp_img1 + (mask1 | mask2) * (1 - DiffGraph) * (1 - mask1) * Temp_img2).astype(np.uint8)
        elif Spe_point[2] == minX and Spe_point[0] != maxX:#Temp_img2 在左边
            newImg = Temp_img2 * (1 - DiffGraph) + DiffGraph * Temp_img1 + ((mask1 | mask2) * DiffGraph * (1 - mask1) * Temp_img2 + (mask1 | mask2) * (1 - DiffGraph) * (1 - mask2) * Temp_img1).astype(np.uint8)
        elif Spe_point == minX and Spe_point[0] == maxX:#Temp_img2 水平包含Temp_img
            newImg = Temp_img1 * (1 - DiffGraph) + DiffGraph * Temp_img2 + ((mask1 | mask2) * DiffGraph * (1 - mask2) * Temp_img1 + (mask1 | mask2) * (1 - DiffGraph) * (1 - mask1) * Temp_img2).astype(np.uint8)
        elif Spe_point != minX and Spe_point[0] != maxX:
            newImg = Temp_img1 * (1 - DiffGraph) + DiffGraph * Temp_img2 + ((mask1 | mask2) * DiffGraph * (1 - mask2) * Temp_img1 + (mask1 | mask2) * (1 - DiffGraph) * (1 - mask1) * Temp_img2).astype(np.uint8)
        return newImg   

    def MergerNewAnother(self,RotateImage,NotRotateImage,point,Spe_point):
        rows_2,cols_2 = NotRotateImage.shape#NotRotateImage是未旋转的图像
        NotRotateImagePoint = [(0,0),(cols_2-1,0),(0,rows_2-1),(cols_2-1,rows_2-1)]
        TotalPoint = point + NotRotateImagePoint
        minX = np.inf
        minY = np.inf
        maxX = -np.inf
        maxY = -np.inf
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
        shape_Rows = np.int(maxY - minY + 1)
        shape_Cols = np.int(maxX - minX + 1)
        Temp_img1 = NotRotateImage.copy()
        Temp_img2 = RotateImage.copy()
        #Temp_img1 是未旋转的图片
        #Temp_img2 是旋转的图片
        if shape_Rows > RotateImage.shape[0]:
            k1 = 0
            k2 = 0
            if Spe_point[3] > 0:#ymin > 0 
                k1 = abs(Spe_point[3])
            if Spe_point[1] < rows_2 - 1:#ymax < rows_2 - 1
                k2 = abs(Spe_point[1] - rows_2 + 1)
            temp_img = np.zeros((shape_Rows,RotateImage.shape[1]),dtype = np.uint8)
            temp_img[k1:shape_Rows-k2,:] = RotateImage
            Temp_img2 = temp_img
        if shape_Rows > NotRotateImage.shape[0]:
            k1 = 0
            k2 = 0
            if Spe_point[3] < 0: #ymin < 0
                k1 = abs(Spe_point[3])
            if Spe_point[1] > rows_2 - 1:#ymax > rows_2 - 1
                k2 = abs(rows_2 - 1 - Spe_point[1])
            temp_img = np.zeros((shape_Rows,NotRotateImage.shape[1]),dtype = np.uint8)
            temp_img[k1:shape_Rows-k2] = NotRotateImage
            Temp_img1 = temp_img
        start = 0
        end = 0
        if shape_Cols > Temp_img1.shape[1]:#未旋转图片
            temp_img = np.zeros((shape_Rows,shape_Cols),dtype = np.uint8)
            if Spe_point[2] != minX:
                temp_img[:,0:Temp_img1.shape[1]] = Temp_img1
                start = Spe_point[2]
                if Spe_point[0] != maxX:
                    end = Spe_point[0]+1
                else:
                    end = Temp_img1.shape[1]
            elif Spe_point[2] == minX and Spe_point[0] != maxX:
                temp_img[:,abs(minX):abs(minX) + Temp_img1.shape[1]] = Temp_img1
                start = 0
                end = Spe_point[0] + 1
            else:
                start = abs(Spe_point[2])
                end = abs(Spe_point[2])+Temp_img1.shape[1]
                temp_img[:,abs(Spe_point[2]):abs(Spe_point[2])+Temp_img1.shape[1]] = Temp_img1
            Temp_img1 = temp_img
        if shape_Cols > Temp_img2.shape[1]:
            temp_img = np.zeros((shape_Rows,shape_Cols),dtype = np.uint8)
            if Spe_point[2] != minX:
                temp_img[:,abs(Spe_point[2]):abs(Spe_point[2])+Temp_img2.shape[1]] = Temp_img2
                start = Spe_point[2]
                if Spe_point[0] != maxX:
                    end = Spe_point[0]+1
            elif Spe_point[2] == minX and Spe_point[0] != maxX:
                temp_img[:,0:Temp_img2.shape[1]] = Temp_img2
            else:
                temp_img[:,:] = Temp_img2
            Temp_img2 = temp_img
        if end - start > 50:
            start = start + 100
        mask1 = 1 - (Temp_img1 == 0)#有值的部分是1 无值的部分是0
        mask2 = 1 - (Temp_img2 == 0)
        Diff = (abs(Temp_img1.astype(np.int32)- Temp_img2.astype(np.int32))*mask1*mask2).astype(np.uint8)
        Diff = Diff[:,start:end]#截取到相交部分
        # cv2.imshow("DIff",Diff)
        # cv2.waitKey(0)
        Length = end - start
        Intensity = [int(Diff[0,i]) for i in range(Length)]
        Route = []
        for i in range(Length):
            Route.append([])
            Route[i].append(i)
        for i in range(Diff.shape[0] - 1):
            for j in range(Diff.shape[1]):
                m = 0
                if j == 0:
                    if Diff[i+1,j] < Diff[i+1,j+1]:
                        m = j+1
                    else:
                        m = j
                elif j == Diff.shape[1] - 1:
                    if Diff[i+1,j] < Diff[i+1,j-1]:
                        m = j - 1
                    else:
                        m = j
                else:
                    if Diff[i+1,j] < Diff[i+1,j+1]:
                        m = j + 1
                    else:
                        m = j
                    if Diff[i+1,m] < Diff[i+1,j-1]:
                        m = j - 1
                Intensity[j] = Intensity[j] + int(Diff[i+1,m])
                Route[j].append(m)
        #print(Intensity)
        minIndex = Intensity.index(min(Intensity))
        minRoute = np.array(Route[minIndex]) + start
        #print(minRoute)
        DiffGraph = np.zeros((shape_Rows,shape_Cols),dtype=np.uint8)
        for i in range(shape_Rows):
            DiffGraph[i,0:minRoute[i]] = 255
        
        # if Spe_point[2] == minX:
        
        newMatrix1 = (mask1|mask2)*(DiffGraph == 0)*(1-mask2)
        DiffGraph = DiffGraph + newMatrix1.astype(np.uint8)*255
        newMatrix2 = (mask1|mask2)*(1 - (DiffGraph==0))*(1-mask1)
        DiffGraph = (DiffGraph * (1 - newMatrix2)).astype(np.uint8)
        r = np.int(np.ceil(np.log2(shape_Rows)))
        s = np.int(np.ceil(np.log2(shape_Cols)))
        New_rows = np.power(2,r)#将行数和列数都换算为2的幂次
        New_cols = np.power(2,s)
        addRow = 0
        addCol = 0
        Temp_img1 = cv2.resize(Temp_img1,(New_cols,New_rows))
        Temp_img2 = cv2.resize(Temp_img2,(New_cols,New_rows))
        DiffGraph = cv2.resize(DiffGraph,(New_cols,New_rows))
        
        # if DiffGraph.any() != 0 and DiffGraph.any() != 255:
        #     raise ValueError("should not occur")
        # cv2.imshow("s",DiffGraph)
        # cv2.waitKey(0)
        #maskDiff = (DiffGraph != 255) and (DiffGraph != 0)


        La = pyr.laplian_image(Temp_img1)
        Lb = pyr.laplian_image(Temp_img2)
        GR = pyr.pyramid_img(DiffGraph)
        N = len(La)
        S = []
        for i in range(N):
            if i != N-1:   
                temp = ((GR[N-2-i]/255)*La[i] + ((255 - GR[N-2-i])/255)*Lb[i]).astype(np.uint8)
                S.append(temp)
            else:
                temp = ((DiffGraph/255)*La[i] + ((255 - DiffGraph)/255)*Lb[i]).astype(np.uint8)
                S.append(temp)
        newImg = None
        for i in range(len(S)-1):
            if i == 0:
                expand = cv2.pyrUp(S[i], dstsize = S[i+1].shape[:2])
                newImg = expand + S[i+1]
            else:
                expand = cv2.pyrUp(newImg,dstsize = S[i+1].shape[:2])
                newImg = expand + S[i+1]
        newImg = newImg.astype(np.uint8)
        newImg = cv2.resize(newImg,(shape_Cols,shape_Rows))
        return newImg    

    def _GetImgHist(self, img):
        rows, cols = img.shape
        Hist = np.zeros((256, 1), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                Hist[img[i, j]] += 1
        NUM = rows*cols
        Hist = Hist / NUM
        return Hist
    def equalHist(self, img):
        Hist_img = self._GetImgHist(img)
        rows,cols = img.shape
        S_k = np.zeros((256,1),dtype=np.uint8)
        for i in range(256):
            S_k[i] = np.rint(255 * sum(Hist_img[0:i]))
        for i in range(rows):
            for j in range(cols):
                img[i,j] = S_k[img[i,j]]
        return img

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
        return ptsA,ptsB

    def _StitchImage(self,img1,img2):#img2 rotate to img1 :like imglist[7]
        gray_img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        Bimg1,Gimg1,Rimg1 = cv2.split(img1)
        Bimg2,Gimg2,Rimg2 = cv2.split(img2)
        Bimg1,Gimg1,Rimg1 = map(cv2.equalizeHist,(Bimg1,Gimg1,Rimg1))
        Bimg2,Gimg2,Rimg2 = map(cv2.equalizeHist,(Bimg2,Gimg2,Rimg2))
        keyPoints1,keyPoints2,Des1,Des2 = self._getKeyPoint(gray_img1,gray_img2)
        ptsA,ptsB = self._MatchPoints(keyPoints1,keyPoints2,Des1,Des2)
        Find_Homography = Homography()
        Homo,in1,in2 = Find_Homography.GetHomography(ptsB,ptsA)#img2 --> img1 
        Homo = Homo / Homo[2,2]
        newBimg = self._MergeSingleChannle(Bimg1,Bimg2,Homo)
        newGimg = self._MergeSingleChannle(Gimg1,Gimg2,Homo)
        newRimg = self._MergeSingleChannle(Rimg1,Rimg2,Homo)
        newImg = cv2.merge((newBimg,newGimg,newRimg))
        return newImg
    def _MergeSingleChannle(self,img1,img2,Homo):
        Point,Spe_point,result = self.RotateImage(Homo,img2)
        #newImg = self.Merge(result,img1,Point,Spe_point)
        #newImg = self.MergeNewAlg(result,img1,Point,Spe_point)
        newImg = self.MergerNewAnother(result,img1,Point,Spe_point)
        return newImg
    


if __name__ == "__main__":
    PATH_list = []
    for i in range(9):
        PATH_list.append("../img/"+str(i+1)+".JPG")
    Merge1 = Merge(PATH_list)
    newImg = Merge1._StitchImage(Merge1.imglist[0],Merge1.imglist[1]) 
    newImg = Merge1._StitchImage(newImg,Merge1.imglist[2]) 
    newImg = Merge1._StitchImage(newImg,Merge1.imglist[6]) 
    newImg = Merge1._StitchImage(newImg,Merge1.imglist[8])
    newImg = Merge1._StitchImage(newImg,Merge1.imglist[3])
    newImg = Merge1._StitchImage(newImg,Merge1.imglist[4])
    newImg = Merge1._StitchImage(newImg,Merge1.imglist[5])
    #cv2.imshow('5th img',Merge1.imglis t[5])
    #newImg = Merge1._StitchImage(Merge1.imglist[0],Merge1.imglist[1]) 
    # newImg = Merge1._StitchImage(newImg,Merge1.imglist[2]) 
    # newImg = Merge1._StitchImage(newImg,Merge1.imglist[6]) 
    # newImg = Merge1._StitchImage(newImg,Merge1.imglist[7]) 
    # newImg = Merge1._StitchImage(newImg,Merge1.imglist[3])
    # newImg = Merge1._StitchImage(newImg,Merge1.imglist[4])
    # newImg = Merge1._StitchImage(newImg,Merge1.imglist[5])
    # newImg = Merge1._StitchImage(newImg,Merge1.imglist[8])
    # Bimg1,Gimg1,Rimg1 = cv2.split(newImg)
    # Bimg1,Gimg1,Rimg1 = map(cv2.equalizeHist,(Bimg1,Gimg1,Rimg1))
    # newImg2 = cv2.merge((Bimg1,Gimg1,Rimg1))
    # chazhi = newImg2 -newImg
    cv2.imshow("testAnothernewAlg.jpg",newImg.astype(np.uint8))
    cv2.waitKey(0)
    print("finish")