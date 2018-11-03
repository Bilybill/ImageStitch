import numpy as np
import cv2
import random

class Homography:
    def __init__(self):
        self.Homography = None
        self.dist = None
    def _GetSample(self,data1,data2):
        if len(data1) != len(data2) or len(data1) != 4:
            print("error data")
            raise ValueError("Not enough input data to fit the model.")
            return None
        A = np.zeros([8,9])
        for i in range(8):
            for j in range(9):
                if (i % 2 == 0 and j == 0) or (i % 2 == 1 and j == 3):
                    A[i][j] = data1[i//2][0]
                elif (i % 2 == 0 and j == 1) or (i % 2 == 1 and j == 4):
                    A[i][j] = data1[i//2][1]
                elif (i % 2 == 0 and j == 2) or (i % 2 == 1 and j == 5):
                    A[i][j] = 1
                elif (i % 2 == 0 and j == 6):
                    A[i][j] = 0 - data2[i//2][0]*data1[i//2][0]
                elif (i % 2 == 0 and j == 7):
                    A[i][j] = 0 - data2[i//2][0]*data1[i//2][1]
                elif (i % 2 == 0 and j == 8):
                    A[i][j] = 0 - data2[i//2][0]
                elif (i % 2 == 1 and j == 6):
                    A[i][j] = 0 - data2[i//2][1]*data1[i//2][0]
                elif (i % 2 == 1 and j == 7):
                    A[i][j] = 0 - data2[i//2][1]*data1[i//2][1]
                elif (i % 2 == 1 and j == 8):
                    A[i][j] = 0 - data2[i//2][1]
        # print("原始A")
        # print(A)
        A_new = np.dot(A.T,A)
        # print("A为")
        # print(A_new.shape)
        Lambda,Vec = np.linalg.eig(A_new)
        # print("特征值为")
        # print(Lambda)
        # print("特征向量为")
        # print(Vec)
        # print("index %d"%(np.argmin(Lambda)))
        # Ham = Vec[:,len(Lambda) - 1]
        Ham = Vec[:,np.argmin(Lambda)]
        # print(Ham)
        # print("验证")
        # print(np.dot(A,Ham))
        Ham = Ham.reshape([3,3])
        # print(Ham)
        # res = np.dot(Ham,np.array(data1[0]+(1,)).reshape(3,1))
        # print("验证第一个点")
        # res1 = res[0]/res[2]
        # res2 = res[1]/res[2]
        # Prop = np.array([res1,res2])
        # print(Prop)
        # Test = np.array(data2[0])
        # print(Test)
        # dist = np.sqrt(np.square(Prop[0]-Test[0])+np.square(Prop[1]-Test[1]))  
        # print("测试距离%lf"%(dist))
        return Ham
    def _RANSAC(self,vec1,vec2,min_samples,iterations=1000,eps=0.1,random_seed=44):
        random.seed(random_seed)
        if len(vec1) <= min_samples or len(vec2) <= min_samples:
            raise ValueError("Not enough input data to fit the model.")
        best_Hom = 0
        Best_Exist = False
        best_inliers = 0 
        best_iteration = None
        best_inliers1 = None
        best_inliers2 = None
        for i in range(iterations):
            Num_inliers = 0
            indices = list(range(len(vec1)))
            random.shuffle(indices)
            inLiers1 = [vec1[i] for i in indices[:min_samples]]
            inLiers2 = [vec2[i] for i in indices[:min_samples]]
            Shuffled_data1 = [vec1[i] for i in indices[min_samples:]]
            Shuffled_data2 = [vec2[i] for i in indices[min_samples:]]
            try:
                Hom = self._GetSample(inLiers1,inLiers2)
                for index in range(len(Shuffled_data1)):
                    I = (1,)
                    Projection_Point = np.array(Shuffled_data2[index])
                    # print("投影点")
                    # print(Projection_Point)
                    Original_Point =  np.array(Shuffled_data1[index]+I).reshape(3,1)
                    temp = np.dot(Hom,Original_Point)
                    Pro_p = np.array([ temp[0]/temp[2] , temp[1]/temp[2] ])
                    # print("计算点")
                    # print(Pro_p)
                    dist = np.sqrt(np.square(Pro_p[0]-Projection_Point[0])+np.square(Pro_p[1]-Projection_Point[1]))
                    # print("第%d轮第%d个匹配点的距离为%lf"%(i,index,dist))
                    if dist < eps:
                        Num_inliers += 1
                if Num_inliers > 0:
                    if Num_inliers > best_inliers:
                        best_inliers = Num_inliers
                        best_Hom = Hom
                        best_iteration = i
                        Best_Exist = True
                        best_inliers1 = inLiers1
                        best_inliers2 = inLiers2
            except ValueError as e:
                print(e)
        if Best_Exist == False:
            raise ValueError("CANNOT FIND A GOOD HOMOGRAPHY")
        else:
            # print(best_inliers,best_iteration)
            return best_Hom,best_inliers1,best_inliers2
    def GetHomography(self,vec1,vec2):
        return self._RANSAC(vec1,vec2,4)