import numpy as np
import cv2

def mySobel(img):
    Ma1= np.array([[-2, -1, -2], [0, 0, 0], [2, 1, 2]])#Y
    Ma2 = Ma1.T#X
    rows, cols = img.shape
    new_img = None
    k = 1
    n = 3
    Sobel1 = np.reshape(Ma1,(1,n*n))
    Sobel2 = np.reshape(Ma2,(1,n*n))
    new_img = img.copy()
    for i in range(k, rows-k):
        for j in range(k, cols-k):
            temp_matrix = img[i-k:i-k+n, j-k:j-k+n].flatten()
            X = np.dot(Sobel2,temp_matrix)
            Y = np.dot(Sobel1,temp_matrix)
            res = (np.square(X)+np.square(Y))
            new_img[i,j] = res
    return new_img