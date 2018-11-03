import numpy as np 
import cv2

img1 = cv2.imread("img/1.jpg")
img2 = cv2.imread("img/2.jpg")
img1 = cv2.resize(img1,(500,500))
img2 = cv2.resize(img2,(500,500))
stitch = cv2.createStitcher(False)
status,res = stitch.stitch((img1,img2))
print(status)
cv2.imshow('res',res)
cv2.waitKey(0)