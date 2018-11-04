import numpy as np
import cv2

def pyramid_img(img,levels = 3):
    temp = img.copy()
    pyramid_images = []
    temp = cv2.GaussianBlur(temp,(3,3),1.0)
    for i in range(levels):
        dst =   cv2.pyrDown(temp)
        pyramid_images.append(dst)
        #cv2.imshow("Gaussian tower"+str(i),dst)
        temp = dst.copy()
    return pyramid_images

def laplian_image(image):
    pyramid_images = pyramid_img(image)
    level = len(pyramid_images)
    lpl_img = []
    for i in range(level-1, -1, -1):
        if(i-1) < 0 :
            expand = cv2.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv2.subtract(image, expand)
            lpl_img.append(lpls)
        else:
            expand = cv2.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv2.subtract(pyramid_images[i-1], expand)
            lpl_img.append(lpls)
    return lpl_img
if __name__ == "__main__":
    src = cv2.resize(cv2.imread("../img/1.JPG"),(512,512))
    cv2.imshow("ori", src)
    laplian_image(src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()