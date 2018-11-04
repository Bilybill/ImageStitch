import numpy as np
import cv2

def pyramid_img(img,levels = 5):
    temp = img.copy()
    pyramid_images = []
    #temp = cv2.GaussianBlur(temp,(5,5),1.0)
    for i in range(levels):
        dst = cv2.pyrDown(temp)
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
            #expand = cv2.resize(pyramid_images[i], (image.shape[1],image.shape[0]),interpolation=cv2.INTER_LINEAR)
            expand = cv2.pyrUp(pyramid_images[i], image.shape[:2])
            lpls = image - expand
            lpl_img.append(lpls)
        else:
            if i == level - 1:
                lpl_img.append(pyramid_images[i])
            #expand = cv2.resize(pyramid_images[i], (pyramid_images[i-1].shape[1],pyramid_images[i-1].shape[0]),interpolation=cv2.INTER_LINEAR)
            expand = cv2.pyrUp(pyramid_images[i], pyramid_images[i-1][:2])
            lpls = pyramid_images[i-1] - expand
            lpl_img.append(lpls)
    return lpl_img

if __name__ == "__main__":
    src = cv2.resize(cv2.imread("../img/1.JPG"),(1024,1024))
    src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    cv2.imshow("ori", src)
    lap = laplian_image(src)
    tmp = None
    for i in range(len(lap)-1):
        if i == 0:
            expand = cv2.pyrUp(lap[i], dstsize = lap[i+1].shape[:2])
            tmp = expand + lap[i+1]
        else:
            expand = cv2.pyrUp(tmp,dstsize = lap[i+1].shape[:2])
            tmp = expand + lap[i+1]
    #tmp = src|tmp
    cv2.imshow("test",tmp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()