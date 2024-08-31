import cv2
import numpy as np

#applying perspective transform
img = cv2.imread('img.png')
pt1 = [1190,609]
pt2 =[1350,680]
pt3 =[1180,670]
pt4 =[1340,725]
cv2.circle(img,pt1,10,(0,255,0),cv2.FILLED)
cv2.circle(img,pt2,10,(0,255,0),cv2.FILLED)
cv2.circle(img,pt3,10,(0,255,0),cv2.FILLED)
cv2.circle(img,pt4,10,(0,255,0),cv2.FILLED)
height, width = 32,150
mat_pt = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(np.float32([pt1,pt2,pt3,pt4]),mat_pt)
transformed_img = cv2.warpPerspective(img,matrix,(width,height))
cv2.imshow('img',img)
cv2.imshow('transformed_img',transformed_img)
#save transformed image 
cv2.imwrite('transformed_img.png',transformed_img)
#height, width = 1073,1850
cv2.waitKey(0)