import cv2
import numpy as np
cv2.waitKey(5000)

cap=cv2.VideoCapture(0)

ret,img1=cap.read()


img1=cv2.resize(img1,(600,500))
#cv2.imshow('original',img1)
cap.release()



img=cv2.GaussianBlur(img1,(5,5),0)
img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#cv2.imshow('hsv',img)
lower = np.array([0, 100, 80])
upper= np.array([10,255,255])
mask = cv2.inRange(img, lower, upper)
#cv2.imshow('masked',mask)
kernel=np.ones((27,15),np.uint8)
open=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
#cv2.imshow('open',open)
contours,hierarchy =cv2.findContours(open,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    c=contours[i]
    M=cv2.moments(c)
    if M['m00']!=0:
        cx=int(M['m10']/M['m00'])
        cy = int(M['m01'] / M['m00'])
        c_0=contours[0]
        x,y,w,h=cv2.boundingRect(contours[i])
        img_box=cv2.rectangle(img1,(x,y),(x+w,y+h),color=(255,255,255),thickness=2)
        cv2.circle(img1,(cx,cy),2,(0,255,0),-1)
        print("(",cx,",",cy,")")
#cv2.drawContours(img1,contours,-1,(0,255,0),3)
cv2.imshow('final',img1)
cv2.waitKey(0)
cv2.destroyAllWindows


