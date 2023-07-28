import numpy as np
import cv2

Cap= cv2.VideoCapture(0)
Img = cv2.imread("BOOK_IMAGE.jpg")

while True:
    Rec, Frame = Cap.read()
    
    #GRAY SCALE TO FRAMES AND IMAGE
    Gray_Frames=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    Gray_Img=cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    
    #NUMBERS OF POINTS TO DETECT
    Number=500
    #START THE FUNCTION .ORB_create(NUMBERS OF POINTS)
    Orb = cv2.ORB_create(Number)
    
    #DETECT THE POINTS WITH FUNCTION .detectAndComputer(Img, None )     CAUTION: IN THE VARIABLES OF THE BEGINNING ALWAYS PUT KEYPOINT AND DESCRIPTOR AS VARIABLES
    Keypoints1, descriptor1= Orb.detectAndCompute(Gray_Frames,None)
    Keypoints2, descriptor2= Orb.detectAndCompute(Gray_Img,None)
    
    print("DESCRIPTOR1",descriptor1)
    print("DESCRIPTOR2",descriptor2)
    
    #DRAW THE POINT DETECTED
    Frames_Display= cv2.drawKeypoints(Frame, Keypoints1, np.array([]), color=(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    Img_Display= cv2.drawKeypoints(Img, Keypoints2, np.array([]), color=(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    #SHOW THE POINTS IN SCREEN
    cv2.imshow("VIDEO CAPTURE",Frames_Display)
    cv2.imshow("IMAGE",Img_Display)
    
    
    t = cv2.waitKey(1)
    if t == 27:
        break
    
    
Cap.release()
cv2.destroyAllWindows()
    
    
    