import cv2 
import time 
import glob
from emailing import send_email

video = cv2.VideoCapture(0)
time.sleep(1)
first_frame = None
statuts_list = []
count = 1

while True:
    status = 0
    check, frames = video.read()
    gray_frames = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    gray_frames_gau = cv2.GaussianBlur(gray_frames,(21,21),0)
    
    if first_frame is None:
        first_frame = gray_frames_gau
    
    delta_frame = cv2.absdiff(first_frame,gray_frames_gau)
    thresh_frame = cv2.threshold(delta_frame,60,255,cv2.THRESH_BINARY)[1]
    dil_frame = cv2.dilate(thresh_frame,None,iterations=2)
    
    
    contours ,check = cv2.findContours(dil_frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < 10000:
            continue
        x,y,w,h = cv2.boundingRect(contour)
        rectangle =  cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0))
        if rectangle.any():
            status = 1
            cv2.imwrite(f"images/{count}.png",frames)
            count = count+1
            all_images = glob.glob("images/*.png")
            index = int(len(all_images)/2)
            image_obj = all_images[index]
            
    statuts_list.append(status)
    statuts_list = statuts_list[-2:]
    if statuts_list[0] == 1 and statuts_list[1] == 0:
            send_email()
    
    cv2.imshow("video",frames)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()