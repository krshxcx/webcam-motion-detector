import cv2 
import time 

# Access the webcam
video = cv2.VideoCapture(0)

# Allow the camera to warm up for a second
time.sleep(1)

# Initialize the variable to store the first frame
first_frame = None

# Loop to continuously process frames from the webcam
while True:
    # Read a frame from the video feed
    check, frames = video.read()
    
    # Convert the frame to grayscale
    gray_frames = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gray_frames_gau = cv2.GaussianBlur(gray_frames, (21, 21), 0)
    
    # Set the first frame as the reference frame
    if first_frame is None:
        first_frame = gray_frames_gau
    
    # Calculate the difference between the current frame and the reference frame
    delta_frame = cv2.absdiff(first_frame, gray_frames_gau)
    
    # Create a threshold frame to highlight the differences
    thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate the thresholded frame to enhance areas of motion
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)
    
    # Display the processed frame with detected motion
    cv2.imshow('my video', dil_frame)
    
    # Find contours in the thresholded frame
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop through the detected contours
    for contour in contours:
        # If the contour area is smaller than a threshold, ignore it
        if cv2.contourArea(contour) < 10000:
            continue
        
        # Get the bounding rectangle for significant contours and draw a rectangle around them
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0))
    
    # Display the original frame with rectangles around detected motion
    cv2.imshow("video", frames)
    
    # Check for the 'q' key press to exit the loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video object and close windows
video.release()
cv2.destroyAllWindows()
