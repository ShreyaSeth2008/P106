import cv2


# Create our body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray = cv2.cvtColor(cap, cv2.COLOR.BGR2GRAY)

    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiscale(gray, 1.1, 2)
    
    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(cap,(x,y),(x+w,y+h),(255,0,0),2)
        roi = cap[y:y+h, x:x+w]
        cv2.imwrite("walking.jpeg", roi)

    cv2.imshow('cap', cap)

    if cv2.waitKey(1) == 32: #32 is the Space Key'
        break

cap.release()
cv2.destroyAllWindows()
