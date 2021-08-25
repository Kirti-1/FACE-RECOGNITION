#import the required modules or libraries
import cv2
import numpy as np

# initialize web cam
cap = cv2.VideoCapture(0)

# creating haarcascasde classfier object 
path = "haarcascade/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(path)

face_data = []
dataset_path = "./data/"
filename = input("Enter your name : ") 

while True:
    
    ret, frame = cap.read()
    # checking whether the frame is extracted properly or not
    if ret == False:
        continue
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # haarcascade frontal face classifier - pre trained classifier on convolutional neural network
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5);

    faces = sorted(faces, key = lambda f:f[2]*f[3])

    for face in faces[-1:]:
        x,y,w,h = face
        # detect and create bounding box around the face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2);
        # extract or crop out the required face : region of interest
        offset = 10
        face_section = gray_frame[y-offset:y+offset+h, x-offset:x+offset+w]
        # flatten the face section part and then store it
        face_section = cv2.resize(face_section, (100,100))
        face_data.append(face_section)
    

    cv2.imshow("Frame", frame);
    # cv2.imshow("Gray Frame", gray_frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if(key_pressed == ord('q')):
        break

# convert the datalist into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))

# save the file with the filename included
np.save(dataset_path + filename + ".npy", face_data);
print("Data saved Successfully!")
cap.release()
cv2.destroyAllWindows()