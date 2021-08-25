# Recognise faces using some classification algorithm - like logistic, KNN, SVM etc

# 1. load the training data (numpy arrays of all the persons)
    # x-values are stored in the numpy arrays
    # y-values we need to assign for each person
# 2. Read a video stream using openCV 
# 3. extract faces out of it
# 4. use knn to find the prediction of face(int)
# 5. map the predicted id to name of the user
# 6. Display the predictions on the screen - bounding box and name

import numpy as np
import cv2
import os

############ KNN ##############
def distance(d1,d2):
	return np.sqrt(((d1-d2)**2).sum())

def knn(train,test,k=5):
	dist = []

	for i in range(train.shape[0]):
		ix = train[:,:-1]
		iy = train[i,-1]

		d = distance(test,ix)
		dist.append([d,iy])

	dk = sorted(dist , key = lambda x : x[0])[:k]

	labels = np.array(dk)[:,-1]

	output = np.unique(labels,return_counts = True)

	index = np.argmax(output[1])

	return output[0][index]

########## Knn Ends #############

#initialise webcam

cap = cv2.VideoCapture(0)

#face Detection
path = "haarcascade/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(path)

#data preparation

class_id = 0 #labels for given file
names = {} #mapping id with name



dataset_path = "./data/"
face_data = []
labels = []


for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		#create a mapping bte class_id and name
		names[class_id] = fx[:-4]
		print("Loaded " + fx)
		data_item = np.load(dataset_path+fx, allow_pickle=True)
		face_data.append(data_item)

		#create labels for the class
		target = class_id*np.ones((data_item.shape[0],))

		class_id+=1
		labels.append(target)


face_dataset = np.concatenate(face_data, axis = 0)
face_labels = np.concatenate(labels, axis = 0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

train_set = np.concatenate((face_dataset, face_labels), axis = 1)


print(train_set.shape)

#testing

while True:
	ret , frame = cap.read()

	if(ret == False):
		continue

	#gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


	faces = face_cascade.detectMultiScale(frame,1.3,5)

	if (len(faces)) == 0:
		continue

	
	# pick the last face(largest area)

	for face in faces:
		#draw bounding box
		x,y,w,h = face

		#extract (crop out face) the region of interest
		offset = 10
		face_section = frame[y-offset:y+h+offset, x-offset : x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		# prediction of the label
		out = knn(train_set,face_section.flatten())

		#display output on screen
		pred_name = names[int(out)]
		cv2.putText(frame, pred_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 2, cv2.LINE_AA)

		cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,), 2)

		
		
		


	cv2.imshow("Faces" , frame)
	#cv2.imshow("gray_frame" , gray_frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()