
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import cv2
import winsound
import numpy as np
frequency = 2500
duration = 1000

model = tf.keras.models.load_model('nice_model (1).h5')

img = cv2.imread('download (4).jfif')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('cant open webcam')
counter = 0
while True:
    ret,frame = cap.read()
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    eyes = eye_cascade.detectMultiScale(gray, 1.1,4)
    for x, y, w, h, in eyes:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        eyess = eye_cascade.detectMultiScale(roi_gray)
        if len(eyess) ==0:
            print('eyes are not detected')
        else:
            for (ex, ey, ew, eh) in eyess:
                eyes_roi = roi_color[ey: ey+eh, ex:ex + ew]

    final_img = cv2.resize(eyes_roi, (224, 224))
    final_img = np.expand_dims(final_img, axis=0)
    final_img = final_img/225.0                
            
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(face_cascade.empty())
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    #drawa rectangle around the face
    
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
            
    


    prediction = model.predict(final_img)
    if (prediction>0):
        status = "open eyes"
        cv2.putText(frame, status, (150, 150), font, 3, (0, 255, 0), 2, cv2.LINE_4)
        x1,y1,w1,h1 = 0,0,175,75
    #rectangle
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
        cv2.putText(frame, 'Active', (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
    
    

    else:
        counter = counter +1
        status = 'closed eyes'
        cv2.putText(frame,
                    status,
                    (150, 150),
                    font, 3,
                    (0, 255, 0),
                    2,
                    cv2.LINE_4)
        x1,y1,w1,h1 = 0,0,175,75
        
    #rectangle
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
        if counter >5:

            x1, y1, w1, h1, = 0,0,175,75
    #rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
            cv2.putText(frame, 'sleep alert !!', (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)  
            winsound.Beep(frequency, duration)
            counter = 0

            
            
    cv2.imshow('drowsiness detection system', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllindows()
