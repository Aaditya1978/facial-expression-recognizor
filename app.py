from flask import Flask, render_template, Response, request, jsonify
import numpy as np
import cv2
from model import FacialExpressionModel
import os
from PIL import Image
from io import StringIO


model = FacialExpressionModel("model.json", "model_weights.h5")

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
font = cv2.FONT_HERSHEY_SIMPLEX



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def camera(image):
    cap = Image.open(StringIO(image))
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
            
            
    for (x,y,w,h) in faces_detected:
        fc = gray_img[y:y+h, x:x+w]

        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
    
        cv2.putText(test_img, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),2)
    
        ret, jpeg = cv2.imencode('.jpg', test_img)
        jpeg = jpeg.tolist()
        return jpeg
           
    
@app.route('/prediction', methods=["POST"])
def prediction():
    if request.method == "POST":
        image = request.data
        face = camera(image)
        return jsonify(faces=face)

if __name__ == '__main__':
    app.run(debug=True)