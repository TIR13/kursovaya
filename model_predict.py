import matplotlib.pyplot as plt
import cv2
import keras
import utils
from keras.models import Sequential, load_model

import sys

import numpy as np


def predict_cnn(path="test/cell61.jpg"):
    model = load_model('mnist.h5')
    image = cv2.imread(path)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 1)
    
    ret, thresh = cv2.threshold(gray_image.copy(), 170, 255, cv2.THRESH_BINARY_INV)
    
    
    # plt.imshow(thresh)
    
    # plt.show()
    test = thresh[20:-10, 20:-10]
    contours, _ = cv2.findContours(test, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    preprocessed_digits = []
    
    # plt.imshow(thresh[20:-10, 20:-10])
    
    # plt.show()
    
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
    
        
        if cv2.arcLength(c, True) >= 200 and cv2.arcLength(c, True)<700:
            # print(f"Length conturs: {cv2.arcLength(c, True)}")
            digit = test[y:y+h, x:x+w]
            resized_digit = cv2.resize(digit, (28,28))
            inp = np.array(resized_digit)
    
            res = model.predict([inp.reshape(1,28,28,1)])[0]
    
            # print(f"Predict: {np.argmax(res)}\n{res.round(3)}")
            print(f"Predict: {np.argmax(res)}")
            plt.imshow(resized_digit)
            
            plt.show()
        
        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        
        # Cropping out the digit from the image corresponding to the current contours in the for loop
    
    # resized_digit = cv2.resize(thresh, (28,28))
    inp = np.array(resized_digit)
    return np.argmax(res)

if __name__=="__main__":
    predict_cnn("test/cell0.jpg")
    predict_cnn("test/cell2.jpg")
    predict_cnn("test/cell5.jpg")
    predict_cnn("test/cell7.jpg")
    predict_cnn("test/cell10.jpg")
    predict_cnn("test/cell11.jpg")
    predict_cnn("test/cell12.jpg")
    predict_cnn("test/cell33.jpg")
    

