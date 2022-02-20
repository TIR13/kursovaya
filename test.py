import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import cv2
import keras
import utils
from keras.models import Sequential, load_model

import sys

import numpy as np


path = "cell61.jpg"
model = load_model('mnist.h5')
image = cv2.imread(path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = cv2.GaussianBlur(gray_image, (3, 3), 1)

ret, thresh = cv2.threshold(gray_image.copy(), 150, 255, cv2.THRESH_BINARY_INV)


# thresh = thresh[20:-3,20:]
plt.imshow(thresh)

plt.show()

contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
preprocessed_digits = []


for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    # x+=20
    # y+=20
    print(cv2.arcLength(c, True))
    if cv2.arcLength(c, True) >= 200 and cv2.arcLength(c, True)<700:
        digit = thresh[y:y+h, x:x+w]
        resized_digit = cv2.resize(digit, (28,28))
        inp = np.array(resized_digit)

        res = model.predict([inp.reshape(1,28,28,1)])[0]

        print(np.argmax(res), res.round(3))
    
    # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
    cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
    
    # Cropping out the digit from the image corresponding to the current contours in the for loop

plt.imshow(resized_digit)

plt.show()
resized_digit = cv2.resize(thresh, (28,28))
inp = np.array(resized_digit)
plt.imshow(resized_digit)

plt.show()


res = model.predict([inp.reshape(1,28,28,1)])[0]

print(np.argmax(res), res.round(3))

