# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 08:56:02 2021

@author: Адм
"""

from PIL import Image
import pytesseract
import cv2
import subprocess as s
import os
from pytesseract import Output
#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

image = cv2.imread ('test.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
filtered = cv2.adaptiveThreshold(~gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)
cv2.imshow('threshold image', threshold_img)
# Maintain output window until user presses a key
#cv2.waitKey(0)

# Destroying present windows on screen

#cv2.destroyAllWindows()

custom_config = '--oem 3 --psm 6'

# now feeding image to tesseract
 
details = pytesseract.image_to_data(threshold_img,output_type=Output.DICT, config=custom_config, lang='rus')
print(details)
#print(details.keys())

total_boxes = len(details['text'])

for sequence_number in range(total_boxes):

	if int(details['conf'][sequence_number]) >30:

 		(x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])

 		threshold_img = cv2.rectangle(threshold_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# display image

#cv2.imshow('captured text', threshold_img)

# Maintain output window until user presses a key

#cv2.waitKey(0)

# Destroying present windows on screen

#cv2.destroyAllWindows()
parse_text = []

word_list = []

last_word = ''

for word in details['text']:

    if word!='':

        word_list.append(word)

        last_word = word

    if (last_word!='' and word == '') or (word==details['text'][-1]):
        parse_text.append(word_list)

        word_list = []
#The next code will convert the result text into a file:
print(parse_text)

