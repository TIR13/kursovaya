import cv2 as cv
import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
import subprocess as s
import os
#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
"""
Apply morphology operations
"""
def isolate_lines(src, structuring_element):
	cv.erode(src, structuring_element, src, (-1, -1)) # makes white spots smaller
	cv.dilate(src, structuring_element, src, (-1, -1)) # makes white spots bigger

"""
Verify if the region inside a contour is a table
If it is a table, returns the bounding rect
and the table joints. Else return None.
"""
MIN_TABLE_AREA = 50 # min table area to be considered a table
EPSILON = 3 # epsilon value for contour approximation
def verify_table(contour, intersections):
    area = cv.contourArea(contour)

    if (area < MIN_TABLE_AREA):
        return (None, None)

    # approxPolyDP approximates a polygonal curve within the specified precision
    curve = cv.approxPolyDP(contour, EPSILON, True)

    # boundingRect calculates the bounding rectangle of a point set (eg. a curve)
    rect = cv.boundingRect(curve) # format of each rect: x, y, w, h
    
    #print(rect)
    # Finds the number of joints in each region of interest (ROI)
    # Format is in row-column order (as finding the ROI involves numpy arrays)
    # format: image_mat[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w]
    possible_table_region = intersections[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    #print(possible_table_region)
    (possible_table_joints, _) = cv.findContours(possible_table_region, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Determines the number of table joints in the image
    # If less than 5 table joints, then the image
    # is likely not a table
    if len(possible_table_joints) < 5:
        return (None, None)

    return rect, possible_table_joints

"""
Creates the build directory if it doesn't already exist."
"""
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

"""
Displays an image with opencv for durationMillis milliseconds
"""
def showImg(name, matrix, durationMillis = 0):
    cv.imshow(name, matrix)
    cv.waitKey(durationMillis)

"""
Clean the image by using the textcleaner script
"""
def run_textcleaner(filename, img_id):
    mkdir("bin/cleaned/")

    # Run textcleaner
    cleaned_file = "bin/cleaned/cleaned" + str(img_id) + ".jpg"
    print(filename)
    print(cleaned_file)
    s.call(["./textcleaner", "-g", "-e", "none", "-f", str(10), "-o", str(5), filename, cleaned_file], shell=True)

    return cleaned_file

"""
Run tesseract to perform optical character recognition (OCR)
"""
def run_tesseract(filename, img_id, psm, oem):
    mkdir("bin/extracted/")

    image = Image.open(filename)
    image=cv2.imread (filename)
    language = 'rus'
    configuration = "--psm " + str(psm) + " --oem " + str(oem)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    details = pytesseract.image_to_data(image,output_type=Output.DICT, config=configuration, lang='rus')
    #print(details.keys())

    total_boxes = len(details['text'])

    for sequence_number in range(total_boxes):
    
    	if int(details['conf'][sequence_number]) >30:
    
     		(x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
    
     		image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # display image
    
    #cv2.imshow('captured text', image)
    
    # Maintain output window until user presses a key
    
    #cv2.waitKey(0)
    
    # Destroying present windows on screen
    
    cv2.destroyAllWindows()
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
    #print(parse_text)

    # Run tesseract
    text = pytesseract.image_to_string(image, lang=language, config=configuration)
    #if len(text.strip()) == 0:
    #    configuration += " -c tessedit_char_whitelist=0123456789"
     #   text = pytesseract.image_to_string(image, lang=language, config=configuration)
    #print(text)

    return parse_text
