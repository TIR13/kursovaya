import numpy as np
import cv2 as cv
import cv2
import utils
from table import Table
from PIL import Image
import xlsxwriter
import sys
import re
import csv
from pdf2image import convert_from_path

# =====================================================
# IMAGE LOADING
# =====================================================
FILENAME="8-A_Algebra.csv"

def add_class(allw):
        #name=[[] * 2 for i in range(30)]
        first=[]
        Flags_name=0
        cnt=0
        string_s=''
        for s in allw:
            Flags_s=0
            #print(s)
            #string_s=''
            for rows in s:
                #print(rows)
                if(Flags_s==1):
                    break
                for l in range(len(rows)):
                    
                    #print(rows[l])
                    string_s=string_s+str(rows[l])
                    
                    if(Flags_name==0 and l<len(rows)-1):
                        FIO=str(rows[l])+str(rows[l+1])
                        #print(FIO)
                        if(re.fullmatch('^[^А-Яа-я]*[А-Я]{1}[а-я]{3,}(-[А-Я]{1}[а-я]{3,})?[А-Я]{1}[а-я]{3,}[^А-Яа-я]*', FIO)):
                           first_name=re.search('[А-Я]{1}[а-я]{3,}(-[А-Я]{1}[а-я]{3,})?',re.search('^[^А-Яа-я]*[А-Я]{1}[а-я]{3,}(-[А-Я]{1}[а-я]{3,})?', FIO).group(0)).group(0)
                           last_name=re.search('[А-Я]{1}[а-я]{3,}',re.search('[А-Я]{1}[а-я]{3,}[^А-Яа-я]*$', FIO).group(0)).group(0)
                           #first[cnt].append(FIO)
                           print(first_name+" "+last_name)
                           first.append(first_name)
                           first.append(last_name)
                           #name[cnt]=rows[l],rows[l+1]
                           cnt=cnt+1
                           Flags_name=1
                           Flags_s=1
                           break
                    if(Flags_name==1):
                        #print(rows[l])
                        #print(re.fullmatch('[З]{1}',rows[l]))
                        if(re.fullmatch('.?[2-5]{1}.?',rows[l])):
                            #print(cnt)
                            #first[cnt-1].append(int(rows[l]))
                            first.append(re.search('[2-5]{1}', rows[l]).group(0))
                            Flags_s=1
                            break
                        elif(re.fullmatch('[З]{1}',rows[l])):
                            #first[cnt-1].append(3)
                            first.append(3)
                            Flags_s=1
                            break
                        elif(re.fullmatch('.?(Н|н){1}.?',rows[l])):
                            #print(cnt)
                            #first[cnt-1].append(rows[l])
                            first.append('Н')
                            Flags_s=1
                            break
            if(Flags_s==0 and Flags_name==1):
                #print(s)
                #first[cnt-1].append(0)
                first.append(0)
        return first

def add_first(allw):
    f=open(FILENAME,'r')
    first=[line.replace("\n","").split() for line in f]
    f.close()

path = "2.jpg"
print(path)
if not path.endswith(".pdf") and not path.endswith(".jpg"):
    print("Must use a pdf or a jpg image to run the program.")
    sys.exit(1)

if path.endswith(".pdf"):
    ext_img = convert_from_path(path)[0]
else:
    #ext_img = Image.open(path)
    gray_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

    ext_img = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #cv2.imshow('threshold image', threshold_img)
cv2.imwrite("data/target.png",ext_img)
#ext_img.save("data/target.png", "PNG")
image = cv.imread("data/target.png")
img = cv.imread("data/target.png")
# Convert resized RGB image to grayscale
NUM_CHANNELS = 3
if len(image.shape) == NUM_CHANNELS:
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# =====================================================
# IMAGE FILTERING (using adaptive thresholding)
# =====================================================
"""
ADAPTIVE THRESHOLDING
Thresholding changes pixels' color values to a specified pixel value if the current pixel value
is less than a threshold value, which could be:

1. a specified global threshold value provided as an argument to the threshold function (simple thresholding),
2. the mean value of the pixels in the neighboring area (adaptive thresholding - mean method),
3. the weighted sum of neigborhood values where the weights are Gaussian windows (adaptive thresholding - Gaussian method).

The last two parameters to the adaptiveThreshold function are the size of the neighboring area and
the constant C which is subtracted from the mean or weighted mean calculated.
"""
MAX_THRESHOLD_VALUE = 255
BLOCK_SIZE = 15
THRESHOLD_CONSTANT = 0

# Filter image
filtered = cv.adaptiveThreshold(~grayscale, MAX_THRESHOLD_VALUE, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, BLOCK_SIZE, THRESHOLD_CONSTANT)
cv2.imwrite("filt1.jpg",filtered)
# =====================================================
# LINE ISOLATION
# =====================================================
"""
HORIZONTAL AND VERTICAL LINE ISOLATION
To isolate the vertical and horizontal lines, 

1. Set a scale.
2. Create a structuring element.
3. Isolate the lines by eroding and then dilating the image.
"""
SCALE = 35

# Isolate horizontal and vertical lines using morphological operations
horizontal = filtered.copy()
vertical = filtered.copy()

horizontal_size = int(horizontal.shape[1] / SCALE)
#print(horizontal_size)
horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
utils.isolate_lines(horizontal, horizontal_structure)

vertical_size = int(vertical.shape[0] / SCALE)
#print(vertical_size)
vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
utils.isolate_lines(vertical, vertical_structure)

# =====================================================
# TABLE EXTRACTION
# =====================================================
# Create an image mask with just the horizontal
# and vertical lines in the image. Then find
# all contours in the mask.
mask = horizontal + vertical
cv2.imwrite("filt2.jpg",mask)
(contours, _) = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,0,(0,255,0), 3)
cv2.imwrite("contour3.jpg",img)
# Find intersections between the lines
# to determine if the intersections are table joints.
intersections = cv.bitwise_and(horizontal, vertical)
#print(intersections)
# Get tables from the images
tables = [] # list of tables

for i in range(len(contours)):
    curve = cv2.approxPolyDP(contours[i], 3, True)
    #print(curve)
    cv2.drawContours(img,curve,-1,(0,255,0), 20)
    
    cv2.imwrite("contour3.jpg",img)
    # Verify that region of interest is a table
    (rect, table_joints) = utils.verify_table(contours[i], intersections)
    if rect == None or table_joints == None:
        continue

    # Create a new instance of a table
    table = Table(rect[0], rect[1], rect[2], rect[3])

    # Get an n-dimensional array of the coordinates of the table joints
    joint_coords = []
    #print(len(table_joints))
    for i in range(len(table_joints)):
        joint_coords.append(table_joints[i][0][0])
    #print(joint_coords)
    joint_coords = np.asarray(joint_coords)

    # Returns indices of coordinates in sorted order
    # Sorts based on parameters (aka keys) starting from the last parameter, then second-to-last, etc
    sorted_indices = np.lexsort((joint_coords[:, 0], joint_coords[:, 1]))
    joint_coords = joint_coords[sorted_indices]
    #print(joint_coords)
    # Store joint coordinates in the table instance
    table.set_joints(joint_coords)
    tables.append(table)
    #cv2.drawContours(image,table_joints,0,(0,255,0), 3)
    #cv2.rectangle(image, (table.x, table.y), (table.x + table.w, table.y + table.h), (0, 255, 0), 10, 8, 0)
    #cv2.imwrite("contour.jpg",image)
    
    #cv.waitKey(0)
    #print(len(contours))

# =====================================================
# OCR AND WRITING TEXT TO EXCEL
# =====================================================
out = "bin/"
table_name = "table.jpg"
psm = 6
oem = 3
mult = 3    

utils.mkdir(out)
utils.mkdir("bin/table/")

utils.mkdir("excel/")
workbook = xlsxwriter.Workbook('excel/tables.xlsx')

for table in tables:
    worksheet = workbook.add_worksheet()

    table_entries = table.get_table_entries()
    #print(table_entries())
    table_roi = image[table.y:table.y + table.h, table.x:table.x + table.w]
    table_roi = cv.resize(table_roi, (table.w * mult, table.h * mult))
    #cv2.rectangle(image, (table.x, table.y), (table.x + table.w, table.y + table.h), (0, 255, 0), 1, 8, 0)
    #cv2.imwrite("contour.jpg",image)
    cv.imwrite(out + table_name, table_roi)
    allw=[]
    num_img = 0
    #name=[[] * 2 for i in range(30)]
    first=[]#[[] * 100 for i in range(30)]
    #cnt=0
    print(len(table_entries))
    for i in range(len(table_entries)):
    #for i in range(5):
        row = table_entries[i]
        print(row)
        for j in range(len(row)):
            entry = row[j]
            entry_roi = table_roi[entry[1] * mult: (entry[1] + entry[3]) * mult, entry[0] * mult:(entry[0] + entry[2]) * mult]
            #print(entry_roi.shape)
            fname = out + "table/cell" + str(num_img) + ".jpg"
            cv.imwrite(fname, entry_roi)
            #img=cv.imread("bin/table.jpg")
            cv2.rectangle(table_roi, (entry[0]*mult, entry[1]*mult), (entry[0]*mult + entry[2]*mult, entry[1]*mult + entry[3]*mult), (0, 0, 255), 8, 8, 0)
            cv2.imwrite("contour2.jpg",table_roi)
            #fname = utils.run_textcleaner(fname, num_img)
            text = utils.run_tesseract(fname, num_img, psm, oem)
            #text=[]
            allw.append(text)
            num_img += 1
            #print(text)
            #worksheet.write(i, j, text)
        users=add_class(allw)
        #users=[1]
        #print(users)
        if len(users)!=0:
            first.append(users)
        #print(allw)
        allw=[]
    #print(first)
    for i in range(len(first)):
        print(first[i])
        #print(name[i])
    
    
    """FILENAME="8-A_Algebra.csv"
    with open(FILENAME, 'w', newline="") as file:
        csv.writer(file, delimiter=" ").writerows(first)"""
workbook.close()
