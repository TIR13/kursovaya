import cv2 as cv
import cv2
FILENAME="8-A_Algebra.csv"

f=open(FILENAME,'r')
first=[line.replace("\n","").split() for line in f]
f.close()
print(first[0][0])
