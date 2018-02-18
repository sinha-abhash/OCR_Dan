from PIL import Image
import pytesseract
import argparse
import cv2
import os

'''
args = argparse.ArgumentParser()
args.add_argument("--image", required=True)
args = vars(args.parse_args())
'''
image = cv2.imread("valve.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_otsu = cv2.threshold(gray, 0, 255, 
	cv2.THRESH_OTSU)[1]
'''
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename,gray_otsu)
cv2.waitKey(0)

gray_blur = cv2.medianBlur(gray,3)
filename1 = "{}.png".format(os.getpid())
cv2.imwrite(filename1,gray_blur)
cv2.waitKey(0)'''

gray_adaptive_threshold = cv2.adaptiveThreshold(gray, 175,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,3)
filename2 = "{}.png".format(os.getpid())
cv2.imwrite(filename2, gray_adaptive_threshold)

#text = pytesseract.image_to_string(Image.open(filename))
#text1 = pytesseract.image_to_string(Image.open(filename1))
text2 = pytesseract.image_to_string(Image.open(filename2))
'''print("text from: ", filename)
print(text)
print("--------------------------------------------------------------")
print("text from: ", filename1)
print(text1)'''
print("--------------------------------------------------------------")
print("text from: ", filename2)
print(text2)
