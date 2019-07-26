import imutils 
import cv2 
import numpy as np 

def preprocess(image): 
    # Resize cropped part
    result = imutils.resize(image, width = 1000)

    # Convert to gray scale
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Threshold 
    _, result = cv2.threshold(result, 140, 255, cv2.THRESH_BINARY)


    '''
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, closing_kernel)
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, opening_kernel)

    for i in range(0, 5):
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5 + 2 * i))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, closing_kernel)
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5 + 2 * i))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, opening_kernel)

    for i in range(5, 15):
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5 + 2 * i))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, opening_kernel)
    '''
    return result


