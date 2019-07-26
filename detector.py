from image_util import *
import cv2 
import numpy as np 


'''
crop_around_rect 
\:brief crops an image around a rectangle 
\:param image The original image 
\:param rect the (rotated) bbox around which we wish to crop 
\:returns UMat a cv2 mat representing the cropped image 
'''
def crop_around_rect(image, rect): 
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect) 
    h, w, _ = image.shape 

    # Locate the desired  cropping
    x1 = min(box[0][0], box[1][0], box[2][0], box[3][0])
    y1 = min(box[0][1], box[1][1], box[2][1], box[3][1])
    x2 = max(box[0][0], box[1][0], box[2][0], box[3][0])
    y2 = max(box[0][1], box[1][1], box[2][1], box[3][1])

    # Add padding proportional to the sqrt of the area of the box
    x_padding = 0.1 * (np.sqrt((x2 - x1) * (y2 - y1)))
    y_padding = 0.1 * (np.sqrt((x2 - x1) * (y2 - y1)))
    if (x1 >= x_padding): x1 -= x_padding
    else: x1 = 0
    if (y1 >= y_padding): y1 -= y_padding
    else: y1 = 0
    if (w - x2 >= x_padding): x2 += x_padding
    else: x2 = w
    if (h - y2 >= y_padding): y2 += y_padding
    else: y2 = h

    # Crop the image
    result = image[int(y1):int(y2), int(x1):int(x2)]

    # Rotate the image so that barcode is upright
    M = cv2.getRotationMatrix2D((w / 2 + x_padding, h / 2 + y_padding), correct_degree(rect[-1]), 1)
    result = cv2.warpAffine(result, M, (int(x2) - int(x1), int(y2) - int(y1)))

    return result 


'''
\:brief Detects location of barcodes in image and splits image for decoding 
\:param image The original image, likely a frame of a video
\:returns a list of tuples of the form (cropped_part, box) 
'''
def detect_and_split(image):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Scharr gradient magnitude representation
    # In both the X and Y directions
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray_frame, ddepth = ddepth, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray_frame, ddepth = ddepth, dx = 0, dy = 1, ksize = -1)

    # subtract the y-gradient from the x-gradient
    gradient_diff = cv2.subtract(gradX, gradY)
    gradient_diff = cv2.convertScaleAbs(gradient_diff)

    # blur and threshold the image
    blurred = cv2.blur(gradient_diff, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 225, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closing_kernel)

    # perform several iterations of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

#    cv2.imshow("closed", cv2.resize(closed, (960, 640)))

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if (len(cnts) == 0): return []

    filtered_cnts = filter(lambda c: contour_resembles_barcode(c), cnts)
    result_parts = []

    for c in filtered_cnts:
        # compute the rotated bounding box of the largest 2 contours
        rect = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
        box = np.int0(box)

        # draw a bounding box around the detected barcode
        # cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

        # compute the matrix of the rotation needed
        # rows, cols, _ = frame.shape
        # M = cv2.getRotationMatrix2D((cols/2, rows/2), correct_degree(rect[-1]), 1)
        cropped_part = crop_around_rect(image, rect)
        result_parts += [(cropped_part, box)]

    return result_parts


