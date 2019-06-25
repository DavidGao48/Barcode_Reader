import imutils
import cv2
import numpy as np


import matplotlib.pyplot as plt

#########################
#Helpers
#########################

'''
\:brief Helper function; finds if a 1d array contains a sufficiently long subarray of a certain value 
\:note O(length of array) 
\:param a An ndarray, required to be 1d
\:param length The length of the subarray we are looking for 
\:param consec_val The value we want each entry in the subarray to be 
\:returns boolean True iff a contains sufficiently long subarray of consecutive consec_val's. 
'''
def contains_consecutive_with_length(a: np.ndarray, length, consec_val):
    array_length = a.shape[0]
    if (array_length < length):
        return False

    max_length = 0
    temp_count = 0
    for i in range(0, array_length):
        if (a[i] == consec_val):
            temp_count += 1
        else:
            temp_count = 0

        if temp_count > max_length:
            max_length = temp_count

        if max_length >= length:
            return True

    return False

'''
\:brief Helper function; finds if a 1d array contains a certain value at a sufficient density 
\:note O(length of array)
\:param a An array, required to be 1d. 
\:param density The density wanted. 
\:param val The value to be counted. 
\:returns boolean True iff a contains suffient density of val. 
'''
def sufficient_density(a: np.ndarray, density, val):
    count = (a == val).sum()
    length = a.shape[0]
    return count >= length * density

'''
\brief Helper function; gets the minimum degree of rotation needed to straighten a rotated rect
\param[in] degree The angle given by rect[-1]
\returns double The correct degree 
'''
def correct_degree(degree):
    sign = 1 if degree >= 0 else -1
    threshold = 80.0

    if np.abs(degree) > threshold:
        if sign == 1:
            return 90 - degree
        else:
            return -90 - degree

    return degree

'''
\brief Sanity checks if a contour looks anything like the outline of a barcode 
\param[in] contour The contour that needs to be judged. 
\returns boolean False iff the contour is unlikely to be the outline of a barcode 
'''
def contour_resembles_barcode(contour):
    # Any contour sufficiently large deserve to be tried
    # TODO this is only a temporary measure to boost performance in high quality videos
    if cv2.contourArea(contour) >= 14000: return True

    # Find the minimum bounding rectangle (may be rotated) of the contour
    rect = cv2.minAreaRect(contour)
    height = rect[1][0]
    width = rect[1][1]
    angle = correct_degree(rect[-1])

    # Find the difference in area between the bounding rect and the contour
    contour_area = cv2.contourArea(contour)
    rect_area = height * width
    area_diff = rect_area - contour_area

    return (area_diff < 0.4 * rect_area) \
           and (cv2.contourArea(contour) > 1000) \
           and (-15 < angle and angle < 15)  \
           and ((4 < width / height and width/height < 10) or (0.5 < width / height and width / height < 2))

'''
\brief Increases the color image contrast.
\note Inspired by this answer: https://stackoverflow.com/questions/19363293/whats-the-fastest-way-to-increase-color-image-contrast-with-opencv-in-python-c/19384041
\raises AssertionError if either phi or theta are set to zero
\param[in] img Numpy image to be processed
\param[in] phi Parameters for manipulating the image intensity
\param[in] theta Parameters for manipulating the image intensity
\returns Enhanced image
'''
def contrastBrightnessCurve(img, phi=1, theta=20):
    if(phi==0 or theta==0):
        raise AssertionError('The phi and theta parameters cannot be zero')
    maxIntensity = 255.0 # depends on dtype of image data
    # Parameters for manipulating image data
    # Increase intensity such that bright pixels become slightly bright
    newImage0 = (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5
    img = np.array(newImage0,dtype=np.uint8)
    return img

'''
\brief Equalizes the histogram of the lighting channel in LAB color space
\note The image is expected to be in BGR color space
\raises AssertionError in case the image does not have 3 channels.
\param[in] img Numpy image to be processed.
\param[in] clip_limit From opencv: "Threshold for contrast limiting"
\param[in] tile_grid_size From opencv: "Size of grid for histogram equalization"
\returns The equalized image
'''
def adaptiveHistogram(img, clip_limit=2.0, tile_grid_size=(8,8)):
    if(len(img.shape)<3 or img.shape[2] != 3):
        raise AssertionError('The image need to be a colored one')
    ### Adaptive Histogram Equalisation of the Contrast
    ### CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2,a,b))  # merge channels
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img

'''
\brief Adjust the color saturation
\note The image is expected to be in BGR color space
\raises AssertionError in case the image does not have 3 channels
\param[in] img Numpy image to be processed
\param[in] intensity Saturation intensity
\returns Enhanced image
'''
def saturation(img, intensity=1.75):
    if(len(img.shape)<3 or img.shape[2] != 3):
        raise AssertionError('The image need to be a colored one')
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    s = s*intensity
    s = np.clip(s,0,255)
    imghsv = cv2.merge([h,s,v])
    img = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return img

'''
\brief Filter that enhance corners without blurring the image too much
\param[in] img Numpy image to be processed
\param[in] pixel_diameter from opencv: Diameter of each pixel neighborhood that is used during filtering
\param[in] intensity Intensity of the sharpening effect. Used for both sigmaColor and sigmaSpace in opencv
'''
def bilateralFilter(img, pixel_diameter=4, intensity=112):
    return cv2.bilateralFilter(img,pixel_diameter,intensity,intensity)

'''
\brief Sharpens the image as a whole
\note inspired by this article: (https://en.wikipedia.org/wiki/Unsharp_masking#Digital_unsharp_masking)
\param[in] img Numpy image to be processed
\param[in] a Weights for sharpening
\param[in] b Same
\param[in] c Same
'''
def sharpening(img, a=0.3, b=1.5, c=-0.5):
    blur = cv2.GaussianBlur(img, (0, 0), a)
    img  = cv2.addWeighted(blur, b, img, c, 0)
    return img

#TODO This function needs to be separated into two functions: crop_around_rect and preprocess_cropped_part
'''
\:brief Crops and preprocesses a part of an image for barcode decoding
\:param image The original image (likely a frame in a video) 
\:rect rect The (rotated) bbox around which we wish to crop 
\:returns UMat a cv2 Mat representing the cropped and preprocessed part of image inside rect 
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

    just_cropped = result

    # Resize cropped part
    result = imutils.resize(result, width = 1000)

    # Convert to gray scale
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Threshold again
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
 #   result = cv2.blur(result, (3,3))

 #   cv2.imshow("cropped", result)
 #   cv2.waitKey(100)

    return result, just_cropped

#TODO This function is currently not in use. Need to assess whether it should be removed.
'''
\:brief More expensive option for processing a cropped part extra finely for barcode decoding 
'''
def extra_processing(processed_image, plain_image, vertical_padding_removal):
    result = np.asarray(plain_image)
    h, w, c = result.shape
    # separately threshold in channels
    sep_thresh = np.zeros((h, w, c))
    '''
    cv2.rectangle(result, (0, 0), (w/10, h), (255,255,255), 2)
    cv2.rectangle(result, (w - w/10, 0), (w, h), (255,255,255), 2)
    cv2.rectangle(result, (0, 0), (w, vertical_padding_removal), (255,255,255), 2)
    cv2.rectangle(result, (0, h - vertical_padding_removal), (w, h), (255,255,255), 2)

    cv2.imshow("vertical padding removed", result)
    '''


    for i in range(0, c):
        _, sep_thresh[0:h, 0:w, i] = cv2.threshold(result[0:h, 0:w, i], 150, 255, cv2.THRESH_BINARY)
    # take the brightest of thresholded channels as a gray scale
    result = np.max(sep_thresh, 2)

    '''
    vert_removed = result[vertical_padding_removal: h - vertical_padding_removal, :]
    vert_collapsed = np.sum(vert_removed, 0) / (h - 2 * vertical_padding_removal) / 255
    for x in range (0, w):
        result[vertical_padding_removal:h - vertical_padding_removal, x] = vert_collapsed[x] * np.ones((h - 2 * vertical_padding_removal))

    _, result = cv2.threshold(result, 0.5, 1, cv2.THRESH_BINARY)
    '''


    # any vertical line that contains a sufficiently long line of black should be all black
    for x in range(0, w):
        this_column = result[vertical_padding_removal:h-vertical_padding_removal, x]
        if (contains_consecutive_with_length(this_column, 30, 0)):
            result[vertical_padding_removal:h-vertical_padding_removal, x] = np.zeros((h-2 * vertical_padding_removal))

    '''
    # any vertical line that contains sufficient density of black should be all black
    for x in range(0, w):
        this_column = result[vertical_padding_removal:h-vertical_padding_removal, x]
        if (sufficient_density(this_column, 0.2, 0)):
            result[vertical_padding_removal:h-vertical_padding_removal, x] = np.zeros((h-2*vertical_padding_removal))
    '''

    # any vertical line that still contains a sufficiently long line of white should be all white
    for x in range(0, w):
        this_column = result[vertical_padding_removal:h-vertical_padding_removal, x]
        if (contains_consecutive_with_length(this_column, 30, 255)):
            result[vertical_padding_removal:h-vertical_padding_removal, x] = 255 * np.ones((h-2 * vertical_padding_removal))

    '''
    bar_widths = []
    curr_count = 0
    curr_color = result[int(h/2), 0]
    min_width = 1000

    for x in range(0, w):
        if (result[int(h/2), x] == curr_color):
            curr_count += 1
        else:
            bar_widths += [(curr_count, curr_color)]
            if curr_count < min_width:
                min_width = curr_count
            curr_count = 1
            curr_color = result[int(h/2), x]

    for i in range(0, len(bar_widths)):
        if bar_widths[i][0] > 12 * min_width:
            bar_widths[i] = (-1, -1)

    bar_widths = list(filter(lambda a: a != (-1, -1), bar_widths))

    if (bar_widths[0][1] != 0):
        bar_widths.pop(0)

    bar_widths = np.asarray(bar_widths, int)

    print(bar_widths)
    print("------------------")
    '''
    return result
#########################
# body
#########################

'''
\:brief Preprocesses an image for barcode decoding 
\:param image The original image, likely a frame of a video
\:returns a list of tuples of the form (result_part, plain_part, box), where 
            each result_part is a processed cropped part (containing a barcode) of the image 
            plain_part is the corresponding non-processed, cropped part (containing a barcode) of the image 
            each box is the corresponding bbox of that cropped part 
'''
def preprocess_image(image):
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
        result_part, plain_part = crop_around_rect(image, rect)
        result_parts += [(result_part, plain_part, box)]

    return result_parts
