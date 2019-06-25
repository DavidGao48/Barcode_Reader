from pyzbar import pyzbar
import argparse
from image_util import *

'''
\:brief Uses ZBar to find and decode all barcodes in an image 
\:param image An image to be decoded 
\:returns void. But edits the image to show bounding boxes and text around decoded barcodes. 
'''
def decode_image(image):

    barcodes = pyzbar.decode(image)

    for barcode in barcodes:
        #                (x, y, w, h) = barcode.rect
        #                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

        x, y, w, h = barcode.rect.left, barcode.rect.top, barcode.rect.width, barcode.rect.height

        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness = 5)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("results", cv2.resize(image, (960, 640)))
    cv2.waitKey(0)



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input input")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])

decode_image(img)
