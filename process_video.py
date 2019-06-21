import datetime
import time
import cv2
from pyzbar import pyzbar
import argparse
import imutils

from image_util import extra_processing
from image_util import preprocess_image

def process_video(video_path, output_path):

    # initialize video stream
    print("[INFO] Starting video stream... ")
    video = cv2.VideoCapture(video_path)
    fourcc = cv2.cv.CV_FOURCC(*'XVID') if imutils.is_cv2() else cv2.VideoWriter_fourcc(*'XVID')
    annotated_video = cv2.VideoWriter("annotated.avi", fourcc, 20.0, (1440, 960))
    time.sleep(2.0)

    # open the output csv file for writing and initialize the set of barcodes found thus far
    csv = open(output_path, "w")
    found = set()

    framecount = 0

    # Loop over frames in video
    while True:
        for i in range(0, 1):
            #grab the frame and resize
            (grabbed, frame) = video.read()
        # check for end of video
        if (not grabbed):
            break

        cv2.imshow("original", cv2.resize(frame, (960, 640)))
        cv2.waitKey(1)
        framecount += 1

        if (framecount % 10 == 0):
            print("We are at frame {} and time roughly {}.".format(framecount, framecount / 30))

        barcodes = pyzbar.decode(frame)
        for barcode in barcodes:
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            barcodeLoc = barcode.rect
            cv2.putText(frame, "{}({})".format(barcodeData, barcodeType), (barcodeLoc.left, barcodeLoc.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (barcodeLoc.left, barcodeLoc.top), (barcodeLoc.left + barcodeLoc.width, barcodeLoc.top + barcodeLoc.height), (0, 255, 0), thickness = 2)
            cv2.imshow("detected!", cv2.resize(frame, (960, 640)))

        processed_frame = preprocess_image(frame)

        for (processed_part, plain_part, location_box) in processed_frame:

            x = min(location_box[0][0], location_box[1][0], location_box[2][0], location_box[3][0])
            y = min(location_box[0][1], location_box[1][1], location_box[2][1], location_box[3][1])
            x2 = max(location_box[0][0], location_box[1][0], location_box[2][0], location_box[3][0])
            y2 = max(location_box[0][1], location_box[1][1], location_box[2][1], location_box[3][1])
            cv2.drawContours(frame, [location_box], -1, (0, 255, 0), 2)
            #find barcodes in frame and decode each
            barcodes = pyzbar.decode(plain_part) + pyzbar.decode(processed_part)

            if (barcodes == []):
                if ((x2 - x) * (y2 - y) < 14000):
                    cv2.putText(frame, "Failed to decode", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    continue
                # If we cannot decode the barcode, we use extra processing on the processed part
                extra_processed = extra_processing(processed_part, plain_part, vertical_padding_removal=int((y2 - y) / 6))
                if (framecount % 10 ==0):
                    cv2.imshow("extra processed" , extra_processed)
                barcodes = pyzbar.decode(extra_processed)
                if (barcodes == []):
                    cv2.putText(frame, "Failed to decode", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    continue
                #elif (len(barcodes) > 1):
                #    cv2.putText(frame, "This barcode was decoded as multiple barcodes. Probability of failure is high",
                #                (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    for barcode in barcodes:
                        #                (x, y, w, h) = barcode.rect
                        #                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        barcodeData = barcode.data.decode("utf-8")
                        barcodeType = barcode.type

                        text = "{}({})".format(barcodeData, barcodeType)
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow("detected!", cv2.resize(frame, (960, 640)))
                        cv2.waitKey(100)

                        if barcodeData not in found:
                            csv.write("{}, {}\n".format(datetime.datetime.now(), barcodeData))
                            csv.flush()
                            found.add(barcodeData)

                # cv2.putText(frame, "Failed to decode", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # continue
            # elif (len(barcodes) > 1):
                # cv2.putText(frame, "This barcode was decoded as multiple barcodes. Probability of failure is high", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #loop oer the detected barcodes
            for barcode in barcodes:
#                (x, y, w, h) = barcode.rect
#                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                barcodeData = barcode.data.decode("utf-8")
                barcodeType = barcode.type

                text = "{}({})".format(barcodeData, barcodeType)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("detected!", cv2.resize(frame, (960, 640)))
                cv2.waitKey(100)

                if barcodeData not in found:
                    csv.write("{}, {}\n".format(datetime.datetime.now(), barcodeData))
                    csv.flush()
                    found.add(barcodeData)

        if (framecount % 10 == 0):

            for i in range (0, len(processed_frame)):
                cv2.imshow("processed part " + str(i), processed_frame[i][0])
                cv2.imshow("plain part " + str(i), processed_frame[i][1])
            cv2.waitKey(10)

#        to_display = cv2.resize(frame, (960, 640))
#        cv2.imshow("frame", to_display)
#        cv2.waitKey(0)

        to_write = cv2.resize(frame, (1440, 960))
        annotated_video.write(to_write)

    print("[INFO] cleaning up...")
    csv.close()
    video.release()
    annotated_video.release()
    cv2.destroyAllWindows()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="barcodes.csv",
	help="path to output CSV file containing barcodes")
ap.add_argument("-v", "--video", required = True, help = "path to input video")
args = vars(ap.parse_args())

process_video(args["video"], args["output"])
