import datetime
import time
import cv2
from pyzbar import pyzbar
import argparse
import imutils

from image_util import extra_processing
from image_util import preprocess_image

'''
Video_Processor 
A Video Processor is a stateful object that decodes barcodes in a video by stepping through a video, 
and decoding frames as individual images 
\:param video_path the path to the video to be processed 
\:param out_video_path the path to store annotated video, leave as None if no video output needed 
\:param fourcc the fourcc code of the output format, leave as default if no video output needed, defaults to XVID 
'''
class Video_Processor():
    def __init__(self, video_path, out_video_path = None, fourcc = 'XVID'):
        ## Open video file and create video capture
        print("[Video Processor] Starting video stream... ")
        self.video_path = video_path
        self.video = cv2.VideoCapture(self.video_path)

        ## Check if we want to write an annotated video and create resources
        self.write_annotated_vid = (not out_video_path is None)
        if self.write_annotated_vid:
            print("[Video Processor] Opening annotated video file location... ")
            self.fourcc = cv2.cv.CV_FOURCC(*fourcc) if imutils.is_cv2() else cv2.VideoWriter_fourcc(*fourcc)
            self.annotated_video = cv2.VideoWriter(out_video_path, self.fourcc, 20.0, (1440, 960))

        ## Wait for resources to finish opening
        time.sleep(2.0)

        ## Prepare the states
        self.curr_frame_barcodes = []
        self.detections_til_now = []
        self.frame = None
        self.annotated_frame = None
        self.framecount = 0
        self.video_time = 0

    '''
    step 
    \:brief takes a few frames forwards in video 
    \:param frames the number of frames to move forward in, default to 1 
    \:returns a boolean representing whether there were still enough frames left in video 
    '''
    def step(self, frames = 1):
        video_cont = True
        ## Take frames steps forward
        for i in range(frames):
            if self.write_annotated_vid and not self.annotated_frame is None:
                to_write = cv2.resize(self.annotated_frame, (1440, 960))
                self.annotated_video.write(to_write)

            (grabbed, self.frame) = self.video.read()

            if not grabbed:
                video_cont = False
                break

            self.annotated_frame = self.frame.copy()
        ## If not enough frames left, return False
        if not video_cont:
            return False

        ## Update relevant states
        self.framecount = self.video.get(cv2.CAP_PROP_POS_FRAMES)
        self.video_time = self.video.get(cv2.CAP_PROP_POS_MSEC)
        self.curr_frame_barcodes = []

        print("[Video Processor] We are at frame number {} and time {} seconds".format(self.framecount, self.video_time / 1000))

        return True

    '''
    decode_frame 
    \:brief decodes current frame and stores decoded barcodes
    '''
    def decode_frame(self, input_size = (1280, 720)):
        ## Resize frame to input size
        self.frame = cv2.resize(self.frame, input_size)
        self.annotated_frame = cv2.resize(self.annotated_frame, input_size)
        ## First, grab the barcodes that can be read directly from the entire image
        self.curr_frame_barcodes += pyzbar.decode(self.frame)
        for barcode in self.curr_frame_barcodes:
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            barcodeLoc = barcode.rect
            cv2.putText(self.annotated_frame, "{}({})".format(barcodeData, barcodeType), (barcodeLoc.left, barcodeLoc.top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(self.annotated_frame, (barcodeLoc.left, barcodeLoc.top),
                          (barcodeLoc.left + barcodeLoc.width, barcodeLoc.top + barcodeLoc.height), (0, 255, 0),
                          thickness=2)

        ## Grab the barcodes from preprocessed parts of the frame
        processed_frame = preprocess_image(self.frame)

        for processed_part, plain_part, location_box in processed_frame:
            x1 = min(location_box[0][0], location_box[1][0], location_box[2][0], location_box[3][0])
            y1 = min(location_box[0][1], location_box[1][1], location_box[2][1], location_box[3][1])
            cv2.drawContours(self.annotated_frame, [location_box], -1, (0, 255, 0), 2)

            part_barcodes = pyzbar.decode(plain_part) + pyzbar.decode(processed_part)
            self.curr_frame_barcodes += part_barcodes

            if part_barcodes == []:
                cv2.putText(self.annotated_frame, "Failed to decode", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for barcode in part_barcodes:
                    barcodeData = barcode.data.decode("utf-8")
                    barcodeType = barcode.type
                    cv2.putText(self.annotated_frame, "{}({})".format(barcodeData, barcodeType),
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for barcode in self.curr_frame_barcodes:
            self.detections_til_now += [{'framecount': self.framecount,
                                         'videotime': self.video_time,
                                         'barcode': barcode}]

    '''
    show_monitors 
    \:brief Displays the current frame and its annotated version
    \:param waitKey the wait behavior, defaults to 1 msecs 
    '''
    def show_monitors(self, waitKey = 1):
        cv2.imshow("Original_Frame", cv2.resize(self.frame, (960, 640)))
        cv2.imshow("Annotated_Frame", cv2.resize(self.annotated_frame, (960, 640)))
        cv2.waitKey(waitKey)

    '''
    dump_data 
    \:brief Dumps detections til now into a csv file, where column 0 is the frame of detection, column 1
            is the video time of detection, and column 2 is the content of barcode 
    \:param csv_path the path to store the csv file 
    '''
    def dump_data(self, csv_path):
        print("[Video Processor] Writing to file {}".format(csv_path))
        csv = open(csv_path, "w")
        for detection in self.detections_til_now:
            csv.write("{}, {}, {}\n".format(detection['framecount'], detection['videotime'], detection['barcode'].data.decode("utf-8")))
        csv.flush()
        csv.close()
        print("[Video Processor] Finished writing to file {}".format(csv_path))

    '''
    close 
    \:brief closes all resources 
    '''
    def close(self):
        print("[Video Processor] Cleaning up ... ")
        self.video.release()
        if self.write_annotated_vid:
            self.annotated_video.release()
        cv2.destroyAllWindows()
        print("[Video Processor] Closed. ")

    '''
    run 
    \:brief runs the Video_Processor through the assigned video 
    \:param step_size the number of frames to skip at each step; defaults to 1
    \:param show_monitors should monitors be displayed? defaults to True 
    '''
    def run(self, step_size = 1, show_monitors = True, csv_path = None):
        ## Step into first frame
        self.step()

        while True:
            ## Decode current frame
            self.decode_frame()
            ## Display monitors
            if show_monitors:
                self.show_monitors()
            ## Step to next frame, and break if no such frame
            if not self.step(frames = step_size):
                break

        ## Dump results
        if not csv_path is None:
            self.dump_data(csv_path)
        ## Close resources
        self.close()

    '''
    run_eval 
    \:brief evaluates the precision and recall of this video processor, given ground truth
    \:param ground_truth_path the path to the ground truth file, which must be a csv formatted as follows: 
            - There is one row for each barcode that makes an appearance in the video 
            - The first column contains string identifiers for the barcodes in each row 
            - The second column contains the contents of the barcodes 
            - The third column contains 
    '''
    def run_eval(self, ground_truth_path, show_monitors = False, log_path = None):
        return

video_processor = Video_Processor("../Images/3rd-stage_no-light.mp4", out_video_path="annotated.avi")
video_processor.run(csv_path = "detections.csv")