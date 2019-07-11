import datetime
import time
import cv2
from pyzbar import pyzbar
import argparse
import imutils
import json

from image_util import extra_processing
from image_util import preprocess_image

'''
boxes_intersect (helper) 
\:brief checks if two boxes intersect 
\:param box1, box2: two boxes, in the form (left, top, right, bottom) 
'''
def boxes_intersect(box1, box2):
    intersect_in_x = (box2[0] < box1[0] < box2[2]) or (box1[0] < box2[0] < box1[2])
    intersect_in_y = (box2[1] < box1[1] < box2[3]) or (box1[1] < box2[1] < box1[3])
    return intersect_in_x and intersect_in_y

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
    jump 
    \:brief jumps to a specified frame number in the video 
    \:param frame_num the frame number to jump to 
    \:returns a boolean representing whether there is such a frame
    '''
    def jump(self, frame_num):
        if self.write_annotated_vid and not self.annotated_frame is None:
            to_write = cv2.resize(self.annotated_frame, (1440, 960))
            self.annotated_video.write(to_write)

        if frame_num <= self.video.get(cv2.CAP_PROP_FRAME_COUNT):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            (grabbed, self.frame) = self.video.read()

            if not grabbed:
                return False

            self.annotated_frame = self.frame.copy()

            self.framecount = self.video.get(cv2.CAP_PROP_POS_FRAMES)
            self.video_time = self.video.get(cv2.CAP_PROP_POS_MSEC)
            self.curr_frame_barcodes = []

            print("[Video Processor] We are at frame number {} and time {} seconds".format(self.framecount, self.video_time / 1000))

            return True

        return False

    '''
    decode_frame 
    \:brief decodes current frame and stores decoded barcodes
    \:param input_size the resolution at which the decoder will read the frame 
    \:returns a list of all decoded barcodes in this frame 
    '''
    def decode_frame(self, input_size = None):
        ## Resize frame to input size
        if not input_size is None:
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
            x2 = max(location_box[0][0], location_box[1][0], location_box[2][0], location_box[3][0])
            y2 = max(location_box[0][1], location_box[1][1], location_box[2][1], location_box[3][1])
            cv2.drawContours(self.annotated_frame, [location_box], -1, (0, 255, 0), 2)

            part_barcodes = list(pyzbar.decode(plain_part) + pyzbar.decode(processed_part))
            ## Remove detections that are repetitions of what was detected in the first step from the entire image
            ## Two barcodes are considered the same if their contents are identical and their location boxes overlap
            for index, part_barcode in enumerate(part_barcodes):
                part_barcode_loc = (x1, y1, x2, y2)
                for already_detected in self.curr_frame_barcodes:
                    already_detected_loc = (already_detected.rect.left,
                                  already_detected.rect.top,
                                  already_detected.rect.left + already_detected.rect.width,
                                  already_detected.rect.top + already_detected.rect.height)
                    if part_barcode.data == already_detected.data and boxes_intersect(already_detected_loc, part_barcode_loc):
                        del part_barcodes[index]
                        break

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

        return self.curr_frame_barcodes

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
    def run(self, step_size = 1, show_monitors = True, csv_path = None, resolution_cap = None):
        ## Step into first frame
        self.step()

        while True:
            ## Decode current frame
            if not resolution_cap is None:
                self.decode_frame(input_size = resolution_cap)
            else:
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
    \:param ground_truth_path the path to the ground truth file, which must be a json formatted as follows: 
              top structure:       {'note' : str, 
                                    'frames' : [frame_data, frame_data ...] } 
              frame_data:          {'frame_num' : int, 
                                    'barcodes' : [barcode_data, barcode_data, ...]}
              barcode_data:        {'content' : str, 
                                    'loc' : [int, int, int, int], 
                                    'type' : str, 
                                    'id' : str } 
              Note: There must be exactly one frame_data for every frame in the video, except for those frames that 
                    contain no barcodes. If a barcode appears in two different frames in the video (that is, if 
                    two barcode_data describe the same physical barcode), then the same id should be assigned. 
    \:param show_monitors true if user wants to see monitors while eval runs. default to False 
    \:param log_path the path to the file where logs should be written. default to None 
    '''
    def run_eval(self, ground_truth_path, show_monitors = False, log_path = None):
        print("[Video Processor] Evaluator is opening ground truth file... ")
        gt_file = open(ground_truth_path)
        gt = json.load(gt_file)
        print("[Video Processor] Ground truth loaded. ")

        logging = False
        if not log_path is None:
            log_file = open(log_path, "w")
            logging = True

        ## Initialize counters
        true_posv = 0
        false_posv = 0
        false_negv = 0
        set_of_distinct_barcodes = set()
        set_of_detected_barcodes = set()

        ## Loop through frame data in gt
        for frame_data in gt['frames']:
            frame_num = frame_data['frame_num']
            if logging:
                log_file.write("{} ".format(frame_num))
            true_barcodes = frame_data['barcodes']

            ## Try to jump to the frame, and skip the data point if such frame does not exist
            if not self.jump(frame_num):
                print("[Video Processor] Frame {} doesn't exist. ")
                if logging:
                    log_file.write(", no such frame found. \n")
                continue

            ## Detect barcodes on this frame
            detected_barcodes = self.decode_frame()

            for detected_barcode in detected_barcodes:
                truth = False
                rect = detected_barcode.rect
                rect = (rect.left, rect.top, rect.left + rect.width, rect.top + rect.height)
                for index, true_barcode in enumerate(true_barcodes):
                    ## If we find a true barcode that matches this detection
                    if true_barcode['content'] == detected_barcode.data.decode("utf-8") \
                        and boxes_intersect(true_barcode['loc'], rect):
                        true_posv += 1
                        set_of_distinct_barcodes = set_of_distinct_barcodes | set([true_barcode['id']])
                        set_of_detected_barcodes = set_of_detected_barcodes | set([true_barcode['id']])
                        del true_barcodes[index]
                        truth = True
                        if logging:
                            log_file.write(", true_posv: {}".format(detected_barcode.data.decode("utf-8")))
                        break
                    ## Otherwise, we keep looking through the true barcodes
                    else:
                        set_of_distinct_barcodes = set_of_distinct_barcodes | set([true_barcode['id']])
                ## If none of the true barcodes matched this detection, it was a false positive
                if truth == False:
                    false_posv += 1
                    if logging:
                        log_file.write(", false_posv: {}".format(detected_barcode.data.decode("utf-8")))
            ## Any true barcodes that are still not matched are false negatives
            for true_barcode in true_barcodes:
                false_negv += 1
                if logging:
                    log_file.write(", missed: {}".format(true_barcode['content']))

            if logging:
                log_file.write("\n")

        frame_wise_precision = true_posv / (true_posv + false_posv)
        frame_wise_recall = true_posv / (true_posv + false_negv)
        object_wise_recall = len(set_of_detected_barcodes) / len(set_of_distinct_barcodes)

        print("[Video Processor] Here are the evaluation results: \n" +
              "Frame-wise Precision: {:.9f} \n".format(frame_wise_precision) +
              "Frame-wise Recall:    {:.9f} \n".format(frame_wise_recall) +
              "Object-wise Recall:   {:.3f} \n".format(object_wise_recall))

        if logging:
            log_file.write("Frame-wise Precision: {:.9f} \n".format(frame_wise_precision) +
                           "Frame-wise Recall:    {:.9f} \n".format(frame_wise_recall) +
                           "Object-wise Recall:   {:.3f} \n".format(object_wise_recall))
            log_file.flush()
            log_file.close()

        self.close()
        return (frame_wise_precision, frame_wise_recall, object_wise_recall)
