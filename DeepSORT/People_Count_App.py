
from __future__ import division, print_function, absolute_import
import threading
import time
import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

track_list = []
total=0
#App Function 
from flask import Flask
from flask import request ,jsonify
from flask_restplus import reqparse
app = Flask(__name__)
@app.route('/People_Count', methods = ['POST'])
def postJsonHandler():
    global total
    print (request.is_json)
    content = request.get_json()
    path=content['path1']

    #print('Starting of thread :', threading.currentThread().name)
    #time.sleep(5)
    #print('Finishing of thread :', threading.currentThread().name)

    # Counting The People
    
    def Counting_People(path):
        yolo=YOLO()
        global total

        global track_list
        # Definition of the parameters
        max_cosine_distance = 0.3
        nn_budget = None
        nms_max_overlap = 1.0

        # deep_sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        writeVideo_flag = True

        video_capture = cv2.VideoCapture(path)

        if writeVideo_flag:
            # Define the codec and create VideoWriter object
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
            list_file = open('detection.txt', 'w')
            frame_index = -1

        fps = 0.0
        while True:
            ret, frame = video_capture.read()  # frame shape 640*480*3
            if ret != True:
                break;
            t1 = time.time()

            image = Image.fromarray(frame)
            boxs = yolo.detect_image(image)
            # print("box_num",len(boxs))
            features = encoder(frame, boxs)

            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update > 1:
                    continue
                    # count=0
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
                # count=count+1
                # track.track_id=1
                list1 = track.track_id
                # print(track_list)

                track_list.append(track.track_id)
                # for t in track_list:
                # print("Debugging",t)
                # else:

                #     for var in list1:
                #         print("Debugging for TrackId value",var)
                #     my_list=[]
                # print(track.track_id)

            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

            cv2.imshow('', frame)

            if writeVideo_flag:
                # save a frame
                out.write(frame)
                frame_index = frame_index + 1
                # count=0
                # count=count+1
                # print(count)
                list_file.write(str(frame_index) + ' ')

                if len(boxs) != 0:
                    for i in range(0, len(boxs)):
                        list_file.write(str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(
                            boxs[i][3]) + ' ')
                list_file.write('\n')

            fps = (fps + (1. / (time.time() - t1))) / 2
            # print (my_list)
            print("fps= %f" % (fps))

            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                brea

        video_capture.release()
        if writeVideo_flag:
            out.release()
            list_file.close()
        cv2.destroyAllWindows()
        # print(frame_index)
        # print(list_file)
        # print(track_list)
        total = (len(set(track_list)))
        print("Total Number of People In Whole Video Are :", total)
        #return jsonify({'Total number of people:' : total})
    Counting_People(path)   
    return jsonify({'Total number of people:' : total})



if __name__ == '__main__':
    
    
    app.run(debug=True)
     

