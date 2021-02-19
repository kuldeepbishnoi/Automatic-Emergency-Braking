# @Main Function
# @Author : Kuldeep

import matplotlib.pyplot as plt
import glob
import os
import tqdm
import cv2
import configuration as config
from LaneNet.lane_handler import Lane_Detection
from DeepSORT.vehicle_handler import VehicleTracker
from Helper.ebs_handler import EmergencyBrakeEventDetector

if __name__ == "__main__":
    
    # Giving Weights to DeepSORT & LaneNet
    deepSORT_weight_path = config.deepSORT_weight_path
    laneNet_weight_path = config.laneNet_weight_path

    lane_detection = Lane_Detection(laneNet_weight_path)
    vehicle_tracker = VehicleTracker(deepSORT_weight_path)
    
    # Input & Output path initalization
    input_image_dir = config.input_image_dir
    output_image_dir = config.output_image_dir
    output_video_dir = config.output_video_dir
    sign_image_dir = config.sign_image_dir
    video_name = config.video_name

    sub_dirs = [o for o in os.listdir(input_image_dir) if os.path.isdir(os.path.join(input_image_dir,o))]

    #Frame to be skiped
    skip_factor = 5

    #turn-Left sign, turn-right sign & stop sign Initalization
    right_to_ego_img = cv2.imread(os.path.join(sign_image_dir, 'turn-left.png'))
    left_to_ego_img = cv2.imread(os.path.join(sign_image_dir, 'turn-right.png'))
    stop_img = cv2.imread(os.path.join(sign_image_dir, 'stop.png'))

    emergencyBrakeEventDetector = EmergencyBrakeEventDetector()

    for sub_dir in sub_dirs:
        #initalizing trackers
        vehicle_tracker.initialize_new_tracker()
        emergencyBrakeEventDetector.initialize_new_tracker()

        #making output folders
        output_image_sub_dir = os.path.join(output_image_dir, sub_dir)
        output_video_sub_dir = os.path.join(output_video_dir, sub_dir)
        os.makedirs(output_image_sub_dir, exist_ok=True)
        os.makedirs(output_video_sub_dir, exist_ok=True)

        #gettin input image list
        image_list = glob.glob('{:s}/*.jpeg'.format(os.path.join(input_image_dir, sub_dir)), recursive=False)
        #initalizing lanes
        all_lane_dict = -1
        y_list, all_x_list = -1, -1
        ego_left_x_list, ego_right_x_list = -1, -1
        last_ego_left_x_list, last_ego_right_x_list, last_y_list =-1, -1, []

        #initalizing detected vehicle's dictionary
        all_detected_vehicles = {}

        #initalizing videowriter
    
        height, width, layers = cv2.imread(image_list[0]).shape
        video = cv2.VideoWriter(os.path.join(output_video_sub_dir,video_name), 0, 4, (width, height))    

        for index, image_path in tqdm.tqdm(enumerate(image_list[:]), total=len(image_list[:])):
            if index%skip_factor != (skip_factor-1):
                continue
            print(f'Input Image Path : {image_path}')
            frame = cv2.imread(image_path)

            #vehicle detection
            detected_vehicles = vehicle_tracker.get_vehicles(image_path)
            frame = vehicle_tracker.plot_detected_vehicles(frame, detected_vehicles)

            #lane detection
            all_lane_dict = lane_detection.test_lanenet(image_path)
            y_list, all_x_list = lane_detection.extrapolate_lanes(all_lane_dict)
            ego_left_x_list, ego_right_x_list = lane_detection.ego_lane_extraction(y_list, all_x_list)
            ego_left_x_list, ego_right_x_list = lane_detection.ego_lane_tracking(last_ego_left_x_list, last_ego_right_x_list, ego_left_x_list, ego_right_x_list)
            if len(last_y_list) > 0:
                y_list = last_y_list
            last_y_list = y_list
            last_ego_left_x_list, last_ego_right_x_list = ego_left_x_list, ego_right_x_list
            frame = lane_detection.plot_ego_path(frame, y_list, ego_left_x_list, ego_right_x_list)

            cv2.putText(frame, os.path.basename(image_path), (500, 30), 0, 5e-3 * 200, (0, 0, 255), 5)

            #EBS
            emergency_brake_event_details = emergencyBrakeEventDetector.detect_emergency_brake_event(y_list, ego_left_x_list, ego_right_x_list, detected_vehicles)

            if len(emergency_brake_event_details[0]) > 0:
                for emergency_brake_event_detail in list(emergency_brake_event_details[0].keys()):
                    if int(detected_vehicles[detected_vehicles['Vehicle_No'] ==  emergency_brake_event_detail]['Bottom_Right_y']) > 0:
                        if emergency_brake_event_details[0][emergency_brake_event_detail] == 'Left':
                            frame[50:178,200:328,:] = left_to_ego_img
                        elif emergency_brake_event_details[0][emergency_brake_event_detail] == 'Right':
                            frame[50:178,952:1080,:] = right_to_ego_img

            if len(emergency_brake_event_details[1]) > 0:
                for emergency_brake_event_detail in emergency_brake_event_details[1]:
                    print(f'Vehicle no {emergency_brake_event_detail[0]} came from {emergency_brake_event_detail[1]} side during frame no {index - (emergency_brake_event_detail[2]*skip_factor)} to frame no {index}')

            for vehicle_no in emergency_brake_event_details[2]:
                if int(detected_vehicles[detected_vehicles['Vehicle_No'] == vehicle_no]['Bottom_Right_y']) > 300:
                    frame[50:178,576:704,:] = stop_img
                    print(f'Vehicle no {vehicle_no} is close to us')

            plt.figure('Detected Vehicles and Ego Lanes')
            plt.imshow(frame[:, :, (2, 1, 0)])
            plt.show(block=False)
            plt.pause(2)
            plt.close()
            cv2.imwrite(os.path.join(output_image_sub_dir, os.path.basename(image_path)), frame)
            print(f'Output Image Path : {os.path.join(output_image_sub_dir, os.path.basename(image_path))}')
            video.write(frame)

        cv2.destroyAllWindows()
        video.release()
        print(f'Output Video Path : {os.path.join(output_image_sub_dir, os.path.basename(image_path))}')