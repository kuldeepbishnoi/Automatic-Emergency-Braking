#Emergency Brake Event Detection v1.0.0
'''
@Emergency Brake Event Detection
@Author : Kuldeep
'''

class EmergencyBrakeEventDetector():
    
    def __init__(self,  columns_names = ['Vehicle_No', 'Top_Left_x', 'Top_Left_y', 'Bottom_Right_x', 'Bottom_Right_y']):
        """
        Constructor to initialize tracked vehicles dataframe's column names
        :columns_names: List of output dataframe column names
        :return: 
        """
        self.columns_names = columns_names
    
    def initialize_new_tracker(self):
        """
        To initialize a new tracker for EBS
        :return:
        """
        self.tracker = {}
        self.tracker_center = {}
        self.emergency_brake_event_flags = {}
    
    def detect_emergency_brake_event(self, y_list, ego_left_x_list, ego_right_x_list, detected_vehicles):
        """
        To detect whether to apply emergency brake or not
        :y_list: list of y coordinates of frame
        :ego_left_x_list: list of ego lane's left coordinates
        :ego_right_x_list: list of ego lane's right coordinates
        :detected_vehicles: The dataframe with tracked vehicle information of current frame
        :return: tuple containing emergency_brake_event_flags, emergency_brake_event, ego_veh_list
        """
        if len(y_list) == 0:
            return

        detected_vehicles = detected_vehicles[(detected_vehicles[self.columns_names[4]] > y_list[0])]
        
        tracker = {}
        tracker_center = {}
        emergency_brake_event_flags = {}
        emergency_brake_event = []
        ego_veh_list = []
        for index in detected_vehicles.index:
            vehicle_no = detected_vehicles[self.columns_names[0]][index]
            y = detected_vehicles[self.columns_names[4]][index]
            left_x = detected_vehicles[self.columns_names[1]][index]
            right_x = detected_vehicles[self.columns_names[3]][index]
            ego_left = ego_left_x_list[y_list.index(y)]
            ego_right = ego_right_x_list[y_list.index(y)]
            vehicle_centre = (right_x + left_x) /2
            if right_x == -1 or left_x == -1:
                vehicle_lane = 'NA'
            elif right_x <= ego_left :
                vehicle_lane = 'Left'
                vehicle_centre_lane = 'Left'
            elif left_x >= ego_right:
                vehicle_lane = 'Right'
                vehicle_centre_lane = 'Right'
            elif(left_x >= ego_left and right_x <= ego_right) or (left_x <= ego_left and right_x >= ego_right):
                vehicle_lane = 'Ego'
                vehicle_centre_lane = 'Ego'
            elif left_x <= ego_left and right_x <= ego_right:
                vehicle_lane = 'Left-Ego'
                if vehicle_centre > ego_left:
                    vehicle_centre_lane = 'Ego'
                else:
                    vehicle_centre_lane = 'Left'
            elif left_x >= ego_left and right_x >= ego_right:
                vehicle_lane = 'Right-Ego'
                if vehicle_centre< ego_right:
                    vehicle_centre_lane = 'Ego'
                else:
                    vehicle_centre_lane = 'Right'
            else:
                vehicle_lane = 'NA'
            
            if vehicle_centre_lane == 'Ego':
                ego_veh_list.append(vehicle_no)
            
            if vehicle_no not in self.tracker:
                tracker[vehicle_no] = (vehicle_lane, 1)
            elif self.tracker[vehicle_no][0] == vehicle_lane:
                tracker[vehicle_no] = (vehicle_lane, self.tracker[vehicle_no][1] + 1)
            else:
                if (self.tracker[vehicle_no][0] == 'Left-Ego' or self.tracker[vehicle_no][0] == 'Right-Ego') \
                and vehicle_lane == 'Ego' and self.tracker[vehicle_no][1] >= 5:
#                     print(vehicle_no, '********Vehicle-Came-In-Ended********')
                    if vehicle_no in self.emergency_brake_event_flags:
                        if self.tracker[vehicle_no][0][:4] == self.emergency_brake_event_flags[vehicle_no][:4]:
                            emergency_brake_event.append((vehicle_no, self.emergency_brake_event_flags[vehicle_no], self.tracker[vehicle_no][1]))
                        self.emergency_brake_event_flags.pop(vehicle_no)
                tracker[vehicle_no] = (vehicle_lane, 1)
                 
            if vehicle_no not in self.tracker_center:
                tracker_center[vehicle_no] = (vehicle_centre_lane, 1)
            elif self.tracker_center[vehicle_no][0] == vehicle_centre_lane:
                tracker_center[vehicle_no] = (vehicle_centre_lane, self.tracker_center[vehicle_no][1] + 1)
            else:
                if (self.tracker_center[vehicle_no][0] == 'Left' or self.tracker_center[vehicle_no][0] == 'Right') \
                    and vehicle_centre_lane == 'Ego' and self.tracker_center[vehicle_no][1] >= 5:
#                     print(vehicle_no, '********Vehicle-Came-In-Started**')
                    if vehicle_no not in self.emergency_brake_event_flags:
                        self.emergency_brake_event_flags[vehicle_no] = self.tracker_center[vehicle_no][0]
                tracker_center[vehicle_no] = (vehicle_centre_lane, 1)
        self.tracker_center = tracker_center                 
        self.tracker = tracker
        emergency_brake_event_vehicles = list(self.emergency_brake_event_flags.keys())
        for emergency_brake_event_vehicle in emergency_brake_event_vehicles:
            if emergency_brake_event_vehicle not in self.tracker:
                self.emergency_brake_event_flags.pop(emergency_brake_event_vehicle)
#         print(f'emergency_brake_event_flags : {self.emergency_brake_event_flags}')
#         print(f'emergency_brake_event : {emergency_brake_event}')
#         print('---------------------------')
        return (self.emergency_brake_event_flags, emergency_brake_event, ego_veh_list)                  
      