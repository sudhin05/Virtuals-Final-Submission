import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from cv_bridge import CvBridge
import time
import yaml
import json

class DataSaver(Node):
    def __init__(self):
        super().__init__('data_saver')
        
        self.gnss_subscriber = self.create_subscription(
            NavSatFix,
            '/carla/dtc_vehicle/gnss',
            self.gnss_callback,
            10
        )
        
        self.rgb_image_subscriber = self.create_subscription(
            Image,
            '/carla/dtc_vehicle/front_rgb/image',
            self.rgb_image_callback,
            10
        )

        self.file_path = '/home/uas-dtu/Desktop/final_docker_maybe/dtcvc-public-7e4ce49-r20240822/dtcvc/deployment/config/scenario.example.yaml'
        with open(self.file_path, 'r') as file:
            self.data = yaml.safe_load(file)
        self.waypoints = self.data['waypoints']
        self.zone_list = [waypoint['zone'] for waypoint in self.waypoints]

        self.bridge = CvBridge()
        
        self.consecutive_stability_threshold = 10 
        self.stability_count = 0  
        
        self.current_lat = None
        self.current_lon = None
        self.current_alt = None

        self.last_lat = None
        self.last_lon = None
        self.last_alt = None

        self.current_sec = None
        self.last_sec = None

        self.start_time = None
        self.end_time = None
        
        self.type_occular = "ocular"
        self.saving_data = False
        self.latest_rgb_images = [] 

        self.ctr = 1

        self.image_save_dir = "/home/uas-dtu/SudhinDarpa/images"
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)
        
        self.frames_to_discard = 5  
        self.occular_threshold = 50

    def gnss_callback(self, msg):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self.current_alt = msg.altitude
        self.current_sec = msg.header.stamp.sec
        
        if self.last_lat is None or self.last_lon is None:
            self.last_lat = self.current_lat
            self.last_lon = self.current_lon
            self.last_alt = self.current_alt
            self.last_sec = self.current_sec
            return

        if self.current_lat == self.last_lat and self.current_lon == self.last_lon:
            self.stability_count += 1
        else:
            self.stability_count = 0
            self.last_lat = self.current_lat
            self.last_lon = self.current_lon
            self.last_alt = self.current_alt
            self.last_sec = self.current_sec

        if self.stability_count >= self.consecutive_stability_threshold:
            if not self.saving_data:
                self.saving_data = True
                self.start_time = self.current_sec
                self.get_logger().info("GPS is stable, starting to save data.")
                
        if len(self.latest_rgb_images) == self.occular_threshold:
            print(f"{len(self.latest_rgb_images)} images saved")
            self.end_time = self.current_sec
            self.create_metadata_file(self.type_occular,self.folder_name)
            print("ocular_metadata.json")

        if self.saving_data and self.stability_count < self.consecutive_stability_threshold:
            self.reset_saving_data()

    def create_casualty_folder(self):
        self.folder_name = f"waypt{self.ctr}_{self.zone_list[self.ctr-1]}_{self.current_lat}_{self.current_lon}_{self.current_alt}_{self.start_time}"
        self.current_casualty_folder = os.path.join(self.image_save_dir, self.folder_name)
        self.ctr += 1

        if not os.path.exists(self.current_casualty_folder):
            os.makedirs(self.current_casualty_folder)

    def create_metadata_file(self,type,name):
        metadata = {
            "observation_start": float(self.start_time) if self.start_time else None,
            "observation_end": float(self.end_time) if self.end_time else None,
            "assessment_time": float(self.end_time) if self.end_time else None,
            "casualty_id": int(self.zone_list[self.ctr-1]),  
            "drone_id": 0,  
            "location": {
                "lon": float(self.current_lon) if self.current_lon else None,
                "lat": float(self.current_lat) if self.current_lat else None,
                "alt": float(self.current_alt) if self.current_alt else None
            },
            "vitals": {
                "heart_rate": None, 
                "respiration_rate": None 
            },
            "injuries": {
                "severe_hemorrhage": None, 
                "respiratory_distress": None,  
                "trauma": None,  
                "alertness": {
                    "ocular": None, 
                    "verbal": None, 
                    "motor": None 
                }
            }
        }
        metadata_filename = os.path.join(self.image_save_dir, f'{name}.json')
        with open(metadata_filename, 'w') as file:
            json.dump(metadata, file, indent=4)

    def rgb_image_callback(self, msg):
        if self.saving_data:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb_images.append(cv_image)
            
            if len(self.latest_rgb_images) == self.frames_to_discard + 1:
                self.create_casualty_folder()

            if len(self.latest_rgb_images) > self.frames_to_discard:
                image_filename = os.path.join(self.current_casualty_folder, f"image_{len(self.latest_rgb_images):05d}.jpg")
                cv2.imwrite(image_filename, cv_image)

    def reset_saving_data(self):
        self.get_logger().info("GPS is no longer stable, stopping data saving.")
        self.saving_data = False
        self.stability_count = 0
        self.end_time = self.current_sec
        self.latest_rgb_images = []

def main(args=None):
    rclpy.init(args=args)
    data_saver = DataSaver()
    rclpy.spin(data_saver)
    data_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
