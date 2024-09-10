import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from cv_bridge import CvBridge
import time
import yaml
import subprocess

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

        # self.file_path = 'scenario.example.yaml'
        # with open(self.file_path, 'r') as file:
        #     self.data = yaml.safe_load(file)
        # self.waypoints = self.data['waypoints']
        self.zone_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

        self.bridge = CvBridge()
        
        self.consecutive_stability_threshold = 10 
        self.stability_count = 0  
        # self.ctr_1 = True
        
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

        self.saving_data = False
        self.latest_rgb_images = [] 

        self.ctr = 1

        self.image_save_dir = "/media/images"
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)
        
        self.frames_to_discard = 5  
        self.trauma_threshold = 80
        

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
                
        # if len(self.latest_rgb_images) == self.trauma_threshold:
        #     # if self.ctr_1:
        #     print(f"{len(self.latest_rgb_images)} images saved")
        #     # self.ctr_1 = False
        #     self.end_time = self.current_sec
        #     self.new_folder = f"{self.current_casualty_folder}_{self.end_time}"
        #     os.rename(self.current_casualty_folder, self.new_folder)
            # print("STARTING FINAL SCRIPT OF TRAUMA")
            # subprocess.Popen(["grand_finale/final.sh", str(self.new_folder)])
        
        if self.saving_data and self.stability_count < self.consecutive_stability_threshold:
            self.reset_saving_data()

    def create_casualty_folder(self):
        folder_name = f"waypt{self.ctr}_{self.zone_list[self.ctr-1]}_{self.current_lat}_{self.current_lon}_{self.current_alt}_{self.start_time}"
        self.current_casualty_folder = os.path.join(self.image_save_dir, folder_name)
        self.ctr += 1

        if not os.path.exists(self.current_casualty_folder):
            os.makedirs(self.current_casualty_folder)

    def rgb_image_callback(self, msg):
        if self.saving_data:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb_images.append(cv_image)
            
            if len(self.latest_rgb_images) == self.frames_to_discard + 1:
                self.create_casualty_folder()

            if len(self.latest_rgb_images) > self.frames_to_discard:
                image_filename = os.path.join(self.current_casualty_folder, f"image_{len(self.latest_rgb_images):05d}_{self.current_sec}.jpg")
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
