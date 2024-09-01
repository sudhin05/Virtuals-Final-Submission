import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from ultralytics import YOLO

ctr = 0
class YOLOImageSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/dtc/BP_AirVehicle_C_2147482163/BP_CameraSensor_RGB/image',
            self.listener_callback,
            10)
        self.bridge = CvBridge()

        self.frame_count = 0
        self.save_path = "FrameFolder"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def listener_callback(self, msg):
        self.get_logger().info('Received an image!')
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Save to mainimages folder before YOLO processing
            self.save_cropped_frames(frame, self.save_path)
            self.frame_count +=1

            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

    
    def save_cropped_frames(self, frame, save_path):
        filename = os.path.join(save_path, f"frame_{self.frame_count:04d}.jpg")
        cv2.imwrite(filename, frame)

def main(args=None):
    rclpy.init(args=args)
    yolo_image_subscriber = YOLOImageSubscriber()
    rclpy.spin(yolo_image_subscriber)
    yolo_image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()