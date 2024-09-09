
import datetime
import hashlib
import os
import pprint

import rclpy
import rclpy.node

# from geometry_msgs.msg import Quaternion
from rclpy.parameter import Parameter
# from sensor_msgs.msg import Image, NavSatFix, Imu, PointCloud2
from std_msgs.msg import Empty, Bool, String

import subprocess

# from dtcvc_submission.dtcvc_triage_report_generator import DtcvcTriageReportGenerator
"""
  Please check self.file_path which is int the final-saver script : and the manner to retreive the waypoints list
"""

class DtcvcSubmission(rclpy.node.Node):
    # TOPIC_RGB_CAMERA = "/carla/{}/front_rgb/image"
    # TOPIC_IR_CAMERA = "/carla/{}/front_ir/image"

    # TOPIC_GPS = "/carla/{}/gnss"
    # TOPIC_IMU = "/carla/{}/imu"
    # TOPIC_AUDIO = "/carla/{}/audio_file_name"
    
    TOPIC_COMPETITION_START = "/dtc/simulation_start"
    TOPIC_COMPETITION_STOP = "/dtc/simulation_stop"
    TOPIC_COMPETITOR_READY = "/dtc/competitor_ready"
    
    # TOPIC_TRIAGE_REPORT = "/competitor/drone_results"

    def __init__(self):
        super().__init__(
            "dtcvc_submission",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self._info_text = []

        # self.role_name = self.get_parameter_or("role_name", "ego_vehicle")
        # if type(self.role_name) == Parameter:
        #     self.role_name = self.role_name.get_parameter_value().string_value

        # self.latitude = 0
        # self.longitude = 0
        # self.orientation = Quaternion()

        #Initializes Competitor start and stop
        self.init_topics()
        #Initializes Competitor Sensors
        # self.init_sensors()
        #Initializies Competitor Data Collectors


        #Initializes Competitor Reporter
        # self.init_triage_reporting()

        self.is_ready = True
        self.publish_ready_state()

    def init_topics(self):
        
        # Handles competition start signal
        self.comp_start_subscriber = self.log_subscription(
            self.create_subscription(
                Empty,
                DtcvcSubmission.TOPIC_COMPETITION_START,
                self.comp_start_callback,
                qos_profile=10,
            )
        )

        self.original_start_time = -1.0

        # Handles competition stop signal
        self.comp_stop_subscriber = self.log_subscription(
            self.create_subscription(
                Empty,
                DtcvcSubmission.TOPIC_COMPETITION_STOP,
                self.comp_stop_callback,
                qos_profile=10,
            )
        )

        # Publish competitor ready signal
        self.ready_publisher = self.create_publisher(Empty, DtcvcSubmission.TOPIC_COMPETITOR_READY, 10)

        # Periodically log current state
        self.log_timer = self.create_timer(5.0, self.log_info)

    def comp_start_callback(self, msg):
        self.get_logger().info("comp_start_callback(): Received [simulation_start] signal")
        self.original_start_time = self.get_clock().now().nanoseconds / float(10**9)
        self.get_logger().info(f"comp_start_callback(): start_time | {self.original_start_time}")
        # self.start_report_timer()

        self.final_saver_process = subprocess.Popen(["python", "final-saver-sudhin.py"])
        self.make_txt_process = subprocess.Popen(["sh", "model.py"])


    def comp_stop_callback(self, msg):
        self.get_logger().info("comp_stop_callback(): Received [simulation_stop] signal")
        # Perform any necessary cleanup before shutting down
        if self.final_saver_process:
            self.final_saver_process.terminate()
        # if self.make_txt_process:
        #     self.make_txt_process.terminate()

    def publish_ready_state(self):
        self.get_logger().info("publish_ready_state(): Sending [competitor-node] ready signal")
        self.ready_publisher.publish(Empty())

    # def init_sensors(self):
        
    #     self.rgb_subscriber = self.log_subscription(
    #         self.create_subscription(
    #             Image,
    #             DtcvcSubmission.TOPIC_RGB_CAMERA.format(self.role_name),
    #             self.rgb_image_callback,
    #             qos_profile=10,
    #         )
    #     )

    #     self.ir_subscriber = self.log_subscription(
    #         self.create_subscription(
    #             Image,
    #             DtcvcSubmission.TOPIC_IR_CAMERA.format(self.role_name),
    #             self.ir_image_callback,
    #             qos_profile=10,
    #         )
    #     )

    #     self.gps_subscriber = self.log_subscription(
    #         self.create_subscription(
    #             NavSatFix,
    #             DtcvcSubmission.TOPIC_GPS.format(self.role_name),
    #             self.gnss_callback,
    #             qos_profile=10,
    #         )
    #     )

    #     self.imu_subscriber = self.log_subscription(
    #         self.create_subscription(
    #             Imu,
    #             DtcvcSubmission.TOPIC_IMU.format(self.role_name),
    #             self.imu_callback,
    #             qos_profile=10,
    #         )
    #     )

    #     self.audio_subscriber = self.log_subscription(
    #         self.create_subscription(
    #             String, DtcvcSubmission.TOPIC_AUDIO.format(self.role_name), self.on_audio_msg, qos_profile=10
    #         )
    #     )

    # def log_subscription(self, subscription):
    #     self.get_logger().info(f"Subscribed to: {subscription.topic_name}")
    #     return subscription

    # def gnss_callback(self, data: NavSatFix):
    #     """
    #     Callback on GPS sensor updates
    #     """
    #     self.latitude = data.latitude
    #     self.longitude = data.longitude

    # def imu_updated(self, data: Imu):
    #     """
    #     Callback on IMU sensor updates
    #     """
    #     self.orientation = data.orientation

    # def on_audio_msg(self, name: String):
    #     if name.data == "None":
    #         return

    #     path = os.path.join("/mnt/wavs", name.data + ".wav")
    #     try:
    #         sha = hashlib.sha256()
    #         with open(path, "rb") as audio:
    #             while True:
    #                 data = audio.read(64 * 1024)
    #                 if not data:
    #                     break
    #                 sha.update(data)
    #         self.get_logger().info(f"Audio digest: {name.data} yields {sha.hexdigest()}")
    #     except Exception as e:
    #         self.get_logger().info(str(e))
    #         pass

    # def on_rgb_image(self, image: Image):
    #     """
    #     Callback when receiving a camera image
    #     """
    #     self.get_logger().info(f"RGB frame: {image.width}x{image.height} ({image.encoding})")

    # def on_ir_image(self, image: Image):
    #     """
    #     Callback when receiving a camera image
    #     """
    #     self.get_logger().info(f"IR frame: {image.width}x{image.height} ({image.encoding})")

    def log_info(self):
        """
        Update the displayed info text
        """

        time = str(datetime.timedelta(seconds=self.get_clock().now().nanoseconds / float(10**9)))[:10]

        self._info_text = [
            "Simulation time: % 12s" % time,
            "GPS:% 24s" % ("(% 2.6f, % 3.6f)" % (self.latitude, self.longitude)),
            "Orientation: %s" % self.orientation,
        ]

        self.get_logger().info("Info:\n%s" % "\n".join(self._info_text), throttle_duration_sec=1)

    # def init_triage_reporting(self):
    #     self.publisher_ = self.create_publisher(String, "/competitor/drone_results", 10)

        # self.report_generator = DtcvcTriageReportGenerator()

    # def start_report_timer(self):
    #     # Creates a timer to periodically generate dummy reports
    #     timer_period = 10  # seconds
    #     self.report_timer = self.create_timer(
    #         timer_period, lambda: self.report_generator.publish_random(self.publisher_)
    #     )


def main(args=None):
    rclpy.init(args=args)
    node = DtcvcSubmission()

    try:
        rclpy.spin(node)
    except SystemExit:
        print("main(): SystemExit caught in the [competitor-node]")
    except KeyboardInterrupt:
        print("main(): KeyboardInterrupt caught in the [competitor-node]")
    finally:
        print("main(): Awaiting shutdown for the [competitor-node]")


if __name__ == "__main__":
    main()