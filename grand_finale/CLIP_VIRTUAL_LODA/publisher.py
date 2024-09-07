"""Will publish a single report and then spin forever"""

import json
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from report_generator import DtcvcTriageReportGenerator


class DtcvcScoringTestCompetitor(Node):

    def __init__(self,casuality, lat , lon ,alt , obs_start,obs_end,trauma):
        super().__init__("dtcvc_scoring_test_competitor_node")
        self.generator = DtcvcTriageReportGenerator(casuality, lat , lon ,alt , obs_start,obs_end,trauma)
        self.triage_report_publisher = self.create_publisher(String, "/competitor/drone_results", 10)

        self.generate_reports(1,casuality, lat , lon ,alt , obs_start,obs_end,trauma)


    def generate_reports(self, drone_id: int=2,casuality=1, lat =1, lon =1,alt=1 , obs_start=1,obs_end=1,trauma={}):
        """Generate each of the victim reports once.

        Args:
            drone_id (int): The ID of the drone
            casualty_id (int): The ID of the casualty
        """
        time.sleep(3)
        valid_report = self.generator.generate_valid_victim_report(drone_id ,casuality, lat , lon ,alt , obs_start,obs_end,trauma)
        self.get_logger().info(f"valid_report={valid_report}")
        self.triage_report_publisher.publish(String(data=json.dumps(valid_report)))
        time.sleep(3)
        # casualty_report = self.generator.generate_random_victim_vitals_only(drone_id=drone_id, casualty_id=casualty_id)
        # self.get_logger().info(str(casualty_report))
        # self.triage_report_publisher.publish(String(data=json.dumps(casualty_report)))
        # time.sleep(3)
        # casualty_report = self.generator.generate_random_victim_injuries_only(
        #     drone_id=drone_id, casualty_id=casualty_id
        # )
        # self.get_logger().info(str(casualty_report))
        # self.triage_report_publisher.publish(String(data=json.dumps(casualty_report)))
        # time.sleep(3)
        # casualty_report = self.generator.generate_random_victim_complete_report(
        #     drone_id=drone_id, casualty_id=casualty_id
        # )
        # self.get_logger().info(str(casualty_report))
        # self.triage_report_publisher.publish(String(data=json.dumps(casualty_report)))


def pub(casuality, lat , lon ,alt , obs_start,obs_end,trauma,args=None):
    print("Starting the publisher")
    rclpy.init(args=args)

    dtcvc_scoring_test_competitor_node = DtcvcScoringTestCompetitor(casuality, lat , lon ,alt , obs_start,obs_end,trauma)

    rclpy.spin_once(dtcvc_scoring_test_competitor_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    dtcvc_scoring_test_competitor_node.destroy_node()
    rclpy.shutdown()
