
import json
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
class DtcvcScoringTestCompetitor(Node):

    def __init__(self,report):
        super().__init__("dtcvc_scoring_test_competitor_node")
        self.triage_report_publisher = self.create_publisher(String, "/competitor/drone_results", 10)
        self.publish_report(report)
    def publish_report(self,report):
        msg = String()
        msg.data = json.dumps(report)
        self.triage_report_publisher.publish(msg)
      
        
        
        
        
def pub(report,args=None):
    print("Starting the publisher")
    rclpy.init(args=args)

    dtcvc_scoring_test_competitor_node = DtcvcScoringTestCompetitor(report)
    

    rclpy.spin_once(dtcvc_scoring_test_competitor_node , timeout_sec=10)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    dtcvc_scoring_test_competitor_node.destroy_node()
    rclpy.shutdown()
