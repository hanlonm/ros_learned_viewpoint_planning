import rclpy
from rclpy.context import Context
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from typing import List
from pytransform3d import transformations as pt
import numpy as np


class SpotViewpointPlanningNode(Node):
    def __init__(self) -> None:
        super().__init__("spot_viewpoint_planner")
        
        self.move_spot_sub = self.create_subscription(PoseStamped, "move_spot_relay", self.move_spot_cb, 10)
        self.move_spot_pub = self.create_publisher(PoseStamped, "/spot/go_to_pose", 10)

    def move_spot_cb(self, msg: PoseStamped):
        msg.header.stamp = self.get_clock().now().to_msg()
        self.move_spot_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    print('Hi from learned_viewpoint_planning.')

    spot_viewpoint_planner = SpotViewpointPlanningNode()

    rclpy.spin(spot_viewpoint_planner)

    spot_viewpoint_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()