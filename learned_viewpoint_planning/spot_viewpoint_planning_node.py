import rclpy
from rclpy.context import Context
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from typing import List
from pytransform3d import transformations as pt
import numpy as np
# from viewpoint_planning_interfaces.srv import HandPose
from spot_msgs.srv import HandPose
from std_srvs.srv import Empty


class SpotViewpointPlanningNode(Node):
    def __init__(self) -> None:
        super().__init__("spot_viewpoint_planner")
        
        self.move_spot_sub = self.create_subscription(PoseStamped, "move_spot_relay", self.move_spot_cb, 10)
        self.move_spot_pub = self.create_publisher(PoseStamped, "/spot/go_to_pose", 10)

        self.hand_client = self.create_client(HandPose, '/spot/gripper_pose')
        self.test_arm_service = self.create_service(Empty, 'test_arm', self.test_arm_cb)

    def move_spot_cb(self, msg: PoseStamped):
        msg.header.stamp = self.get_clock().now().to_msg()
        self.move_spot_pub.publish(msg)
    

    def test_hand_move(self):
        request = HandPose.Request()
        request.duration = 5.0
        request.frame = "body"
        pose = Pose()
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        pose.position.x = 0.9
        pose.position.y = 0.1
        pose.position.z = 0.2
        request.pose_point: Pose = pose

        future = self.hand_client.call(request)
    
    def test_arm_cb(self, req: Empty.Request):
        self.test_hand_move()



def main(args=None):
    rclpy.init(args=args)
    print('Hi from learned_viewpoint_planning.')

    spot_viewpoint_planner = SpotViewpointPlanningNode()

    rclpy.spin(spot_viewpoint_planner)

    spot_viewpoint_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()