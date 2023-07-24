import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_srvs.srv import Empty

class PoseArrayPublisher(Node):
    def __init__(self):
        super().__init__('pose_array_publisher')
        self.publisher_ = self.create_publisher(PoseArray, 'planning_poses', 10)
        self.service_ = self.create_service(Empty, 'publish_poses', self.trigger_service_callback)
        #self.timer_ = self.create_timer(1.0, self.publish_pose_array)
        self.pose_array_msg_ = PoseArray()

    def trigger_service_callback(self, request, response):
        self.publish_pose_array()
        return response

    def publish_pose_array(self):
        # Populate the PoseArray message
        # In this example, we'll create a simple list of two poses
        pose1 = Pose()
        pose1.position.x = -8.0
        pose1.position.y = -8.0
        pose1.position.z = 0.7
        pose1.orientation.w = 1.0
        pose1.orientation.x = 0.0
        pose1.orientation.y = 0.0
        pose1.orientation.z = 0.0

        pose2 = Pose()
        pose2.position.x = -8.0
        pose2.position.y = -9.0
        pose2.position.z = 0.7
        pose2.orientation.w = 1.0
        pose2.orientation.x = 0.0
        pose2.orientation.y = 0.0
        pose2.orientation.z = 0.0

        pose3 = Pose()
        pose3.position.x = -8.0
        pose3.position.y = -10.0
        pose3.position.z = 0.7
        pose3.orientation.w = 1.0
        pose3.orientation.x = 0.0
        pose3.orientation.y = 0.0
        pose3.orientation.z = 0.0

        self.pose_array_msg_.header.stamp = self.get_clock().now().to_msg()
        self.pose_array_msg_.header.frame_id = 'map'
        self.pose_array_msg_.poses = [pose1, pose2, pose3]

        self.publisher_.publish(self.pose_array_msg_)

def main(args=None):
    rclpy.init(args=args)
    pose_array_publisher = PoseArrayPublisher()
    rclpy.spin(pose_array_publisher)
    pose_array_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()