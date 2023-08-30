import rclpy
from rclpy.context import Context
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, TransformStamped
from std_msgs.msg import String
from typing import List
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import numpy as np
import pycolmap
import trimesh
from pathlib import Path
import pickle
from sensor_msgs.msg import PointCloud2, PointField
from viewpoint_planner import ViewpointPlanner, PlannerModes
from viewpoint_planning.viewpoint import Camera
from std_msgs.msg import ColorRGBA, Header, Bool



from rclpy.parameter import Parameter


class ViewpointPlanningNode(Node):
    def __init__(self) -> None:
        super().__init__("viewpoint_planner")
        print('Hi from the viewpoint planner.')

        self.declare_parameter("home_directory", "/home")
        home_directory = self.get_parameter(
            'home_directory').get_parameter_value().string_value

        self.declare_parameter("map_name", "map-00195")
        map_name = self.get_parameter(
            'map_name').get_parameter_value().string_value
        environment = map_name.split(sep="-")[-1]

        self.declare_parameter(
            "cam_string", "PINHOLE 1280 720 609.5238037109375 610.1694946289062 640 360")
        cam_string = self.get_parameter(
            'cam_string').get_parameter_value().string_value

        self.declare_parameter(
            "mode_string", "trf_clf")
        mode_string = self.get_parameter(
            'mode_string').get_parameter_value().string_value

        self.declare_parameter(
            "planning_topic_name", "planning_poses")
        planning_topic_name = self.get_parameter(
            'planning_topic_name').get_parameter_value().string_value

        self.declare_parameter("occlusion", False)
        occlusion = self.get_parameter(
            'occlusion').get_parameter_value().bool_value

        self.declare_parameter("num_viewpoint_samples", 100)
        num_viewpoint_samples = self.get_parameter(
            'num_viewpoint_samples').get_parameter_value().integer_value
        
        self.plan_poses_sub = self.create_subscription(
            PoseArray, planning_topic_name, self.plan_poses_callback, 10)
        
        self.plan_single_pose_sub = self.create_subscription(TransformStamped, "plan_viewpoint", self.plan_viewpoint_callback, 10)
        self.viewpoint_result_map_pub = self.create_publisher(PoseStamped, "plan_viewpoint_result_map", 10)
        self.viewpoint_result_loc_cam_pub = self.create_publisher(PoseStamped, "plan_viewpoint_result_loc_cam", 10)
        self.planner_loc_pub = self.create_publisher(PoseStamped, "planner_loc", 10)
        self.mode_sub = self.create_subscription(String, "planner_mode", self.planner_mode_callback, 10)
        
        
        self.result_pose_publisher_map = self.create_publisher(
            PoseArray, 'viewpoint_poses_map', 10)
        
        self.result_pose_publisher_base = self.create_publisher(
            PoseArray, 'viewpoint_poses_base', 10)

        home_dir = Path(home_directory)
        environment_data = home_dir / \
            ('viewpoint_planning/data/' + environment)
        model_dir = home_dir / "viewpoint_planning/data/models"

        _, c_width, c_height, cfx, cfy, cx, cy = cam_string.split()
        camera = Camera(width=int(c_width),
                        height=int(c_height),
                        fx=float(cfx),
                        fy=float(cfy),
                        cx=float(cx),
                        cy=float(cy))

        reconstruction_dir = str(
            home_dir
        ) + f"/Hierarchical-Localization/outputs/{environment}/reconstruction"

        reconstruction = pycolmap.Reconstruction(reconstruction_dir)

        self.pc_publisher = self.create_publisher(PointCloud2, 'planning_pc', 10)

        self.landmark_list = list(reconstruction.points3D.keys())
        self.pcd_points = np.zeros((len(self.landmark_list), 3))
         
        for idx, landmark in enumerate(self.landmark_list):
            self.pcd_points[idx] = np.array(
                reconstruction.points3D[landmark].xyz)
        self.pointcloud_msg = self.point_cloud(points=self.pcd_points, parent_frame="map")
        self.pc_publisher.publish(self.pointcloud_msg)
        
        if occlusion:
            scene: trimesh.Trimesh = trimesh.load(
                str(environment_data) + f"/{environment}_mesh.ply")
        else:
            scene = None

        file = open(environment_data / "pose_dict.pkl", 'rb')
        pose_dict = pickle.load(file)
        file.close()

        file = open(environment_data / "landmark_dict.pkl", 'rb')
        landmark_dict = pickle.load(file)
        file.close()

        file = open(environment_data / "min_max.pkl", 'rb')
        min_max_dict = pickle.load(file)
        file.close()
        min_max_viewing_angles = min_max_dict["angles"]
        min_max_viewing_distances = min_max_dict["distances"]

        try:
            file = open(environment_data / "dino_3_landmark_dict.pkl", 'rb')
            dino_dict = pickle.load(file)
            file.close()
        except:
            print("ERROR")

        planner_mode = PlannerModes(mode_string)

        self.planner = ViewpointPlanner(
            pc_publisher = self.pc_publisher,
            mode=planner_mode,
            occlusion=occlusion,
            num_samples=num_viewpoint_samples,
            reconstruction=reconstruction,
            rc_scene=scene, camera=camera,
            landmark_dict=landmark_dict,
            dino_dict=dino_dict,
            model_dir=str(model_dir),
            min_max_viewing_angles=min_max_viewing_angles,
            min_max_viewing_distances=min_max_viewing_distances)

        self.T_zforward_zup = pt.transform_from(
            np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0,
                                                           0.0]]),
            np.array([0, 0.0, 0.0]))
        self.T_zup_zforward = np.linalg.inv(self.T_zforward_zup)

        self.mode_publisher = self.create_publisher(String, 'current_planner_mode', 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def point_cloud(self, points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx3 array of xyz positions.
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        Code source:
            https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
        """
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.

        data = points.astype(dtype).tobytes()
        fields = [PointField(
            name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]
        header = Header(frame_id=parent_frame)

        return PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            # Every point consists of three float32s.
            point_step=(itemsize * 3),
            row_step=(itemsize * 3 * points.shape[0]),
            data=data
        )

    def timer_callback(self):
        msg = String()
        msg.data = self.planner.mode.value
        self.mode_publisher.publish(msg)
        

    def plan_viewpoint_callback(self, msg: TransformStamped):
        translation = np.array([
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z
        ])
        rotation = np.array([
            msg.transform.rotation.w,
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
        ])
        print(translation)

        planner_pose = PoseStamped()
        planner_pose.header.frame_id = "map"
        planner_pose.header.stamp = self.get_clock().now().to_msg()

        planner_pose.pose.position.x = msg.transform.translation.x
        planner_pose.pose.position.y = msg.transform.translation.y
        planner_pose.pose.position.z = msg.transform.translation.z

        planner_pose.pose.orientation.x = msg.transform.rotation.x
        planner_pose.pose.orientation.y = msg.transform.rotation.y
        planner_pose.pose.orientation.z = msg.transform.rotation.z
        planner_pose.pose.orientation.w = msg.transform.rotation.w

        self.planner_loc_pub.publish(planner_pose)


        pq = np.concatenate([translation, rotation])
        transformation_matrix = pt.invert_transform(pt.transform_from_pq(pq))
        planning_poses = [transformation_matrix]

        viewpoint_dict = self.planner.plan_viewpoints(planning_poses)

        viewpoint_poses_map = []
        viewpoint_poses_base = []

        viewpoint_poses_map_d = []
        viewpoint_poses_base_d = []

        for result in viewpoint_dict.keys():
            T_cam_map = viewpoint_dict[result]["T_cam_map"]
            T_map_cam = np.linalg.inv(self.T_zup_zforward @ T_cam_map)

            T_cam_base = viewpoint_dict[result]["T_cam_base"]
            T_base_cam = np.linalg.inv(self.T_zup_zforward @ T_cam_base)

            viewpoint_poses_map.append(self.transform_to_pose_stamped(T_map_cam, "map"))
            viewpoint_poses_base.append(self.transform_to_pose_stamped(T_base_cam, "hand"))

            viewpoint_poses_map_d.append(self.transform_to_pose(T_map_cam))
            viewpoint_poses_base_d.append(self.transform_to_pose(T_base_cam))
            # print(pt.pq_from_transform(T_base_cam))
            # print(pr.euler_from_quaternion(pt.pq_from_transform(T_base_cam)[3:], 2,1,0, False))
        self.viewpoint_result_map_pub.publish(viewpoint_poses_map[0])
        self.viewpoint_result_loc_cam_pub.publish(viewpoint_poses_base[0])

        result_pose_array_map = self.create_pose_array(viewpoint_poses_map_d)
        result_pose_array_base = self.create_pose_array(viewpoint_poses_base_d)
        self.result_pose_publisher_map.publish(result_pose_array_map)
        self.result_pose_publisher_base.publish(result_pose_array_base)

    def plan_poses_callback(self, msg):
        # Convert each pose in the received request to a 4x4 transformation matrix

        T_cam_base_orig = np.array(
            [[0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, -1.0, 0.0],
             [1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]])
        planning_poses = []
        for pose in msg.poses:

            translation = np.array([
                -pose.position.x,
                -pose.position.y,
                -pose.position.z
            ])
            rotation = np.array([
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,

            ])
            pq = np.concatenate([translation, rotation])
            transformation_matrix = pt.transform_from_pq(pq)
            planning_poses.append(transformation_matrix)

        viewpoint_dict = self.planner.plan_viewpoints(planning_poses)

        viewpoint_poses_map = []
        viewpoint_poses_base = []
        for result in viewpoint_dict.keys():
            T_cam_map = viewpoint_dict[result]["T_cam_map"]
            T_map_cam = np.linalg.inv(self.T_zup_zforward @ T_cam_map)

            T_cam_base = viewpoint_dict[result]["T_cam_base"]
            T_base_cam = np.linalg.inv(self.T_zup_zforward @ T_cam_base)

            viewpoint_poses_map.append(self.transform_to_pose(T_map_cam))
            viewpoint_poses_base.append(self.transform_to_pose(T_base_cam))

        result_pose_array_map = self.create_pose_array(viewpoint_poses_map)
        result_pose_array_base = self.create_pose_array(viewpoint_poses_base)

        self.result_pose_publisher_map.publish(result_pose_array_map)
        self.result_pose_publisher_base.publish(result_pose_array_base)


    def planner_mode_callback(self, msg: String):
        self.planner.change_mode(msg.data)
        print(f"Planner mode changed to {msg.data}")

    def transform_to_pose(self, transformation_matrix):
        pq = pt.pq_from_transform(transformation_matrix)
        pose = Pose()
        pose.position.x = pq[0]
        pose.position.y = pq[1]
        pose.position.z = pq[2]

        pose.orientation.x = pq[4]
        pose.orientation.y = pq[5]
        pose.orientation.z = pq[6]
        pose.orientation.w = pq[3]
        return pose
    
    def transform_to_pose_stamped(self, transformation_matrix, frame):
        pq = pt.pq_from_transform(transformation_matrix)
        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = pq[0]
        pose.pose.position.y = pq[1]
        pose.pose.position.z = pq[2]

        pose.pose.orientation.x = pq[4]
        pose.pose.orientation.y = pq[5]
        pose.pose.orientation.z = pq[6]
        pose.pose.orientation.w = pq[3]
        return pose

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        qw = np.sqrt(
            1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)
        return [qx, qy, qz, qw]

    def create_pose_array(self, poses):
        pose_array = PoseArray()
        pose_array.header.frame_id = 'map'  # Set the frame ID for the PoseArray
        pose_array.poses = poses
        return pose_array


def main(args=None):
    rclpy.init(args=args)
    print('Hi from learned_viewpoint_planning.')

    viewpoint_planner = ViewpointPlanningNode()

    rclpy.spin(viewpoint_planner)

    viewpoint_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
