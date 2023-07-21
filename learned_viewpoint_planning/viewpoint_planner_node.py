import rclpy
from rclpy.context import Context
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from viewpoint_planning_interfaces.srv import ViewpointPlanningPoses

from typing import List
from pytransform3d import transformations as pt
import numpy as np
import pycolmap
import trimesh
from pathlib import Path
import pickle

from viewpoint_planner import ViewpointPlanner, PlannerModes
from viewpoint_planning.viewpoint import Camera


from rclpy.parameter import Parameter


class ViewpointPlanningNode(Node):
    def __init__(self) -> None:
        super().__init__("viewpoint_planner")
        print('Hi from the viewpoint planner.')

        self.declare_parameter("home_directory", "/home")
        home_directory = self.get_parameter(
            'home_directory').get_parameter_value().string_value

        self.declare_parameter("map_name", "map_00195")
        map_name = self.get_parameter(
            'map_name').get_parameter_value().string_value
        environment = map_name.split(sep="_")[-1]

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

        # self.plan_poses_srv = self.create_service(
        #     ViewpointPlanningPoses,
        #     planning_service_name,
        #     self.plan_poses_callback
        # )

        self.plan_poses_sub = self.create_subscription(
            PoseArray, planning_topic_name, self.plan_poses_callback, 10)
        self.result_pose_publisher = self.create_publisher(
            PoseArray, 'viewpoint_poses', 10)

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
        scene: trimesh.Trimesh = trimesh.load(
            str(environment_data) + f"/{environment}_mesh.ply")

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

        # test_list = []
        # for i in range(10):
        #     test_list.append(pt.random_transform())
        # print(self.planner.plan_viewpoints(test_list))

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

        viewpoint_poses = []
        for result in viewpoint_dict.keys():
            T_cam_map = viewpoint_dict[result]["T_cam_map"]
            # T_cam_base = viewpoint_dict[result]["T_cam_base"]

            T_cam_base = pt.transform_from(
                np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0,
                                                               0.0]]),
                np.array([0, 0.0, 0.0]))
            T_base_cam = np.linalg.inv(T_cam_base)
            T_map_base = np.linalg.inv(T_base_cam @ T_cam_map)
            T_base_map = np.linalg.inv(T_map_base)

            # T_base_cam = np.linalg.inv(T_cam_base)
            # T_base_cam_orig = pt.invert_transform(T_cam_base_orig)

            # # T_base_map = pt.invert_transform(T_cam_base) @ T_cam_map
            # T_base_map = T_base_cam_orig @ T_cam_map
            print(T_map_base)
            viewpoint_poses.append(self.transform_to_pose(T_map_base))

        result_pose_array = self.create_pose_array(viewpoint_poses)

        self.result_pose_publisher.publish(result_pose_array)

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