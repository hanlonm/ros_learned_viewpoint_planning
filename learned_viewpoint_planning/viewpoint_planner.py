from typing import List
from pytransform3d import transformations as pt
import numpy as np
import pycolmap
import trimesh
from pathlib import Path
import pickle

import rclpy
from rclpy.context import Context
from rclpy.node import Node

from viewpoint_planning.planning.viewpoint_planning import HlocViewpointPlanner
from viewpoint_planning.viewpoint import Camera


from rclpy.parameter import Parameter



class ViewpointPlanningNode(Node):
    def __init__(self) -> None:
        super().__init__("viewpoint_planner")
        print('Hi from the viewpoint planner.')

        environment = "00195"

        home_dir = Path("/workspace")
        environment_data = home_dir / ('viewpoint_planning/data/' + environment)

        T_map_world = pt.transform_from(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([0, 0, 0.0]))

        planner_config = {
        "criteria": "trf_occ",
        "viz": False,
        "height_difference": 1.1,
        "allowed_rotation_deg": 360,
        "random_angles": False,
        "towards_next_waypoint": False,
        }

        camera = Camera(width=1280,
                    height=720,
                    fx=609.5238037109375,
                    fy=610.1694946289062,
                    cx=640,
                    cy=360)


        reconstruction_dir = str(
        home_dir
        ) + "/Hierarchical-Localization/outputs/{}/reconstruction".format(
            environment)

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

        self.planner = HlocViewpointPlanner(
                path=None,
                path_viewpoints=None,
                reconstruction=reconstruction,
                T_map_world=T_map_world,
                camera=camera,
                landmark_dict=landmark_dict,
                dino_dict=dino_dict,
                min_max_viewing_angles=min_max_viewing_angles,
                min_max_viewing_distances=min_max_viewing_distances,
                config=planner_config,
                rc_scene=scene)
        

def main(args=None):
    rclpy.init(args=args)
    print('Hi from learned_viewpoint_planning.')

    viewpoint_planner = ViewpointPlanningNode()

    rclpy.spin(viewpoint_planner)

    viewpoint_planner.destroy_node()
    rclpy.shutdown()

    


if __name__ == '__main__':
    main()
