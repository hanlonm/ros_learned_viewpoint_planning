from typing import List
from enum import Enum
import random
from tqdm import tqdm
import numpy as np
import pycolmap
from viewpoint_planning.utils import viz_3d
from scipy.spatial.transform import Rotation
from pytransform3d import transformations as pt
import pickle
import torch

from viewpoint_planning.viewpoint import Camera, Viewpoint
from viewpoint_planning.planning.sampling import yaw_pitch_samples

from viewpoint_learning.learning.MLP_classifier import ViewpointClassifier
from viewpoint_learning.learning.regression import ViewpointRegressor
from viewpoint_learning.learning.viewpoint_pct import PCTViewpointTransformer
from viewpoint_learning.learning.utils import pre_process


from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header, Bool



DEG_TO_RAD = np.pi / 180


class PlannerModes(Enum):
    MLP_CLF = "mlp_clf"
    MLP_REG = "mlp_reg"
    TRF_CLF = "trf_clf"
    ANGLE_CRIT = "angle_crit"
    MAX_CRIT = "max_crit"
    FISHER_INFO = "fisher_info"
    RANDOM = "random"


class ViewpointPlanner:
    def __init__(self,
                 pc_publisher,
                 mode=PlannerModes.MLP_CLF,
                 occlusion: bool = False,
                 num_samples: int = 100,
                 reconstruction: pycolmap.Reconstruction = None,
                 rc_scene=None,
                 camera: Camera = None,
                 landmark_dict: dict = {},
                 dino_dict: dict = {},
                 model_dir: str = "..",
                 min_max_viewing_angles: np.ndarray = None,
                 min_max_viewing_distances: np.ndarray = None,
                 ) -> None:
        
        self.pc_publisher = pc_publisher
        ### Planner Parameters ###
        self.mode = mode
        self.occlusion = occlusion
        self.num_samples = num_samples
        self.max_pitch = 30

        ### Map Parameters ###
        # colmap reconstruction
        self.reconstruction = reconstruction
        # ray-cast scene
        if self.occlusion:
            self.rc_scene = rc_scene
        else:
            self.rc_scene = None
        # init T_map_world as identity
        self.T_map_world = np.eye(4)
        # camere with parameters
        self.camera = camera
        # dict of mapping dino features
        self.dino_dict = dino_dict
        # dict containing landmark attributes: image ids, pixel locs, heatmaps
        self.landmark_dict = landmark_dict
        # ordered array containing min/max viewing angles
        self.min_max_viewing_angles = min_max_viewing_angles
        # ordered array containing min/max viewing distances
        self.min_max_viewing_distances = min_max_viewing_distances
        # path to location of learned models
        self.model_dir = model_dir

        # extract point cloud from reconstruction
        self.landmark_list = list(self.reconstruction.points3D.keys())
        self.pcd_points = np.zeros((len(self.landmark_list), 3))
        for idx, landmark in enumerate(tqdm(self.landmark_list)):
            self.pcd_points[idx] = np.array(
                self.reconstruction.points3D[landmark].xyz)
        self.pcd_points_hom = np.hstack(
            (self.pcd_points, np.ones((self.pcd_points.shape[0], 1))))
        landmark_array = np.array(self.landmark_list).reshape(
            (len(self.landmark_list), 1))
        self.id_points_hom = np.hstack(
            (landmark_array, self.pcd_points_hom))

        # determine compute device
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else "cpu")
        

        ### Learned Models ###
        self.classifier = None
        self.transformer = None
        self.regressor = None
        if self.mode == PlannerModes.MLP_CLF:
            if occlusion:
                self.classifier = ViewpointClassifier.load_from_checkpoint(
                    self.model_dir + "/classifiers/mlp_opt_occ_1-5_32/best_test.ckpt",
                    input_dim=146)
            else:
                self.classifier = ViewpointClassifier.load_from_checkpoint(
                    self.model_dir +
                    "/MLP_CLF/best_test_mlp_opt_occless_1-5_32.ckpt",
                    input_dim=146)
            self.classifier.to(self.device)
            self.classifier.eval()

        if self.mode == PlannerModes.MLP_REG:
            if occlusion:
                self.regressor = ViewpointRegressor.load_from_checkpoint(
                    self.model_dir +
                    "/regression/230620/all_info_occ/epoch=149-step=133950.ckpt",
                    input_dim=146,
                    max_error=5.0)
            else:
                self.regressor = ViewpointRegressor.load_from_checkpoint(
                    self.model_dir +
                    "/regression/230620/all_info/epoch=106-step=95979.ckpt",
                    input_dim=146,
                    max_error=5.0)
            self.regressor.to(self.device)
            self.classifier.eval()

        if self.mode == PlannerModes.TRF_CLF:
            # TODO: add case for occlusion
            self.transformer = PCTViewpointTransformer.load_from_checkpoint(self.model_dir + "/TRF_CLF/best_test_10-5_16_occ_opt_norm_small_heatmaps.ckpt",
                                                                            dino_dim=0,
                                                                            im_width=self.camera.width,
                                                                            im_height=self.camera.height)
            self.transformer.eval()
            self.transformer.to(self.device)

    def change_mode(self, new_mode):
        self.mode = PlannerModes(new_mode)
        if self.mode == PlannerModes.TRF_CLF and self.transformer is None:
            self.transformer = PCTViewpointTransformer.load_from_checkpoint(self.model_dir + "/TRF_CLF/best_test_10-5_16_occ_opt_norm_small_heatmaps.ckpt",
                                                                            dino_dim=0,
                                                                            im_width=self.camera.width,
                                                                            im_height=self.camera.height)
            self.transformer.eval()
            self.transformer.to(self.device)
        if self.mode == PlannerModes.MLP_CLF and self.classifier is None:
            if self.occlusion:
                self.classifier = ViewpointClassifier.load_from_checkpoint(
                    self.model_dir + "/classifiers/mlp_opt_occ_1-5_32/best_test.ckpt",
                    input_dim=146)
            else:
                self.classifier = ViewpointClassifier.load_from_checkpoint(
                    self.model_dir +
                    "/MLP_CLF/best_test_mlp_opt_occless_1-5_32.ckpt",
                    input_dim=146)
            self.classifier.to(self.device)
            self.classifier.eval()

    def plan_viewpoints(self, waypoints: List[np.ndarray]) -> dict:
        """
        Plan viewpoints based on the provided list of waypoints.

        Args:
            waypoints (List[np.ndarray]): A list of NumPy arrays representing the T_base_map transforms of the waypoints.

        Returns:
            dict: A dictionary containing the results of the planning for each waypoint.
        """
        print("Planning Waypoints")

        result_dict = {}

        for i, waypoint in enumerate(tqdm(waypoints)):
            if self.mode == PlannerModes.ANGLE_CRIT:
                result_dict[i] = self.plan_waypoint_angle_crit(waypoint)

            elif self.mode == PlannerModes.MAX_CRIT:
                result_dict[i] = self.plan_waypoint_max_crit(waypoint)

            elif self.mode == PlannerModes.FISHER_INFO:
                result_dict[i] = self.plan_waypoint_fisher_info(waypoint)

            elif self.mode == PlannerModes.RANDOM:
                result_dict[i] = self.plan_waypoint_random(waypoint)

            elif self.mode == PlannerModes.MLP_CLF:
                result_dict[i] = self.plan_waypoint_mlp_clf(waypoint)

            elif self.mode == PlannerModes.MLP_REG:
                result_dict[i] = self.plan_waypoint_mlp_reg(waypoint)

            elif self.mode == PlannerModes.TRF_CLF:
                result_dict[i] = self.plan_waypoint_trf_clf(waypoint)

            else:
                print("Invalid planner mode! returning empty dict!")
                return {}

        return result_dict

    def plan_waypoint_angle_crit(self, T_base_map: np.ndarray) -> dict:

        samples = yaw_pitch_samples(num=self.num_samples,
                                    max_yaw=2 * np.pi,
                                    min_pitch=-10 * DEG_TO_RAD,
                                    max_pitch=self.max_pitch * DEG_TO_RAD)

        max_seen_landmarks = 0
        best_viewpoint = None
        result_dict = {}

        for sample in samples:
            T_cam_base = np.array(
                [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])

            rot_y = Rotation.from_rotvec(sample[0] * np.array([0, 1, 0]))
            rot_x = Rotation.from_rotvec(sample[1] * np.array([-1, 0, 0]))
            T_cam_base = pt.transform_from(rot_x.as_matrix(), np.zeros(
                3, )) @ pt.transform_from(rot_y.as_matrix(), np.zeros(
                    3, )) @ T_cam_base

            viewpoint = Viewpoint(camera=self.camera,
                                  T_base_map=T_base_map,
                                  T_cam_base=T_cam_base)

            visible_points = viewpoint.visible_points_viewing_angles(
                point_cloud=self.pcd_points_hom.copy(),
                min_max_viewing_angles=self.min_max_viewing_angles,
                clip_dist=5.0,
                rc_scene=self.rc_scene,
                run_PnP=False)

            if visible_points.shape[0] > max_seen_landmarks:
                max_seen_landmarks = visible_points.shape[0]
                best_viewpoint = viewpoint
                best_visible_points = visible_points

        if best_viewpoint is not None:
            print(f"Found viewpoint with score {max_seen_landmarks}")
            result_dict["T_cam_map"] = best_viewpoint.T_cam_map
            result_dict["T_cam_base"] = best_viewpoint.T_cam_base
            pointcloud_msg = self.point_cloud(points=best_visible_points[:,:-1], parent_frame="map")
            self.pc_publisher.publish(pointcloud_msg)
        else:
            print("WARNING NO VIEWPOINT FOUND")
            T_cam_base = np.array(
                [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])
            viewpoint = Viewpoint(camera=self.camera,
                                  T_base_map=T_base_map,
                                  T_cam_base=T_cam_base)
            result_dict["T_cam_map"] = viewpoint.T_cam_map
            result_dict["T_cam_base"] = viewpoint.T_cam_base

        return result_dict

    def plan_waypoint_max_crit(self, T_base_map: np.ndarray) -> dict:

        samples = yaw_pitch_samples(num=self.num_samples,
                                    max_yaw=2 * np.pi,
                                    min_pitch=-10 * DEG_TO_RAD,
                                    max_pitch=self.max_pitch * DEG_TO_RAD)

        max_seen_landmarks = 0
        best_viewpoint = None
        result_dict = {}

        for sample in samples:
            T_cam_base = np.array(
                [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])

            rot_y = Rotation.from_rotvec(sample[0] * np.array([0, 1, 0]))
            rot_x = Rotation.from_rotvec(sample[1] * np.array([-1, 0, 0]))
            T_cam_base = pt.transform_from(rot_x.as_matrix(), np.zeros(
                3, )) @ pt.transform_from(rot_y.as_matrix(), np.zeros(
                    3, )) @ T_cam_base
            

            viewpoint = Viewpoint(camera=self.camera,
                                  T_base_map=T_base_map,
                                  T_cam_base=T_cam_base)

            if self.occlusion:
                visible_points = viewpoint.visible_points_occlusion(
                    point_cloud=self.pcd_points_hom.copy(),
                    rc_scene=self.rc_scene,
                    clip_dist=5.0)
            else:
                visible_points = viewpoint.visible_points(
                    point_cloud=self.pcd_points_hom.copy(), clip_dist=5.0)
                
                # pointcloud_msg = self.point_cloud(points=visible_points[:,:-1], parent_frame="map")
                # self.pc_publisher.publish(pointcloud_msg)
            if visible_points.shape[0] > max_seen_landmarks:
                max_seen_landmarks = visible_points.shape[0]
                best_viewpoint = viewpoint
                best_visible_points = visible_points

        if best_viewpoint is not None:
            print(f"Found viewpoint with score {max_seen_landmarks}")
            result_dict["T_cam_map"] = best_viewpoint.T_cam_map
            result_dict["T_cam_base"] = best_viewpoint.T_cam_base
            pointcloud_msg = self.point_cloud(points=best_visible_points[:,:-1], parent_frame="map")
            self.pc_publisher.publish(pointcloud_msg)
        else:
            print("WARNING NO VIEWPOINT FOUND")
            T_cam_base = np.array(
                [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])
            viewpoint = Viewpoint(camera=self.camera,
                                  T_base_map=T_base_map,
                                  T_cam_base=T_cam_base)
            result_dict["T_cam_map"] = viewpoint.T_cam_map
            result_dict["T_cam_base"] = viewpoint.T_cam_base

        return result_dict

    def plan_waypoint_fisher_info(self, T_base_map: np.ndarray) -> dict:

        samples = yaw_pitch_samples(num=self.num_samples,
                                    max_yaw=2 * np.pi,
                                    min_pitch=-10 * DEG_TO_RAD,
                                    max_pitch=self.max_pitch * DEG_TO_RAD)

        max_fisher_info = 0
        best_viewpoint = None
        result_dict = {}

        for sample in samples:
            T_cam_base = np.array(
                [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])

            rot_y = Rotation.from_rotvec(sample[0] * np.array([0, 1, 0]))
            rot_x = Rotation.from_rotvec(sample[1] * np.array([-1, 0, 0]))
            T_cam_base = pt.transform_from(rot_x.as_matrix(), np.zeros(
                3, )) @ pt.transform_from(rot_y.as_matrix(), np.zeros(
                    3, )) @ T_cam_base

            viewpoint = Viewpoint(camera=self.camera,
                                  T_base_map=T_base_map,
                                  T_cam_base=T_cam_base)

            fisher_info = viewpoint.viewpoint_fisher_information(
                point_cloud=self.pcd_points_hom.copy(),
                rc_scene=self.rc_scene,
                clip_dist=5.0)

            if fisher_info > max_fisher_info:
                best_viewpoint = viewpoint
                self.prev_rotation = rot_y
                max_fisher_info = fisher_info
                best_visible_points = best_viewpoint.visible_points(point_cloud=self.pcd_points_hom.copy(), clip_dist=5.0)

        if best_viewpoint is not None:
            result_dict["T_cam_map"] = best_viewpoint.T_cam_map
            result_dict["T_cam_base"] = best_viewpoint.T_cam_base
            pointcloud_msg = self.point_cloud(points=best_visible_points[:,:-1], parent_frame="map")
            self.pc_publisher.publish(pointcloud_msg)
            
        else:
            print("WARNING NO VIEWPOINT FOUND")
            T_cam_base = np.array(
                [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])
            viewpoint = Viewpoint(camera=self.camera,
                                  T_base_map=T_base_map,
                                  T_cam_base=T_cam_base)
            result_dict["T_cam_map"] = viewpoint.T_cam_map
            result_dict["T_cam_base"] = viewpoint.T_cam_base

        return result_dict

    def plan_waypoint_random(self, T_base_map: np.ndarray) -> dict:

        samples = yaw_pitch_samples(num=self.num_samples,
                                    max_yaw=2 * np.pi,
                                    min_pitch=-10 * DEG_TO_RAD,
                                    max_pitch=self.max_pitch * DEG_TO_RAD)

        result_dict = {}

        sample = samples[random.randint(0, self.num_samples-1)]

        T_cam_base = np.array(
            [[0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])

        rot_y = Rotation.from_rotvec(sample[0] * np.array([0, 1, 0]))
        rot_x = Rotation.from_rotvec(sample[1] * np.array([-1, 0, 0]))
        T_cam_base = pt.transform_from(rot_x.as_matrix(), np.zeros(
            3, )) @ pt.transform_from(rot_y.as_matrix(), np.zeros(
                3, )) @ T_cam_base

        viewpoint = Viewpoint(camera=self.camera,
                              T_base_map=T_base_map,
                              T_cam_base=T_cam_base)

        result_dict["T_cam_map"] = viewpoint.T_cam_map
        result_dict["T_cam_base"] = viewpoint.T_cam_base

        return result_dict

    def plan_waypoint_mlp_clf(self, T_base_map: np.ndarray) -> dict:

        samples = yaw_pitch_samples(num=self.num_samples,
                                    max_yaw=2 * np.pi,
                                    min_pitch=-10 * DEG_TO_RAD,
                                    max_pitch=self.max_pitch * DEG_TO_RAD)

        best_viewpoint = None
        viewpoints = []
        histograms = []
        result_dict = {}

        for sample in samples:
            T_cam_base = np.array(
                [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])

            rot_y = Rotation.from_rotvec(sample[0] * np.array([0, 1, 0]))
            rot_x = Rotation.from_rotvec(sample[1] * np.array([-1, 0, 0]))
            T_cam_base = pt.transform_from(rot_x.as_matrix(), np.zeros(
                3, )) @ pt.transform_from(rot_y.as_matrix(), np.zeros(
                    3, )) @ T_cam_base

            viewpoint = Viewpoint(camera=self.camera,
                                  T_base_map=T_base_map,
                                  T_cam_base=T_cam_base)

            histogram = viewpoint.viewpoint_histogram(
                landmark_dict=self.landmark_dict,
                ids_point_cloud=self.id_points_hom.copy(),
                min_max_viewing_angles=self.min_max_viewing_angles,
                min_max_viewing_distances=self.min_max_viewing_distances,
                clip_dist=5.0,
                rc_scene=self.rc_scene)
            histograms.append(histogram)
            viewpoints.append(viewpoint)

        input_numpy = np.array(histograms)
        input_numpy = pre_process(input_numpy)
        inputs = torch.from_numpy(input_numpy).to(self.device).to(
            torch.float32)
        scores = self.classifier(inputs)
        scores_np = scores.cpu().detach().numpy()
        scores = torch.nn.functional.softmax(scores, dim=1)
        has_nan = torch.isnan(scores).any().item()
        if has_nan:
            print("nan warning")
        scores = scores[:, 1]
        scores = scores.cpu().detach().numpy()
        best_idx = np.argmax(scores)
        best_viewpoint = viewpoints[best_idx]


        if best_viewpoint is not None:
            
            print(f"Found viewpoint with score {scores[best_idx]}")
            result_dict["T_cam_map"] = best_viewpoint.T_cam_map
            result_dict["T_cam_base"] = best_viewpoint.T_cam_base
            best_visible_points = best_viewpoint.visible_points(point_cloud=self.pcd_points_hom.copy(), clip_dist=5.0)
            pointcloud_msg = self.point_cloud(points=best_visible_points[:,:-1], parent_frame="map")
            self.pc_publisher.publish(pointcloud_msg)
        else:
            print("WARNING NO VIEWPOINT FOUND")
            T_cam_base = np.array(
                [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])
            viewpoint = Viewpoint(camera=self.camera,
                                  T_base_map=T_base_map,
                                  T_cam_base=T_cam_base)
            result_dict["T_cam_map"] = viewpoint.T_cam_map
            result_dict["T_cam_base"] = viewpoint.T_cam_base

        return result_dict
    

    def plan_waypoint_mlp_reg(self, T_base_map: np.ndarray) -> dict:

        samples = yaw_pitch_samples(num=self.num_samples,
                                    max_yaw=2 * np.pi,
                                    min_pitch=-10 * DEG_TO_RAD,
                                    max_pitch=self.max_pitch * DEG_TO_RAD)

        best_viewpoint = None
        viewpoints = []
        histograms = []
        result_dict = {}

        for sample in samples:
            T_cam_base = np.array(
                [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])

            rot_y = Rotation.from_rotvec(sample[0] * np.array([0, 1, 0]))
            rot_x = Rotation.from_rotvec(sample[1] * np.array([-1, 0, 0]))
            T_cam_base = pt.transform_from(rot_x.as_matrix(), np.zeros(
                3, )) @ pt.transform_from(rot_y.as_matrix(), np.zeros(
                    3, )) @ T_cam_base

            viewpoint = Viewpoint(camera=self.camera,
                                  T_base_map=T_base_map,
                                  T_cam_base=T_cam_base)

            histogram = viewpoint.viewpoint_histogram(
                landmark_dict=self.landmark_dict,
                ids_point_cloud=self.id_points_hom.copy(),
                min_max_viewing_angles=self.min_max_viewing_angles,
                min_max_viewing_distances=self.min_max_viewing_distances,
                clip_dist=5.0,
                rc_scene=self.rc_scene)
            histograms.append(histogram)
            viewpoints.append(viewpoint)

        input_numpy = np.array(histograms)
        input_numpy = pre_process(input_numpy)
        inputs = torch.from_numpy(input_numpy).to(self.device).to(
            torch.float32)
        scores = self.regressor(inputs)
        scores = scores.cpu().detach().numpy()
        best_idx = np.argmin(scores)
        best_viewpoint = viewpoints[best_idx]

        if best_viewpoint is not None:
            result_dict["T_cam_map"] = best_viewpoint.T_cam_map
            result_dict["T_cam_base"] = best_viewpoint.T_cam_base
        else:
            print("WARNING NO VIEWPOINT FOUND")
            T_cam_base = np.array(
                [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])
            viewpoint = Viewpoint(camera=self.camera,
                                  T_base_map=T_base_map,
                                  T_cam_base=T_cam_base)
            result_dict["T_cam_map"] = viewpoint.T_cam_map
            result_dict["T_cam_base"] = viewpoint.T_cam_base

        return result_dict
    
    def plan_waypoint_trf_clf(self, T_base_map: np.ndarray) -> dict:

        samples = yaw_pitch_samples(num=self.num_samples,
                                    max_yaw=2 * np.pi,
                                    min_pitch=-10 * DEG_TO_RAD,
                                    max_pitch=self.max_pitch * DEG_TO_RAD)

        best_viewpoint = None
        max_score = 0
        result_dict = {}

        for sample in samples:
            T_cam_base = np.array(
                [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])

            rot_y = Rotation.from_rotvec(sample[0] * np.array([0, 1, 0]))
            rot_x = Rotation.from_rotvec(sample[1] * np.array([-1, 0, 0]))
            T_cam_base = pt.transform_from(rot_x.as_matrix(), np.zeros(
                3, )) @ pt.transform_from(rot_y.as_matrix(), np.zeros(
                    3, )) @ T_cam_base

            viewpoint = Viewpoint(camera=self.camera,
                                      T_base_map=T_base_map,
                                      T_cam_base=T_cam_base,
                                      create_colmap_camera=True)

            token = viewpoint.tokenize_viewpoint(
                landmark_dict=self.landmark_dict,
                dino_dict=self.dino_dict,
                ids_point_cloud=self.id_points_hom.copy(),
                min_max_viewing_angles=self.min_max_viewing_angles,
                min_max_viewing_distances=self.min_max_viewing_distances,
                clip_dist=5.0,
                rc_scene=self.rc_scene,
                dino_dim=3)
            
            token = token[:,:73]
            token = torch.from_numpy(token.astype(np.float32)).to(
                self.device).unsqueeze(0)
            score = self.transformer(token)
            score = torch.nn.functional.softmax(score, dim=1)
            score = score[0][1]

            if score > max_score:
                best_viewpoint = viewpoint
                best_visible_points = viewpoint.visible_points(point_cloud=self.pcd_points_hom.copy(), clip_dist=5.0)
                max_score = float(score)

        if best_viewpoint is not None:
            print(f"Found viewpoint with score {score}")
            result_dict["T_cam_map"] = best_viewpoint.T_cam_map
            result_dict["T_cam_base"] = best_viewpoint.T_cam_base
            pointcloud_msg = self.point_cloud(points=best_visible_points[:,:-1], parent_frame="map")
            self.pc_publisher.publish(pointcloud_msg)
        else:
            print("WARNING NO VIEWPOINT FOUND")
            T_cam_base = np.array(
                [[0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])
            viewpoint = Viewpoint(camera=self.camera,
                                  T_base_map=T_base_map,
                                  T_cam_base=T_cam_base)
            result_dict["T_cam_map"] = viewpoint.T_cam_map
            result_dict["T_cam_base"] = viewpoint.T_cam_base

        return result_dict
    

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
