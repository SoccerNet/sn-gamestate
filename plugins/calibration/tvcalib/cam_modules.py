from typing import Tuple, Dict, Union

from pytorch_lightning import LightningModule
import torch
import kornia

from tvcalib.utils.data_distr import FeatureScalerZScore


class CameraParameterWLensDistDictZScore(LightningModule):
    """Holds individual camera parameters including lens distortion parameters as nn.Modul"""

    def __init__(self, cam_distr, dist_distr, device="cpu"):
        super(CameraParameterWLensDistDictZScore, self).__init__()

        self.cam_distr = cam_distr
        self._device = device

        # phi raw
        self.param_dict = torch.nn.ParameterDict(
            {
                k: torch.nn.parameter.Parameter(
                    torch.zeros(
                        *cam_distr[k]["dimension"],
                        device=device,
                    ),
                    requires_grad=False
                    if ("no_grad" in cam_distr[k]) and (cam_distr[k]["no_grad"] == True)
                    else True,
                )
                for k in cam_distr.keys()
            }
        )

        # denormalization module to get phi_target
        self.feature_scaler = torch.nn.ModuleDict(
            {k: FeatureScalerZScore(*cam_distr[k]["mean_std"]) for k in cam_distr.keys()}
        )

        self.dist_distr = dist_distr
        if self.dist_distr is not None:
            self.param_dict_dist = torch.nn.ParameterDict(
                {
                    k: torch.nn.Parameter(torch.zeros(*dist_distr[k]["dimension"], device=device))
                    for k in dist_distr.keys()
                }
            )
            # TODO: modify later to dynamically cunstruct a tensor of shape (k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])
            #

            self.feature_scaler_dist_coeff = torch.nn.ModuleDict(
                {k: FeatureScalerZScore(*dist_distr[k]["mean_std"]) for k in dist_distr.keys()}
            )

    def initialize(
        self,
        update_dict_cam: Union[Dict[str, Union[float, torch.tensor]], None],
        update_dict_dist=None,
    ):
        """Initializes all camera parameters with zeros and replace specific values with provided values

        Args:
            update_dict_cam (Dict[str, Union[float, torch.tensor]]): Parameters to be updated
        """

        for k in self.param_dict.keys():
            self.param_dict[k].data = torch.zeros(
                *self.cam_distr[k]["dimension"], device=self._device
            )
        if self.dist_distr is not None:
            for k in self.dist_distr.keys():
                self.param_dict_dist[k].data = torch.zeros(
                    *self.dist_distr[k]["dimension"], device=self._device
                )

        if update_dict_cam is not None and len(update_dict_cam) > 0:
            for k, v in update_dict_cam.items():
                self.param_dict[k].data = (
                    torch.zeros(*self.cam_distr[k]["dimension"], device=self._device) + v
                )
        if update_dict_dist is not None:
            raise NotImplementedError

    def forward(self):
        phi_dict = {}
        for k, param in self.param_dict.items():
            phi_dict[k] = self.feature_scaler[k](param)

        if self.dist_distr is None:
            return phi_dict, None

        # This is a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)` depending on the provided dict of coefficients
        # assumes dict is ordered according (k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])
        psi = torch.stack(
            [
                torch.clamp(
                    self.feature_scaler_dist_coeff[k](param),
                    min=self.dist_distr[k]["minmax"][0],
                    max=self.dist_distr[k]["minmax"][1],
                )
                for k, param in self.param_dict_dist.items()
            ],
            dim=-1,  # stack individual features and not arbirary leading dimensions
        )

        return phi_dict, psi


class SNProjectiveCamera:
    def __init__(
        self,
        phi_dict: Dict[str, torch.tensor],
        psi: torch.tensor,
        principal_point: Tuple[float, float],
        image_width: int,
        image_height: int,
        device: str = "cpu",
        nan_check=True,
    ) -> None:
        """Projective camera defined as K @ R [I|-t] with lens distortion module and batch dimensions B,T.

        Following Euler angles convention, we use a ZXZ succession of intrinsic rotations in order to describe
        the orientation of the camera. Starting from the world reference axis system, we first apply a rotation
        around the Z axis to pan the camera. Then the obtained axis system is rotated around its x axis in order to tilt the camera.
        Then the last rotation around the z axis of the new axis system alows to roll the camera. Note that this z axis is the principal axis of the camera.

        As T is not provided for camra location and lens distortion, these parameters are assumed to be fixed accross T.
        phi_dict is a dict of parameters containing:
        {
            'aov_x, torch.Size([B, T])',
            'pan, torch.Size([B, T])',
            'tilt, torch.Size([B, T])',
            'roll, torch.Size([B, T])',
            'c_x, torch.Size([B, 1])',
            'c_y, torch.Size([B, 1])',
            'c_z, torch.Size([B, 1])',
        }

        Internally fuses B and T dimension to pseudo batch dimension.
        {
            'aov_x, torch.Size([B*T])',
            'pan, torch.Size([B*T])',
            'tilt, torch.Size([B*T])'
            'roll, torch.Size([B*T])',
            'c_x, torch.Size([B])',
            'c_y, torch.Size([B])',
            'c_z, torch.Size([B])',
            }

        aov_x, pan, tilt, roll are assumed in radian.

        Note on lens distortion:
            Lens distortion coefficients are independent from image resolution!
            We I(dist_points(K_ndc, dist_coeff, points2d_ndc)) == I(dist_points(K_raster, dist_coeff, points2d_raster))

        Args:
            phi_dict (Dict[str, torch.tensor]): See example above
            psi (Union[None, torch.Tensor]): distortion coefficients as concatinated vector according to https://kornia.readthedocs.io/en/latest/geometry.calibration.html of shape (B, T, {2, 4, 5,8,12, 14})
            principal_point (Tuple[float, float]): Principal point assumed to be fixed across all samples (B,T,)
            image_width (int): assumed to be fixed across all samples (B,T,)
            image_height (int): assumed to be fixed across all samples (B,T,)
        """

        # fuse B and T dimension
        phi_dict_flat = {}
        for k, v in phi_dict.items():
            if len(v.shape) == 2:
                phi_dict_flat[k] = v.view(v.shape[0] * v.shape[1])
            elif len(v.shape) == 3:
                phi_dict_flat[k] = v.view(v.shape[0] * v.shape[1], v.shape[-1])

        self.batch_dim, self.temporal_dim = phi_dict["pan"].shape
        self.pseudo_batch_size = phi_dict_flat["pan"].shape[0]
        self.phi_dict_flat = phi_dict_flat

        self.principal_point = principal_point
        self.image_width = image_width
        self.image_height = image_height
        self.device = device

        self.psi = psi
        if self.psi is not None:
            if self.psi.shape[-1] != 2:
                raise NotImplementedError

            # :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`.
            # psi is a vector with 2, 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`.
            if self.psi.shape[-1] == 2:
                # assume zero tangential coefficients
                psi_ext = torch.zeros(*list(self.psi.shape[:-1]), 4)
                psi_ext[..., :2] = self.psi
                self.psi = psi_ext
            self.lens_dist_coeff = self.psi.view(self.pseudo_batch_size, self.psi.shape[-1]).to(
                self.device
            )

        self.intrinsics_ndc = self.construct_intrinsics_ndc()
        self.intrinsics_raster = self.construct_intrinsics_raster()

        self.rotation = self.rotation_from_euler_angles(
            *[phi_dict_flat[k] for k in ["pan", "tilt", "roll"]]
        )
        self.position = torch.stack([phi_dict_flat[k] for k in ["c_x", "c_y", "c_z"]], dim=-1)
        self.position = self.position.repeat_interleave(
            int(self.pseudo_batch_size / self.batch_dim), dim=0
        )  # (B, 3) # TODO: probably needs modification if B > 0?
        self.P_ndc = self.construct_projection_matrix(self.intrinsics_ndc)
        self.P_raster = self.construct_projection_matrix(self.intrinsics_raster)
        self.phi_dict = phi_dict

        self.nan_check = nan_check
        super().__init__()

    def construct_projection_matrix(self, intrinsics):
        It = torch.eye(4, device=self.device)[:-1].repeat(self.pseudo_batch_size, 1, 1)
        It[:, :, -1] = -self.position  # (B, 3, 4)
        self.It = It
        return intrinsics @ self.rotation @ It  #  # (B, 3, 4)

    def construct_intrinsics_ndc(self):
        # assume that the principal point is (0,0)
        K = torch.eye(3, requires_grad=False, device=self.device)
        K = K.reshape((1, 3, 3)).repeat(self.pseudo_batch_size, 1, 1)
        K[:, 0, 0] = self.get_fl_from_aov_rad(self.phi_dict_flat["aov"], d=2)
        K[:, 1, 1] = self.get_fl_from_aov_rad(
            self.phi_dict_flat["aov"], d=2 * self.image_width / self.image_height
        )
        return K

    def construct_intrinsics_raster(self):
        # assume that the principal point is (W/2,H/2)
        K = torch.eye(3, requires_grad=False, device=self.device)
        K = K.reshape((1, 3, 3)).repeat(self.pseudo_batch_size, 1, 1)
        K[:, 0, 0] = self.get_fl_from_aov_rad(self.phi_dict_flat["aov"], d=self.image_width)
        K[:, 1, 1] = self.get_fl_from_aov_rad(self.phi_dict_flat["aov"], d=self.image_width)
        K[:, 0, 2] = self.principal_point[0]
        K[:, 1, 2] = self.principal_point[1]
        return K

    def __str__(self) -> str:
        return f"aov_deg={torch.rad2deg(self.phi_dict['aov'])}, t={torch.stack([self.phi_dict[k] for k in ['c_x', 'c_y', 'c_z']], dim=-1)}, pan_deg={torch.rad2deg(self.phi_dict['pan'])} tilt_deg={torch.rad2deg(self.phi_dict['tilt'])} roll_deg={torch.rad2deg(self.phi_dict['roll'])}"

    def str_pan_tilt_roll_fl(self, b, t):
        r = f"FOV={torch.rad2deg(self.phi_dict['aov'][b, t]):.1f}°, pan={torch.rad2deg(self.phi_dict['pan'][b, t]):.1f}° tilt={torch.rad2deg(self.phi_dict['tilt'][b, t]):.1f}° roll={torch.rad2deg(self.phi_dict['roll'][b, t]):.1f}°"
        return r

    def str_lens_distortion_coeff(self, b):
        # TODO: T! also need indivudual lens_dist_coeff for each t in T
        # print(self.lens_dist_coeff.shape)
        return f"lens dist coeff=" + " ".join(
            [f"{x:.2f}" for x in self.lens_dist_coeff[b, :2]]
        )  # print only radial lens dist. coeff

    def __repr__(self) -> str:
        return f"{self.__class__}:" + self.__str__()

    def __len__(self):
        return self.pseudo_batch_size  # e.g. self.intrinsics.shape[0]

    def project_point2pixel(self, points3d: torch.tensor, lens_distortion: bool) -> torch.tensor:
        """Project world coordinates to pixel coordinates.

        Args:
            points3d (torch.tensor): of shape (N, 3) or (1, N, 3)

        Returns:
            torch.tensor: projected points of shape (B, T, N, 2)
        """
        position = self.position.view(self.pseudo_batch_size, 1, 3)
        point = points3d - position
        rotated_point = self.rotation @ point.transpose(1, 2)  # (pseudo_batch_size, 3, N)
        dist_point2cam = rotated_point[:, 2]  # (B, N) distance pixel to world point
        dist_point2cam = dist_point2cam.view(self.pseudo_batch_size, 1, rotated_point.shape[-1])
        rotated_point = rotated_point / dist_point2cam  # (B, 3, N) / (B, 1, N) -> (B, 3, N)

        projected_points = self.intrinsics_raster @ rotated_point  # (B, 3, N)
        # transpose vs view? here
        projected_points = projected_points.transpose(-1, -2)  # cannot use view()
        projected_points = kornia.geometry.convert_points_from_homogeneous(projected_points)
        if lens_distortion:
            if self.psi is None:
                raise RuntimeError("Lens distortion requested, but deactivated in module")
            projected_points = self.distort_points(projected_points, self.intrinsics_raster)

        # reshape back from (pseudo_batch_size, N, 2) to (B, T, N, 2)
        projected_points = projected_points.view(
            self.batch_dim, self.temporal_dim, projected_points.shape[-2], 2
        )
        if self.nan_check:
            if torch.isnan(projected_points).any().item():
                print(self.phi_dict_flat)
                print(projected_points)
                raise RuntimeWarning("NaN in project_point2pixel")
        return projected_points

    def project_point2ndc(self, points3d: torch.tensor, lens_distortion: bool) -> torch.tensor:
        """Project world coordinates to pixel coordinates.

        Args:
            points3d (torch.tensor): of shape (N, 3) or (1, N, 3)

        Returns:
            torch.tensor: projected points of shape (B, T, N, 2)
        """
        position = self.position.view(self.pseudo_batch_size, 1, 3)
        point = points3d - position
        rotated_point = self.rotation @ point.transpose(1, 2)  # (pseudo_batch_size, 3, N)
        dist_point2cam = rotated_point[:, 2]  # (B, N) distance pixel to world point
        dist_point2cam = dist_point2cam.view(self.pseudo_batch_size, 1, rotated_point.shape[-1])
        rotated_point = rotated_point / dist_point2cam  # (B, 3, N) / (B, 1, N) -> (B, 3, N)

        projected_points = self.intrinsics_ndc @ rotated_point  # (B, 3, N)
        # transpose vs view? here
        projected_points = projected_points.transpose(-1, -2)  # cannot use view()
        projected_points = kornia.geometry.convert_points_from_homogeneous(projected_points)
        if self.nan_check:
            if torch.isnan(projected_points).any().item():
                print(projected_points)
                print(self.phi_dict_flat)
                print("lens distortion", self.lens_dist_coeff)

                raise RuntimeWarning("NaN in project_point2ndc before distort")
        if lens_distortion:
            if self.psi is None:
                raise RuntimeError("Lens distortion requested, but deactivated in module")
            projected_points = self.distort_points(projected_points, self.intrinsics_ndc)

        # reshape back from (pseudo_batch_size, N, 2) to (B, T, N, 2)
        projected_points = projected_points.view(
            self.batch_dim, self.temporal_dim, projected_points.shape[-2], 2
        )
        if self.nan_check:
            if torch.isnan(projected_points).any().item():
                print(self.phi_dict_flat)
                print(projected_points)
                raise RuntimeWarning("NaN in project_point2ndc after distort")
        return projected_points

    def project_point2pixel_from_P(
        self, points3d: torch.tensor, lens_distortion: bool
    ) -> torch.tensor:
        """Project world coordinates to pixel coordinates from the projection matrix.

        Args:
            points3d (torch.tensor): of shape (1, N, 3)

        Returns:
            torch.tensor: projected points of shape (B, T, N, 2)
        """

        points3d = kornia.geometry.conversions.convert_points_to_homogeneous(points3d).transpose(
            1, 2
        )  # (B, 4, N)
        projected_points = torch.bmm(self.P_raster, points3d.repeat(self.pseudo_batch_size, 1, 1))
        normalize_by = projected_points[:, -1].view(
            self.pseudo_batch_size, 1, projected_points.shape[-1]
        )
        projected_points /= normalize_by
        projected_points = projected_points.transpose(-1, -2)  # cannot use view()
        projected_points = kornia.geometry.convert_points_from_homogeneous(projected_points)
        if lens_distortion:
            if self.psi is None:
                raise RuntimeError("Lens distortion requested, but deactivated in module")
            projected_points = self.distort_points(projected_points, self.intrinsics_raster)
        # reshape back from (pseudo_batch_size, N, 2) to (B, T, N, 2)
        projected_points = projected_points.view(
            self.batch_dim, self.temporal_dim, projected_points.shape[-2], 2
        )
        return projected_points  # (B, T,  N, 2)

    def project_point2ndc_from_P(
        self, points3d: torch.tensor, lens_distortion: bool
    ) -> torch.tensor:
        """Project world coordinates to pixel coordinates from the projection matrix.

        Args:
            points3d (torch.tensor): of shape (1, N, 3)

        Returns:
            torch.tensor: projected points of shape (B, T, N, 2)
        """

        points3d = kornia.geometry.conversions.convert_points_to_homogeneous(points3d).transpose(
            1, 2
        )  # (B, 4, N)
        projected_points = torch.bmm(self.P_ndc, points3d.repeat(self.pseudo_batch_size, 1, 1))
        normalize_by = projected_points[:, -1].view(
            self.pseudo_batch_size, 1, projected_points.shape[-1]
        )
        projected_points /= normalize_by
        projected_points = projected_points.transpose(-1, -2)  # cannot use view()
        projected_points = kornia.geometry.convert_points_from_homogeneous(projected_points)
        if lens_distortion:
            if self.psi is None:
                raise RuntimeError("Lens distortion requested, but deactivated in module")
            projected_points = self.distort_points(projected_points, self.intrinsics_ndc)
        # reshape back from (pseudo_batch_size, N, 2) to (B, T, N, 2)
        projected_points = projected_points.view(
            self.batch_dim, self.temporal_dim, projected_points.shape[-2], 2
        )
        return projected_points  # (B, T,  N, 2)

    def rotation_from_euler_angles(self, pan, tilt, roll):
        # rotation matrices from a batch of pan tilt roll [rad] vectors of shape (?, )

        mask = (
            torch.eye(3, requires_grad=False, device=self.device)
            .reshape((1, 3, 3))
            .repeat(pan.shape[0], 1, 1)
        )
        mask[:, 0, 0] = -torch.sin(pan) * torch.sin(roll) * torch.cos(tilt) + torch.cos(
            pan
        ) * torch.cos(roll)
        mask[:, 0, 1] = torch.sin(pan) * torch.cos(roll) + torch.sin(roll) * torch.cos(
            pan
        ) * torch.cos(tilt)
        mask[:, 0, 2] = torch.sin(roll) * torch.sin(tilt)

        mask[:, 1, 0] = -torch.sin(pan) * torch.cos(roll) * torch.cos(tilt) - torch.sin(
            roll
        ) * torch.cos(pan)
        mask[:, 1, 1] = -torch.sin(pan) * torch.sin(roll) + torch.cos(pan) * torch.cos(
            roll
        ) * torch.cos(tilt)
        mask[:, 1, 2] = torch.sin(tilt) * torch.cos(roll)

        mask[:, 2, 0] = torch.sin(pan) * torch.sin(tilt)
        mask[:, 2, 1] = -torch.sin(tilt) * torch.cos(pan)
        mask[:, 2, 2] = torch.cos(tilt)

        return mask

    def get_homography_raster(self):
        return self.P_raster[:, :, [0, 1, 3]].inverse()

    def get_rays_world(self, x):
        """_summary_

        Args:
            x (_type_): x of shape (B, 3, N)

        Returns:
            LineCollection: _description_
        """
        raise NotImplementedError
        # TODO: verify
        # ray_cam_trans = torch.bmm(self.rotation.inverse(), torch.bmm(self.intrinsics.inverse(), x))
        # # unnormalized direction vector in euclidean points (x,y,z) based on camera origin (0,0,0)
        # ray_cam_trans = torch.nn.functional.normalize(ray_cam_trans, p=2, dim=1)  # (B, 3, N)

        # # shift support vector to origin in world space, i.e. the translation vector
        # support = self.position.unsqueeze(-1).repeat(
        #     ray_cam_trans.shape[0], 1, ray_cam_trans.shape[2]
        # )  # (B, 3, N)
        # return LineCollection(support=support, direction_norm=ray_cam_trans)

    @staticmethod
    def get_aov_rad(d: float, fl: torch.tensor):
        # https://en.wikipedia.org/wiki/Angle_of_view#Calculating_a_camera's_angle_of_view
        return 2 * torch.arctan(d / (2 * fl))  # in range [0.0, PI]

    @staticmethod
    def get_fl_from_aov_rad(aov_rad: torch.tensor, d: float):
        return 0.5 * d * (1 / torch.tan(0.5 * aov_rad))

    def undistort_points(self, points_pixel: torch.tensor, intrinsics, num_iters=5) -> torch.tensor:
        """Compensate for lens distortion a set of 2D image points.

        Wrapper for kornia.geometry.undistort_points()

        Args:
            points_pixel (torch.tensor): tensor of shape (B, N, 2)

        Returns:
            torch.tensor: undistorted points of shape (B, N, 2)
        """
        # print(points_pixel.shape, intrinsics.shape, self.lens_dist_coeff.shape)
        batch_dim, temporal_dim, N, _ = points_pixel.shape
        points_pixel = points_pixel.view(batch_dim * temporal_dim, N, 2)
        true_batch_size = batch_dim

        lens_dist_coeff = self.lens_dist_coeff
        if true_batch_size < self.batch_dim:
            intrinsics = intrinsics[:true_batch_size]
            lens_dist_coeff = lens_dist_coeff[:true_batch_size]

        return kornia.geometry.undistort_points(
            points_pixel, intrinsics, dist=lens_dist_coeff, num_iters=num_iters
        ).view(batch_dim, temporal_dim, N, 2)

    def distort_points(self, points_pixel: torch.tensor, intrinsics) -> torch.tensor:
        """Distortion of a set of 2D points based on the lens distortion model.

        Wrapper for kornia.geometry.distort_points()

        Args:
            points_pixel (torch.tensor): tensor of shape (B, N, 2)

        Returns:
            torch.tensor: distorted points of shape (B, N, 2)
        """
        return kornia.geometry.distort_points(points_pixel, intrinsics, dist=self.lens_dist_coeff)

    def undistort_images(self, images):
        # images of shape (B, T, C, H, W)
        true_batch_size, T = images.shape[:2]
        images = images.view(true_batch_size * T, 3, self.image_height, self.image_width).to(
            self.device
        )
        intrinsics = self.intrinsics_raster
        lens_dist_coeff = self.lens_dist_coeff
        if true_batch_size < self.batch_dim:
            intrinsics = intrinsics[:true_batch_size]
            lens_dist_coeff = lens_dist_coeff[:true_batch_size]

        return kornia.geometry.calibration.undistort_image(
            images, intrinsics, lens_dist_coeff
        ).view(true_batch_size, self.temporal_dim, 3, self.image_height, self.image_width)

    def get_parameters(self, true_batch_size=None):
        """
        Get dict of relevant camera parameters and homography matrix
        :return: The dictionary
        """
        out_dict = {
            "pan_degrees": torch.rad2deg(self.phi_dict["pan"]),
            "tilt_degrees": torch.rad2deg(self.phi_dict["tilt"]),
            "roll_degrees": torch.rad2deg(self.phi_dict["roll"]),
            "position_meters": torch.stack([self.phi_dict[k] for k in ["c_x", "c_y", "c_z"]], dim=1)
            .squeeze(-1)
            .unsqueeze(-2)
            .repeat(1, self.temporal_dim, 1),
            "aov_radian": self.phi_dict["aov"],
            "aov_degrees": torch.rad2deg(self.phi_dict["aov"]),
            "x_focal_length": self.get_fl_from_aov_rad(self.phi_dict["aov"], d=self.image_width),
            "y_focal_length": self.get_fl_from_aov_rad(self.phi_dict["aov"], d=self.image_width),
            "principal_point": torch.tensor(
                [[self.principal_point] * self.temporal_dim] * self.batch_dim
            ),
        }
        out_dict["homography"] = self.get_homography_raster().unsqueeze(1) # (B, 1, 3, 3)

        # expected for SN evaluation
        out_dict["radial_distortion"] = torch.zeros(self.batch_dim, self.temporal_dim, 6)
        out_dict["tangential_distortion"] = torch.zeros(self.batch_dim, self.temporal_dim, 2)
        out_dict["thin_prism_distortion"] = torch.zeros(self.batch_dim, self.temporal_dim, 4)

        if self.psi is not None:
            # in case only k1 and k2 are provided
            out_dict["radial_distortion"][..., :2] = self.psi[..., :2]

        if true_batch_size is None or true_batch_size == self.batch_dim:
            return out_dict

        for k in out_dict.keys():
            out_dict[k] = out_dict[k][:true_batch_size]

        return out_dict

    @staticmethod
    def static_undistort_points(points, cam):

        intrinsics = cam.intrinsics_raster
        lens_dist_coeff = cam.lens_dist_coeff

        true_batch_size = points.shape[0]
        if true_batch_size < cam.batch_dim:
            intrinsics = intrinsics[:true_batch_size]
            lens_dist_coeff = lens_dist_coeff[:true_batch_size]
        # points in homogenous coordinates
        # (B, T, 3, S, N) -> (T, 3, S*N) -> (T, S*N, 3)
        batch_size, T, _, S, N = points.shape
        points = points.view(batch_size, T, 3, S * N).transpose(2, 3)
        points[..., :2] = kornia.geometry.undistort_points(
            points[..., :2].view(batch_size * T, S * N, 2),
            intrinsics,
            dist=lens_dist_coeff,
            num_iters=1,
        ).view(batch_size, T, S * N, 2)

        # (T, S*N, 3) -> (T, 3, S*N) -> (B, T, 3, S, N)
        points = points.transpose(2, 3).view(batch_size, T, 3, S, N)
        return points
