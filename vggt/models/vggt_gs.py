from vggt.models.vggt import VGGT
import torch
import torch.nn as nn
import open3d as o3d
import logging
import torch.distributed as dist

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from training.gs_feature_parser import Parser_GS
from vggt.utils.geometry import closed_form_inverse_se3
import math


from gsplat.rendering import rasterization
from gsplat import export_splats



class VGGT_GS(VGGT):
    def __init__(self,
                #  img_size=518, 
                #  patch_size=14,
                sh_degree=1, 
                embed_dim=1024,
                enable_camera=True,
                enable_depth=True,
                enable_point=True,
                enable_track=True,
                debug=False,
                use_distributed_render=False
                ):  # Ensure embed_dim is passed
        """
        Inherit from VGGT and add GS head.

        Args:
            gs_pos_predict (str): The method to predict the GS (Ground Station) position.
                - "new_xyz": Directly predict the position of GS without using VGGT prior.
                - "from_vggt": Use the position predicted by VGGT as a prior and refine it.
            embed_dim (int): The embedding dimension, passed to VGGT constructor
        """
        # might be useful to check https://github.com/OpenRobotLab/gs-lrm-unofficial/blob/6fe1104d5fe7176b866b877f3ff798b40849d0d0/src/model/encoder/encoder_lrm.py#L99
        # Initialize the parent VGGT class with embed_dim argument
        super().__init__(enable_camera=enable_camera, 
                         enable_depth=enable_depth,
                         enable_point=enable_point,
                         enable_track=enable_track,
                         )  # Pass embed_dim to the parent class
        self.debug = debug
        self.use_distributed_render = use_distributed_render
        self.sh_degree = sh_degree
        self.gs_head = DPTHead(
            dim_in=2*embed_dim,
            output_dim=(sh_degree + 1)**2*3 + 3 + 4 + 1,
            activation="inv_log",
            conf_activation="expp1",
        )
        self.spls = torch.nn.Softplus()
    
    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, step: int = 0):
        """
        Forward pass of the VGGT model finetuned for GS prediction.
        
        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images) # [(B x S x P x 2C) x L], 1 + num_register_tokens
        # 上面2C是因为一个存frame(local)的embed，一个存global的embed

        predictions = {}

        with torch.amp.autocast("cuda", enabled=False):
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
            if self.gs_head is not None:
                if self.gs_head.feature_only == True:
                    raise ValueError("GS head should not be feature_only, it should output predictions directly.")
                gs_features, gs_conf = self.gs_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["gs_features"] = gs_features
                predictions["gs_conf"] = gs_conf

            # We also need camera head for GS rendering
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration 
            outputs = self._render_gs(predictions, images, step)
            predictions["renders"] = outputs
            predictions["original_images"] = images
            return predictions
    def _dist_shard_gaussians(self, pts, quats, scales, colors, opacities):
        # 约定形状：pts=(B, N, 3); quats=(B, N, 4); scales=(B, N, 3);
        # colors=(B, N, K, 3) 或 (B, N, 3)（debug 情况）；opacities=(B, N)
        assert pts.dim() == 3 and pts.size(1) > 0, "Expect pts shape [B, N, 3]"
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        N_total = pts.size(1)
        # 等分 contiguous 切片（简单稳定且无额外开销）
        start = (N_total * rank) // world_size
        end   = (N_total * (rank + 1)) // world_size
        sl = slice(start, end)

        pts      = pts[:, sl, :]
        quats    = quats[:, sl, :]
        scales   = scales[:, sl, :]
        opacities= opacities[:, sl]            # 注意你的代码里是 squeeze(-1) 过后的 (B, N)
        # colors 既可能是 (B, N, 3) 也可能是 (B, N, K, 3)
        if colors.dim() == 3:
            colors = colors[:, sl, :]
        elif colors.dim() == 4:
            colors = colors[:, sl, :, :]
        else:
            raise ValueError("Unexpected colors shape")

        return pts, quats, scales, colors, opacities

    def _render_gs(self, predictions: dict, images: torch.Tensor, step: int):
        """
        Render 3DGS image based on model predictions.
        Args:
            cfg (dict): Configuration dictionary
        """
        
        # parse gs features
        gs_features = predictions["gs_features"] # B,S,H,W,-1
        gs_conf = torch.sigmoid(predictions["gs_conf"]).unsqueeze(-1) # B,S,H,W,1
        points = predictions["world_points"] # B,S,H,W,3
        B,S,H,W,_ = gs_features.shape
        original_colors = images.permute(0,1,3,4,2).reshape(B,-1,3)
        # # soft filter gs features
        # conf_threshold = torch.mean(gs_conf, dim=[2,3]).view(B,S,1,1,1) # B,S,1,1,1
        # sharp_factor = 10.0
        # soft_mask = torch.sigmoid(sharp_factor*(gs_conf - conf_threshold)) * torch.sigmoid(sharp_factor*(gs_conf - 0.1))
        # gs_features = gs_features * soft_mask
        
        
        
        # feature extraction
        scale_factor = 2e-3
        Kx3 = (self.sh_degree + 1)**2 * 3
        sh_coeffs = gs_features[..., :Kx3].reshape(B,S,H,W,Kx3//3,3)
        scales = torch.clamp(self.spls(gs_features[..., Kx3:Kx3+3])*scale_factor,max=0.5)
        quats = gs_features[..., Kx3+3:Kx3+7]
        opacities = torch.sigmoid(gs_features[..., -1:])
        
        # put predictions of each frame together for one scene
        global_points = points.reshape(B,S*H*W,3)
        global_quats = quats.reshape(B,S*H*W,4)
        global_scales = scales.reshape(B,S*H*W,3)
        global_colors = original_colors if self.debug else sh_coeffs.reshape(B, S*H*W, Kx3//3, 3)
        global_opacities = opacities.reshape(B,S*H*W,1).squeeze(-1)
        sh_degree = self.sh_degree if not self.debug else None
        # Camera parameters
        pose_enc = predictions["pose_enc"]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw=images.shape[-2:]) # extrinsics: BxSx3x4, intrinsics: BxSx3x3   
        viewmats = torch.zeros((B, S, 4, 4), device="cuda")
        viewmats[:, :, :3, :4] = extrinsics
        viewmats[:, :, 3, 3] = 1
        
        outputs = []
        
        # if self.debug:
        #     save_ply(
        #         global_points[0], 
        #         global_colors[0], 
        #         "debug.ply"
        #     )
        # batch rasterization
        renders, alphas, meta = rasterization(
            means=global_points,
            quats=global_quats,
            scales=global_scales,
            sh_degree=sh_degree,
            colors=global_colors,
            opacities=global_opacities,
            viewmats=viewmats,
            Ks=intrinsics,
            height=H,
            width=W,
            packed=True,
            distributed=self.use_distributed_render,
        )
        outputs = renders.permute(0,1,4,2,3).contiguous()
        
        # for some training steps, save 3dgs checkpoints
        if step % 10000 == 0 and step != 0:
            with torch.no_grad():
                for b in range(B):
                    save_path = f"saving/scene_step_{step}_batch_{b}"
                    # export_splats(
                    #     means=global_points[b],
                    #     quats=global_quats[b],
                    #     scales=global_scales[b],
                    #     opacities=global_opacities[b],
                    #     sh0=global_colors[b,:,:1,:],
                    #     shN=global_colors[b,:,1:,:],
                    #     save_to=save_path,
                    #     format="ply_compressed"
                    # )
                    # logging.info(f"saved at {save_path}")
                    load_barrier = 200000
                    data = {
                            "step": step, 
                            "splats": 
                                {
                                   "means": torch.nn.Parameter(global_points[b][:load_barrier]),
                                   "quats": torch.nn.Parameter(global_quats[b][:load_barrier]),
                                   "scales": torch.nn.Parameter(global_scales[b][:load_barrier]),
                                   "opacities": torch.nn.Parameter(global_opacities[b][:load_barrier]),
                                   "sh0": torch.nn.Parameter(global_colors[b, :, :1, :][:load_barrier]),
                                   "shN": torch.nn.Parameter(global_colors[b, :, 1:, :][:load_barrier]), 
                                }
                            }
                    torch.save(data, f"{save_path}.pt")
                    logging.info(f"scene data saved at {save_path}")
                

        return outputs
    


def save_ply(points, colors, filename):
    import open3d as o3d   
    import numpy as np             
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).detach().cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    if torch.is_tensor(colors):
        points_visual_rgb = colors.reshape(-1, 3).detach().cpu().numpy()
    else:
        points_visual_rgb = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)