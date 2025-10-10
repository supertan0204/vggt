from vggt.models.vggt import VGGT
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch_scatter import scatter_add, scatter_max
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import os


from gsplat.rendering import rasterization
from gsplat import export_splats



class VGGT_GS(VGGT):
    def __init__(self,
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

        aggregated_tokens_list, frame_attn_list, global_attn_list, patch_start_idx, special_tokens = self.aggregator(images) # [(B x S x P x 2C) x L], 1 + num_register_tokens
        visualize_global_attn_map(global_attn_list, special_tokens, "./saving/global_attn.png")
        import pdb;pdb.set_trace()
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
                predictions["pose_enc_list"] = pose_enc_list
            outputs = self._render_gs(predictions, images, step)
            predictions["renders"] = outputs
            predictions["original_images"] = images
            return predictions

    def voxelize_gaussians(self, points, gs_features, gs_conf, voxel_size=0.002):
        """
        Differentiable(ish) voxelization with batch isolation.

        Inputs:
            points:      [B, S, H, W, 3]
            gs_features: [B, S, H, W, F]
            gs_conf:     [B, S, H, W, 1] or None
            voxel_size:  float or (3,) Tensor

        Returns:
            voxel_points:   [B, M, 3]   (0 padded where invalid)
            voxel_features: [B, M, F]   (0 padded where invalid)
        """
        assert points.dim() == 5 and points.size(-1) == 3
        assert gs_features.dim() == 5
        assert (gs_conf is None) or (gs_conf.dim() == 5 and gs_conf.size(-1) == 1)

        B, S, H, W, _ = points.shape
        F = gs_features.size(-1)
        device, dtype = points.device, points.dtype
        
        points_flatten = points.flatten(0,3) # [N,3] N is the num of gaussians
        voxel_indices = (points_flatten / voxel_size).round().int() # [N,3]
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        ) # obtain unique voxel coordinates
        
        conf_flat = gs_conf.flatten() # [N]
        gs_features_flat = gs_features.flatten(0,3) # [N,3]
        
        # Compute softmax weights per voxel
        conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0) # the max conf of gs within each voxel, [num_unique_voxels]
        conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(
            conf_exp, inverse_indices, dim=0
        ) # [num_unique_voxels]
        eps = 1e-6
        weights = (conf_exp / (voxel_weights[inverse_indices]) + eps).unsqueeze(-1) # [N,1]
        
        # Compute weighted avg of positions and features
        weighted_points = points_flatten * weights
        weighted_features = gs_features_flat.squeeze(1) * weights
        
        # Aggregate per voxel
        voxel_points = scatter_add(
            weighted_points, inverse_indices, dim=0
        ) # [num_unique_voxels, 3]
        voxel_features = scatter_add(
            weighted_features, inverse_indices, dim=0
        )
        
        return voxel_points, voxel_features
        
        
        
    
    def _render_gs(self, predictions: dict, images: torch.Tensor, step: int):
        """
        Render 3DGS image based on model predictions.
        Args:
            cfg (dict): Configuration dictionary
        """
        
        # parse gs features
        gs_features = predictions["gs_features"] # B,S,H,W,-1
        # gs_conf = torch.sigmoid(predictions["gs_conf"]).unsqueeze(-1) # B,S,H,W,1
        gs_conf = predictions["gs_conf"].unsqueeze(-1) # B,S,H,W,1
        points = predictions["world_points"] # B,S,H,W,3
        B,S,H,W,_ = gs_features.shape
        original_colors = images.permute(0,1,3,4,2).reshape(B,-1,3)
        
        # print(f"num before voxelization: {S*H*W}")
        global_points, gs_features = self.voxelize_gaussians(points, gs_features, gs_conf)
        # print(f"num after voxelization: {global_points.shape[1]}")
        
        # # soft filter gs features
        # conf_threshold = torch.mean(gs_conf, dim=[2,3]).view(B,S,1,1,1) # B,S,1,1,1
        # sharp_factor = 10.0
        # soft_mask = torch.sigmoid(sharp_factor*(gs_conf - conf_threshold)) * torch.sigmoid(sharp_factor*(gs_conf - 0.1))
        # gs_features = gs_features * soft_mask
        
        # feature extraction
        scale_factor = 1e-3
        Kx3 = (self.sh_degree + 1)**2 * 3
        sh_coeffs = gs_features[..., :Kx3].reshape(B,-1,Kx3//3,3)
        
        global_colors = sh_coeffs
        global_scales = torch.clamp(self.spls(gs_features[..., Kx3:Kx3+3])*scale_factor,max=0.5)
        global_quats = gs_features[..., Kx3+3:Kx3+7]
        global_opacities = torch.sigmoid(gs_features[..., -1:])
        
        # put predictions of each frame together for one scene
        # global_points = points.reshape(B,S*H*W,3)
        # global_quats = quats.reshape(B,S*H*W,4)
        # global_scales = scales.reshape(B,S*H*W,3)
        # global_colors = (original_colors if self.debug
                #  else sh)  # shape: (B, N, K, 3)
        # global_opacities = opacities.reshape(B,S*H*W,1).squeeze(-1)
        # global_conf = gs_conf.reshape(B,S*H*W,1)
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
        
        # dist shard gaussians
        if self.use_distributed_render:
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError("use_distributed_render=True but torch.distributed is not initialized")
            (global_points,
            global_quats,
            global_scales,
            global_colors,
            global_opacities) = self._dist_shard_gaussians(
                global_points, global_quats, global_scales, global_colors, global_opacities
            )
            
        if not self.use_distributed_render:
            renders, alphas, meta = rasterization(
                means=global_points,
                quats=global_quats,
                scales=global_scales,
                sh_degree=sh_degree,
                colors=global_colors,
                opacities=global_opacities.squeeze(-1),
                viewmats=viewmats,
                Ks=intrinsics,
                height=H,
                width=W,
                packed=True,
                distributed=self.use_distributed_render,
            )
            outputs = renders.permute(0,1,4,2,3).contiguous()
        else:
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError("use_distributed_render=True 但未初始化 torch.distributed")

            renders_list = []
            # 逐个 batch 处理；viewmats 与 Ks 也按 b 取出，去掉 batch 维
            for b in range(B):
                pts_b   = global_points[b:b+1]      # 形状 (1, N, 3)
                quat_b  = global_quats[b:b+1]       # (1, N, 4)
                scale_b = global_scales[b:b+1]      # (1, N, 3)
                col_b   = global_colors[b:b+1]      # (1, N, 3) 或 (1, N, K, 3)
                opa_b   = global_opacities[b:b+1]   # (1, N)

                # rank 内切分（并在函数内 squeeze 掉 batch 维）
                pts_b, quat_b, scale_b, col_b, opa_b = self._dist_shard_gaussians(
                    pts_b, quat_b, scale_b, col_b, opa_b
                )

                # 相机也去 batch 维（要求各 rank 相机数一致）
                viewmats_b = viewmats[b]     # (S, 4, 4)
                Ks_b       = intrinsics[b]   # (S, 3, 3)

                # 分布式模式下不支持 batch 维：直接喂 (N,*) 与 (S,*,*)
                r_b, a_b, m_b = rasterization(
                    means=pts_b,
                    quats=quat_b,
                    scales=scale_b,
                    sh_degree=sh_degree,
                    colors=col_b,
                    opacities=opa_b,
                    viewmats=viewmats_b,
                    Ks=Ks_b,
                    height=H,
                    width=W,
                    packed=True,
                    distributed=True,
                )
                # r_b 形状通常为 (S, H, W, 3)；收集起来后再堆叠回 B 维
                renders_list.append(r_b)

            renders = torch.stack(renders_list, dim=0)    # (B, S, H, W, 3)
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
    

def visualize_global_attn_map(attn_list, special_indices, save_path, cmap="viridis"):
    valid = [t for t in attn_list if t is not None]
    if not valid:
        raise ValueError("No valid attention maps.")
    cat = torch.cat([t.detach().float().cpu() for t in valid], dim=0)  # (sum_B, P, P)
    mean_map = cat.mean(dim=0)                                         # (P, P)
    P = mean_map.shape[0]

    if isinstance(special_indices, torch.Tensor):
        special_indices = special_indices.flatten().tolist()
    special_indices = [int(i) for i in (special_indices or []) if 0 <= int(i) < P]

    arr = mean_map.numpy()
    norm = Normalize(vmin=float(arr.min()), vmax=float(arr.max()))
    fig, ax = plt.subplots()
    im = ax.imshow(arr, interpolation="nearest", cmap=cmap, norm=norm)
    ax.axis("on")

    # annotate special rows/cols on the axes (outside the image)
    if special_indices:
        ax.set_xticks(special_indices)
        ax.set_yticks(special_indices)
        ax.set_xticklabels(["S"] * len(special_indices), color="red")
        ax.set_yticklabels(["S"] * len(special_indices), color="red")
        ax.tick_params(axis='both', which='major', labelsize=8, length=4, colors="red")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.colorbar(ScalarMappable(norm=norm, cmap=get_cmap(cmap)), ax=ax, fraction=0.046, pad=0.04)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return mean_map
    



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
    