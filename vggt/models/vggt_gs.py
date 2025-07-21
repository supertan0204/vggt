from vggt.models.vggt import VGGT
import torch
import torch.nn as nn
import open3d as o3d


from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from training.gs_feature_parser import Parser_GS
from vggt.utils.geometry import closed_form_inverse_se3
import math


from gsplat.rendering import rasterization



class VGGT_GS(VGGT):
    def __init__(self, predict_enable,
                #  img_size=518, 
                #  patch_size=14, 
                 embed_dim=1024,
                 enable_camera=True,
                 enable_depth=True,
                 enable_point=True,
                 enable_track=True,
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
        mode = ""
        if predict_enable.xyz and predict_enable.color:
            # gs head need to predict color and means 
            self.gs_head = DPTHead(
                dim_in=2*embed_dim, 
                output_dim=15, # xyz:3, scale:3, rotation:4, rgb:3, opacity:1, conf:1
                activation="inv_log", 
                conf_activation="expp1"
            )
            mode = "predict_color_and_xyz"
        elif not predict_enable.xyz and not predict_enable.color:
            self.gs_head = DPTHead(
                dim_in=2*embed_dim,
                output_dim=9, # scale:3, rotation:4, opacity:1, conf:1
                activation="inv_log",
                conf_activation="expp1",
            )
            mode = "predict_none"
        else:
            self.gs_head = DPTHead(
                dim_in=2*embed_dim, 
                output_dim=12, # scale:3, rotation:4, rgb/xyz:3, opacity:1, conf:1
                activation="inv_log", 
                conf_activation="expp1", 
            )
            if not predict_enable.xyz and predict_enable.color:
                mode = "predict_color_only"
            else: 
                mode = "predict_xyz_only"
        self.gs_feature_parser = Parser_GS(mode)
    
    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
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

            outputs = self._render_gs(predictions, images)
            predictions["renders"] = outputs
            return predictions
    
    def _render_gs(self, predictions: dict, images: torch.Tensor):
        """
        Render 3DGS image based on model predictions.
        Args:
            cfg (dict): Configuration dictionary
        """
        
        
        # import pdb;pdb.set_trace()
        gs_feature_dict = self.gs_feature_parser.parse_feature(predictions["gs_features"])
        
        means = gs_feature_dict["means"]
        scales = gs_feature_dict["scales"]
        quats = gs_feature_dict["quats"]
        colors = gs_feature_dict["colors"]
        opacities = gs_feature_dict["opacities"]
        image_size = gs_feature_dict["image_size"]
        
        H = image_size[0]
        W = image_size[1]
        
        
        gs_confs = predictions["gs_conf"]
        gs_confs = torch.sigmoid(gs_confs)

        # Camera parameters
        pose_enc = predictions["pose_enc"]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw=images.shape[-2:]) # extrinsics: BxSx3x4, intrinsics: BxSx3x3
        B, S, _, _ = extrinsics.shape    
        viewmats = torch.zeros((B, S, 4, 4), device="cuda")
        viewmats[:, :, :3, :4] = extrinsics
        viewmats[:, :, 3, 3] = 1
        # viewmats = closed_form_inverse_se3(extrinsics[0]).unsqueeze(0)
        
        outputs = []
        world_points = predictions["world_points"]
        # import pdb;pdb.set_trace()
        
        
        # focal = float(W) / math.tan(math.pi/4.)
        # K = torch.tensor(
        #     [
        #         [focal, 0, W / 2],
        #         [0, focal, H / 2],
        #         [0, 0, 1],
        #     ],
        #     device="cuda",
        # )
        # for b in range(B):
        #     for s in range(S):
        #         print(intrinsics[b,s])
        # Render GS for multiple input views
        
        # assign global points by concating all pointmaps
        global_points = world_points.reshape(B,-1,3) if means is None else means.reshape(B,-1,3)
        global_quats = quats.reshape(B,-1,4)
        global_scales = scales.reshape(B,-1,3)
        global_colors = colors.reshape(B,-1,3) if colors is not None else images.permute(0,1,3,4,2).reshape(B,-1,3)
        global_opacities = opacities.reshape(B,-1,1)
        
        save_ply(
                    global_points[0], 
                    global_colors[0], 
                    "debug.ply"
                )
        for b in range(B):
            renders = []
            alphas = []
            meta = []
            for s in range(S):
                # print(f"...........{world_points.shape}..........")
                r, a, m = rasterization(
                # means = world_points[b,s].reshape(H*W,3),
                # quats=quats[b,s].reshape(H*W,4),
                # scales = torch.ones(H*W,3,device="cuda")*1e-3,
                # colors=images[b,s].permute(1,2,0).reshape(H*W,3),
                # opacities=torch.ones(H*W,device="cuda"),
                # scales = test_scales[b],
                # scales=scales[b,s],
                means = global_points[b],
                quats=global_quats[b],
                scales=global_scales[b],
                colors=global_colors[b],
                # colors = test_colors[b],
                opacities=global_opacities[b].squeeze(),
                # opacities = test_opacities[b].squeeze(),
                viewmats=viewmats[b,s][None],
                Ks=intrinsics[b,s][None],
                # Ks = K[None],
                # opacities=opacities[b,s].squeeze(),
                height=image_size[0],
                width=image_size[1],
                packed=False,
                # backgrounds=torch.tensor([0.0,0.0,0.0],device="cuda",dtype=torch.float32)
                )
                # visualize render for one scene
                # import pdb;pdb.set_trace()
                # img_tensor = r.squeeze().detach().cpu() * 255.
                # img_np = img_tensor.numpy().astype(np.uint8)
                # img_rendered = Image.fromarray(img_np)
                # img_rendered.save(f"./saving/scene_{s}.png")
                
                # save_ply(world_points[b,s].reshape(H*W,3), 
                #          images[b,s].permute(1,2,0).reshape(H*W,3),
                #          f'scene_{s}.ply')
                
                renders.append(r.squeeze())
                del a
                del m
                # alphas.append(a.squeeze())
                # meta.append(m)
            renders = torch.stack(renders, dim=0)
            # alphas = torch.stack(alphas, dim=0)
            
            batch_output = {}
            
            batch_output["renders"] = renders
            # batch_output["alphas"] = alphas
            # batch_output["meta"] = meta
            batch_output["gs_conf"] = gs_confs[b]
            # outputs.append(batch_output)
            outputs.append(renders.permute(0,3,1,2).contiguous())

        return torch.stack(outputs, dim=0)
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