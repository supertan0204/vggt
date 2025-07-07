from vggt.models.vggt import VGGT
import torch
import torch.nn as nn
import open3d as o3d


from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import math


from gsplat.rendering import rasterization



class VGGT_GS(VGGT):
    def __init__(self, gs_pos_predict="new_xyz", 
                 img_size=518, 
                 patch_size=14, 
                 embed_dim=1024,
                 enable_camera=True,
                 enable_depth=True,
                 enable_point=True,
                 enable_track=True):  # Ensure embed_dim is passed
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
        super().__init__()  # Pass embed_dim to the parent class
        if gs_pos_predict == "new_xyz":
            self.gs_head = DPTHead(
                dim_in=2*embed_dim, 
                output_dim=15, # xyz:3, scale:3, rotation:4, rgb:3, opacity:1, conf: 1
                activation="inv_log", 
                conf_activation="expp1"
            )
        elif gs_pos_predict == "from_vggt":
            self.gs_head = DPTHead(
                dim_in=2*embed_dim, 
                output_dim=12, # scale:3, rotation:4, rgb:3, opacity:1, conf: 1
                activation="inv_log", 
                conf_activation="expp1", 
            )
        else:
            raise ValueError(f"Unsupported gs_pos_predict: {gs_pos_predict}. Choose from 'new_xyz' or 'from_vggt'.")
    
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

            outputs = self._render_gs(predictions)
            predictions["renders"] = outputs
            return predictions
             
    def freeze(self):
        """
        Freeze the model parameters except for the GS head.
        """
        for name, param in self.named_parameters():
            if "gs_head" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def _render_gs(self, predictions: dict):
        """
        Render 3DGS image based on model predictions.
        Args:
            cfg (dict): Configuration dictionary
        """
        # GS parameters
        B, S, H, W, _ = predictions["gs_features"].shape
        # print(f"H,W={H,W}")
        # import pdb; pdb.set_trace()
        gs_features = predictions["gs_features"].view(B, S, H*W, -1)

        means = gs_features[..., :3]  # BxSxHWx3
        
        # for i in range(100):
        #     print(means[0,0,i,:])
        #     import pdb;pdb.set_trace()
        scales = gs_features[..., 3:6] # BxSxHWx3
        scales = torch.relu(scales)
        quats = gs_features[..., 6:10] # BxSxHWx4
        colors = gs_features[..., 10:13] # BxSxHWx3
        # make sure colors are above zero
        colors = torch.sigmoid(colors)
        # for i in range(100):
        #     print(colors[0,0,i,:])
        #     import pdb;pdb.set_trace()
        opacities = gs_features[..., 13:14] # BxSxHWx1
        opacities = torch.ones_like(opacities)
        # opacities = torch.sigmoid(opacities)
        gs_confs = predictions["gs_conf"]
        gs_confs = torch.sigmoid(gs_confs)

        # Camera parameters
        pose_enc = predictions["pose_enc"]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H,W)) # extrinsics: BxSx3x4, intrinsics: BxSx3x3

        B, S, _, _ = extrinsics.shape
        extra_row = torch.tensor([0, 0, 0, 1], dtype=extrinsics.dtype, device=extrinsics.device).view(1, 1, 1, 4)
        viewmats = torch.cat((extrinsics, extra_row.expand(B, S, 1, 4)), dim=2) # B,S,4,4       
        # for i in range(B):
        #     for j in range(S):
        #         print(viewmats[i,j])
        # import pdb;pdb.set_trace()
       
        outputs = []
        # TODO: Remove this part, this is just test
        world_points = predictions["world_points"]

        focal = 0.5*float(W) / math.tan(math.pi/4.)
        K = torch.tensor(
            [
                [focal, 0, W / 2],
                [0, focal, H / 2],
                [0, 0, 1],
            ],
            device="cuda",
        )
        # print(f"K: {K}")
        # for b in range(B):
        #     for s in range(S):
        #         print(intrinsics[b,s])
        #         import pdb;pdb.set_trace()
        # Render GS for multiple input views
        for b in range(B):
            renders = []
            alphas = []
            meta = []
            for s in range(S):
                # print(f"...........{world_points.shape}..........")
                # import pdb;pdb.set_trace()
                r, a, m = rasterization(
                # means=means[b,s],
                means=world_points[b,s].view(H*W, 3),
                quats=quats[b,s],
                scales=scales[b,s],
                colors=colors[b,s],
                viewmats=viewmats[b,s][None,:,:],
                Ks=intrinsics[b,s][None],
                # Ks = K[None],
                opacities=opacities[b,s].squeeze(),
                width=W,
                height=H,
                backgrounds=torch.tensor([0.0,0.0,0.0],device="cuda",dtype=torch.float32)
                )
                renders.append(r.squeeze())
                alphas.append(a.squeeze())
                meta.append(m)
            renders = torch.stack(renders, dim=0)
            alphas = torch.stack(alphas, dim=0)
            
            batch_output = {}
            
            batch_output["renders"] = renders
            batch_output["alphas"] = alphas
            batch_output["meta"] = meta
            batch_output["gs_conf"] = gs_confs[b]
            # outputs.append(batch_output)
            outputs.append(renders.permute(0,3,1,2))

        return torch.stack(outputs, dim=0)
