import torch
max_scale = 1
scale_factor = 1e-3

class Parser_GS():
    def __init__(self, mode):
        if mode not in ["predict_color_and_xyz", "predict_color_only", "predict_xyz_only", "predict_none"]:
            raise ValueError(f"expectin mode is either predict_color_and_xyz, predict_color_only, predict_xyz_only or predict_none, got {mode}")
        self.mode = mode
    
    def parse_feature(self, gs_features: torch.Tensor) -> dict:
        if not isinstance(gs_features, torch.Tensor):
            raise ValueError(f"gs_features should be a tensor, got {type(gs_features)}")
        
        if len(gs_features.shape) != 5:
            raise ValueError(f"expecting gs_features of shape B,S,H,W,f, got {gs_features.shape}")
        
        B, S, H, W, _ = gs_features.shape
        gs_features = gs_features.view(B,S,H*W,-1)
        
        means = None
        scales = None
        quats = None
        colors = None
        opacities = None
        
        spls = torch.nn.Softplus()
        if self.mode == "predict_color_and_xyz":
            if gs_features.shape[-1] != 14:
                raise ValueError(f"The last dimension of gs_features is expected to be 14 in {self.mode} mode, got {gs_features.shape[-1]}")
            means = gs_features[..., :3]
            scales = gs_features[..., 3:6]
            quats = gs_features[..., 6:10]
            colors = gs_features[..., 10:13]
            opacities = gs_features[..., 13:14]
            

            scales = spls(scales) * scale_factor
            scales = torch.clamp(scales, max=max_scale)
            colors = torch.sigmoid(colors)
            opacities = torch.sigmoid(opacities)
            
            return {
                "means": means,
                "scales": scales,
                "quats": quats,
                "colors": colors,
                "opacities": opacities,
                "image_size": (H,W)
            }
            
        if self.mode == "predict_color_only":
            if gs_features.shape[-1] != 11:
                raise ValueError(f"The last dimension of gs_features is expected to be 11 in {self.mode} mode, got {gs_features.shape[-1]}")

            scales = gs_features[..., :3]
            quats = gs_features[..., 3:7]
            colors = gs_features[..., 7:10]
            opacities = gs_features[..., 10:11]
            
            
            scales = spls(scales) * scale_factor
            scales = torch.clamp(scales, max=max_scale)
            colors = torch.sigmoid(colors)
            opacities = torch.sigmoid(opacities)
            
            return {
                "means": means,
                "scales": scales,
                "quats": quats,
                "colors": colors,
                "opacities": opacities,
                "image_size": (H,W)
            }
            
        if self.mode == "predict_xyz_only":
            if gs_features.shape[-1] != 11:
                    raise ValueError(f"The last dimension of gs_features is expected to be 11 in {self.mode} mode, got {gs_features.shape[-1]}")

            means = gs_features[..., :3]
            scales = gs_features[..., 3:6]
            quats = gs_features[..., 6:10]
            opacities = gs_features[..., 10:11]
            
            scales = spls(scales) * scale_factor
            scales = torch.clamp(scales, max=max_scale)
            opacities = torch.sigmoid(opacities)
            
            return {
                "means": means,
                "scales": scales,
                "quats": quats,
                "colors": colors,
                "opacities": opacities,
                "image_size": (H,W)
                
            }
        
        if self.mode == "predict_none":
            if gs_features.shape[-1] != 8:
                    raise ValueError(f"The last dimension of gs_features is expected to be 8 in {self.mode} mode, got {gs_features.shape[-1]}")
            scales = gs_features[..., :3]
            quats = gs_features[..., 3:7]
            opacities = gs_features[..., 7:8]
            
            scales = spls(scales) * scale_factor
            scales = torch.clamp(scales, max=max_scale)
            opacities = torch.sigmoid(opacities)
            
            return {
                "means": means,
                "scales": scales,
                "quats": quats,
                "colors": colors,
                "opacities": opacities,
                "image_size": (H,W)
                
            }
        
        
        
        