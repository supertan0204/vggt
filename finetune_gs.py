from vggt.models.vggt_gs import VGGT_GS
import torch.nn.init as init
from vggt.utils.load_fn import load_and_preprocess_images
import hydra
# import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from training.trainer import Trainer


def init_model_head(model, head_name: str, init_fn: str):
    if hasattr(model, head_name):
        print(f"Initializing {head_name} with {init_fn} initialization...")
        head = getattr(model, head_name)
        for param in head.parameters():
            if param.dim() > 1:  # Check if it is a weight matrix (not a bias)
                if init_fn == "kaiming":
                    init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')  # Kaiming normal initialization
                elif init_fn == "xavier":
                    init.xavier_normal_(param)  # Xavier normal initialization
                else:
                    raise ValueError(f"Unsupported initialization function: {init_fn}")
            else:
                init.zeros_(param)  # Initialize biases to zeros if it's a bias term

        # Notify the user that the initialization was successful
        print(f"{head_name} initialized successfully with {init_fn} initialization.")
    else:
        print(f"{head_name} not found in the model.")



@hydra.main(config_path="training/config", config_name="default")
def tune(cfg: DictConfig):
    # Set up the output directory
    # output_dir = Path(
    #     hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    # )
    # print(f"[INFO] Saving outputs to {output_dir}.")
    # Set up logging with wandb
    # if cfg.wandb.mode != "disabled":
    #     wandb.init(project="tune_vggt_gs",
    #        name=f"{cfg.wandb.name}",
    #        config=OmegaConf.to_container(cfg),
    #        )
    #     print("[INFO] WandB initialized.")
    # else:
    #     raise ValueError("WandB is disabled or not available. Please check your configuration.")
    
    # Set up trainer
    trainer = Trainer(**cfg)
    print("[INFO] Trainer initialized.")
    trainer.train_gs()










    # device = cfg.train.device
    # dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # # Load model
    # print("[INFO] start loading model...")
    # model = VGGT_GS.from_pretrained("facebook/VGGT-1B").to(device)
    # print("[INFO] model loaded successfully.")

    # # Initialize the heads of the model with specific method
    # init_model_head(model, "camera_head", "kaiming")
    # # Freeze all layers except gs head
    # model.freeze()

    # # Check if freeze has been successful
    # # for name, param in model.named_parameters():
    # #     print(f"name: {name}, requires_grad: {param.requires_grad}")
    # image_names = ["/workspace/user_code/vggt/examples/kitchen/images/00.png", 
    #             "/workspace/user_code/vggt/examples/kitchen/images/01.png",
    #             "/workspace/user_code/vggt/examples/kitchen/images/02.png",
    #             "/workspace/user_code/vggt/examples/kitchen/images/03.png",
    #             "/workspace/user_code/vggt/examples/kitchen/images/04.png"
    #             ]  
    # images = load_and_preprocess_images(image_names).to(device)
    # with torch.no_grad():
    #     with torch.amp.autocast("cuda", dtype=dtype):
    #     # Predict attributes including cameras, depth maps, and point maps.
    #         predictions = model(images)
    # for i in range(len(predictions)):
    #     predict_one_batch = predictions[i]
    #     print(f"Batch {i} predictions: {predict_one_batch['renders'].shape}")


    # print("successfully made predictions")
    
    

if __name__ == "__main__":
    tune()












# # Load and preprocess example images (replace with your own image paths)
# image_names = ["/workspace/user_code/vggt/examples/llff_fern/images/000.png", "/workspace/user_code/vggt/examples/llff_fern/images/001.png", "/workspace/user_code/vggt/examples/llff_fern/images/003.png"]  
# images = load_and_preprocess_images(image_names).to(device)

# with torch.no_grad():
#     with torch.cuda.amp.autocast(dtype=dtype):
#         # Predict attributes including cameras, depth maps, and point maps.
#         predictions = model(images)