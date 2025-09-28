import torch
from vggt.models.vggt_gs import VGGT_GS
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT_GS.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()
image_names = ["examples/kitchen/images/00.png", 
               "examples/kitchen/images/01.png",  
               "examples/kitchen/images/02.png",
               "examples/kitchen/images/03.png"]
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)