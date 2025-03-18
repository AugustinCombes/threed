import torch
import torchvision
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
import json
from threed.utils import sample_video, predictions_to_glb, make_serializable
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

sampled_images = sample_video("/Users/gus/Desktop/threed/inputs/IMG_3875.MOV")

tmp_path = 'inputs/tmp'
os.makedirs(tmp_path, exist_ok=True)
for idx, sampled_image in enumerate(sampled_images):
    # Convert tensor shape from [H,W,C] to [C,H,W] and normalize to [0,1]
    # PyTorch expects [C,H,W] format for image tensors
    permuted_image = sampled_image.permute(2, 0, 1).float() / 255.0
    torchvision.utils.save_image(
        permuted_image,
        os.path.join(tmp_path, f'{idx}.png')
    )

# Load and preprocess example images (replace with your own image paths)
image_names = os.listdir(tmp_path)
images = load_and_preprocess_images([os.path.join(tmp_path, image_name) for image_name in image_names]).to(device)

with torch.no_grad():
    # Predict attributes including cameras, depth maps, and point maps.
    predictions = model(images)

extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic
predictions["images"] = predictions["images"].numpy()

# serializable_predictions = make_serializable(predictions)
# with open('predictions.json', 'w') as f:
#     json.dump(serializable_predictions, f)

glbscene = predictions_to_glb(
    predictions,
    conf_thres=3.0,
    filter_by_frames="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    target_dir="output.glb",
    prediction_mode="Pointmap Regression",
)
glbscene.export(file_obj="output.glb")
