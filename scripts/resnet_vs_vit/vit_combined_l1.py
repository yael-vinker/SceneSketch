import os
import argparse
import subprocess as sp
from shutil import copyfile

# CUDA_VISIBLE_DEVICES=1 python scripts/resnet_vs_vit/vit_combined_l1.py --im_name "semi-complex.jpeg" --mask 1 --num_strokes 32

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--mask", type=int, default=0)
parser.add_argument("--num_strokes", type=int, default=64)
args = parser.parse_args()

# # ===================
# # ====== demo =======
# # ===================
# num_strokes = 64
# num_sketches = 1
# num_iter = 51
# use_wandb = 0
# wandb_project_name = "none"
# images = ["yael.jpg", "forest.jpg", "semi-complex.jpeg", "complex-scene-crop.png", "assaf.jpg"]
# output_pref = f"background_project/experiements/resnet_vs_vit_demo"
# loss_mask = "none"
# mask_object_attention = 0
# if args.mask:
#     loss_mask = "for"
#     mask_object_attention = 1
# # ===================


# ===================
# ====== real =======
# ===================
num_strokes = args.num_strokes
num_sketches = 2
num_iter = 2001
use_wandb = 1
wandb_project_name = "rn_vs_vit_18_07"
# images = ["yael.jpg", "forest.jpg", "semi-complex.jpeg", "complex-scene-crop.png", "assaf.jpg"]
# images = ["horse_large.png"]
output_pref = f"background_project/experiements/resnet_vs_vit_07_18"
loss_mask = "none"
mask_object = 0
mask_object_attention = 0
if args.mask:
    mask_object = 1
    # loss_mask = "for"
    # mask_object_attention = 1
# ===================

path_to_files = "/home/vinker/dev/input_images/background_sketching/"
model = "ViT-B/32"
clip_fc_loss_weight = 0
# layers = [0,1,2,3, 4, 5,6,7,8,9,10,11]
clip_conv_layer_weights_int = [0 for k in range(12)]
clip_conv_layer_weights_int[1] = 1
clip_conv_layer_weights_int[2] = 1
clip_conv_layer_weights_int[-2] = 0.5
clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)
# layers = [0,1,2,3]

im_name = args.im_name
# for im_name in images:
file_ = f"{path_to_files}/{im_name}"
# for i in layers:
    # clip_conv_layer_weights_int = [0 for k in range(12)]
    # clip_conv_layer_weights_int[i] = 1
    # clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
    # clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)
test_name = f"{model[:3]}_l1_l2_l10-05_{num_strokes}s_mask{mask_object}_{os.path.splitext(os.path.basename(file_))[0]}"
sp.run(["python", 
        "scripts/resnet_vs_vit/run_sketch.py", 
        "--target_file", file_,
        "--output_pref", output_pref,
        "--num_strokes", str(num_strokes),
        "--num_iter", str(num_iter),
        "--test_name", test_name,
        "--num_sketches", str(num_sketches),
        "--mask_object", str(mask_object),
        "--fix_scale", "0",
        "--clip_fc_loss_weight", str(clip_fc_loss_weight),
        "--clip_conv_layer_weights", clip_conv_layer_weights,
        "--clip_model_name", model,
        "--use_wandb", str(use_wandb),
        "--wandb_project_name", wandb_project_name,
        "--loss_mask",loss_mask,
        "--mask_object_attention", str(mask_object_attention)])
