import os
import argparse
import subprocess as sp
from shutil import copyfile

# ===========================================
# ========= baseline vit script =============
# ===========================================
# This script is to run the baseline sketching method with the ViT clip model.
# We use this to create the first row in the abstraciton matrix.
# You can use this to create both the objects and background. 
# The default parameters are set for the background case.
# Example of a running command:
# CUDA_VISIBLE_DEVICES=1 python scripts/all_together/baseline_vit.py --im_name "man_flowers" --layer_opt 2 --object_or_background "object" --resize_obj 1
# CUDA_VISIBLE_DEVICES=2 python scripts/all_together/baseline_vit.py --im_name "man_flowers" --layer_opt 8 --object_or_background "background"


parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer_opt", type=int, default=4)
parser.add_argument("--object_or_background", type=str, default="background")
parser.add_argument("--resize_obj", type=int, default=0)
args = parser.parse_args()


path_to_files = "/home/vinker/dev/input_images/background_sketching/" # where the input images are located
output_pref = f"/home/vinker/dev/background_project/experiements/all_together_09_09"
model = "ViT-B/32" # for background it's vit
clip_fc_loss_weight = 0
mlp_train = 1
lr = 1e-4
num_strokes = 64
num_sketches = 2

# if you run on objects, this need to be changed:
gradnorm = 0
mask_object = 0
if args.object_or_background == "object":
    gradnorm = 1
    mask_object = 1
    # change the images as well


# ===================
# ====== demo =======
# ===================
# output_pref = f"/home/vinker/dev/background_project/experiements/obj_vit_demo"
# num_strokes = 64
# num_sketches = 2
# num_iter = 51
# use_wandb = 0
# wandb_project_name = "none"
# im_filename = f"{args.im_name}.jpg"
# if args.object_or_background == "background":
#     im_filename = f"{args.im_name}_mask.png"
# ===================


# ===================
# ====== real =======
# ===================
num_strokes = 64
num_sketches = 2
num_iter = 2001
use_wandb = 0
wandb_project_name = "all_together_09_09"
im_filename = f"{args.im_name}.jpg"
if args.object_or_background == "background":
    im_filename = f"{args.im_name}_mask.png"
# ===================


# set the weights for each layer
clip_conv_layer_weights_int = [0 for k in range(12)]
if args.object_or_background == "object":
    # we combine two layers if we train on objects
    clip_conv_layer_weights_int[4] = 0.5
clip_conv_layer_weights_int[args.layer_opt] = 1
clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

file_ = f"{path_to_files}/{im_filename}"
test_name = f"{args.object_or_background}_l{args.layer_opt}_{num_strokes}s_{os.path.splitext(im_filename)[0]}_resize{args.resize_obj}"
print(test_name)
sp.run(["python", 
        "scripts/all_together/run_sketch.py", 
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
        "--mlp_train", str(mlp_train),
        "--lr", str(lr),
        "--gradnorm", str(gradnorm),
        "--resize_obj", str(args.resize_obj)])
