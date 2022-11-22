import os
import argparse
import subprocess as sp
from shutil import copyfile

# CUDA_VISIBLE_DEVICES=1 python scripts/video/baseline_vit.py

parser = argparse.ArgumentParser()
parser.add_argument("--mask", type=int, default=0)
args = parser.parse_args()

# # ===================
# # ====== demo =======
# # ===================
# num_strokes = 64
# num_sketches = 1
# num_iter = 51
# use_wandb = 0
# wandb_project_name = "none"
# images = ["yael.jpg"]
# output_pref = f"background_project/experiements/resnet_vs_vit_demo"
# loss_mask = "none"
# mask_object_attention = 0
# if args.mask:
#         loss_mask = "for"
#         mask_object_attention = 1
# # ===================


# ===================
# ====== real =======
# ===================
num_strokes = 64
num_sketches = 2
num_iter = 2001
use_wandb = 0
wandb_project_name = "none"
images = ["swan_texture_orig_back.png"]
output_pref = f"/home/vinker/dev/background_project/experiements/video_24_07"
loss_mask = "none"
mask_object_attention = 0
# ===================

path_to_files = "/home/vinker/dev/input_images/video_sketching/"
model = "ViT-B/32"
clip_fc_loss_weight = 0
mlp_train = 1
lr = 1e-4

# layers = [3,4,5,6,7,8,9,10,11]
layers = [4]



for im_name in images:
    for layer in layers:
        clip_conv_layer_weights_int = [0 for k in range(12)]
        clip_conv_layer_weights_int[layer] = 1
        clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
        clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

        file_ = f"{path_to_files}/{im_name}"
        test_name = f"{model[:3]}_l{layer}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}"
        print(test_name)
        sp.run(["python", 
                "scripts/video/run_sketch.py", 
                "--target_file", file_,
                "--output_pref", output_pref,
                "--num_strokes", str(num_strokes),
                "--num_iter", str(num_iter),
                "--test_name", test_name,
                "--num_sketches", str(num_sketches),
                "--mask_object", "0",
                "--fix_scale", "0",
                "--clip_fc_loss_weight", str(clip_fc_loss_weight),
                "--clip_conv_layer_weights", clip_conv_layer_weights,
                "--clip_model_name", model,
                "--use_wandb", str(use_wandb),
                "--wandb_project_name", wandb_project_name,
                "--loss_mask",loss_mask,
                "--mask_object_attention", str(mask_object_attention),
                "--mlp_train", str(mlp_train),
                "--lr", str(lr)])



# parser.add_argument("--loss_mask", type=str, default="none", 
#                         help="mask the object during training, can be none|back|for, if you want to mask out the background choose back")
# parser.add_argument("--mask_object_attention", type=int, default=0)

# loss_mask = for