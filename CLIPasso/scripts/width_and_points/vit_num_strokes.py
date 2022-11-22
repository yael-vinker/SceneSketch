import os
import argparse
import subprocess as sp
from shutil import copyfile


# CUDA_VISIBLE_DEVICES=6 python scripts/width_and_points/vit_num_strokes.py --num_paths 32
# CUDA_VISIBLE_DEVICES=2 python scripts/width_and_points/vit_num_strokes.py --num_paths 24
# CUDA_VISIBLE_DEVICES=5 python scripts/width_and_points/vit_num_strokes.py --num_paths 18
# CUDA_VISIBLE_DEVICES=1 python scripts/width_and_points/vit_num_strokes.py --num_paths 12
# 32 , 24, 18, 12

parser = argparse.ArgumentParser()
parser.add_argument("--num_paths", type=int, default=64)
args = parser.parse_args()


# ===================
# ====== real =======
# ===================
num_strokes = args.num_paths
num_sketches = 2
num_iter = 2001
use_wandb = 1
wandb_project_name = "big_test_07_27"
images = ["woman_city_mask.png", "london_mask.png", "bull_mask.png", "man_flowers_mask.png"]
output_pref = f"/home/vinker/dev/background_project/experiements/big_test_07_27"
loss_mask = "none"
mask_object_attention = 0
# ===================

path_to_files = "/home/vinker/dev/input_images/background_sketching/"
model = "ViT-B/32"
clip_fc_loss_weight = 0
mlp_train = 1
lr = 1e-4

layers = [4,6,7,8,11]

# im_name = args.image

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
                "scripts/width_and_points/run_sketch.py", 
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