import os
import argparse
import subprocess as sp
from shutil import copyfile

# CUDA_VISIBLE_DEVICES=1 python scripts/width_and_points/baseline_clipasso.py

# parser = argparse.ArgumentParser()
# parser.add_argument("--mask", type=int, default=0)
# args = parser.parse_args()

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
use_wandb = 1
wandb_project_name = "big_test_07_27"
# images = ["woman_city.jpg", "van2.jpg", "lion.jpg", "man_flowers.jpg"]
images = ["bull.jpg", "zebra.jpg", "panda.jpg", "chicks.jpg", "japan.jpg", "boat_tree.jpg", "venice.jpg", "van.jpg"]
# images = ["face.png", "semi-complex.jpeg", "complex-scene-crop.png", "flamingo.png", "horse.png"]

output_pref = f"/home/vinker/dev/background_project/experiements/big_test_07_27"
loss_mask = "none"
mask_object_attention = 0
# ===================

path_to_files = "/home/vinker/dev/input_images/background_sketching/"
model = "RN101"
clip_conv_layer_weights = "0,0,1.0,1.0,0"
clip_fc_loss_weight = 0.1
mlp_train = 1
lr = 1e-4


for im_name in images:
        file_ = f"{path_to_files}/{im_name}"
        test_name = f"{model[:3]}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}"
        print(test_name)
        sp.run(["python", 
                "scripts/width_and_points/run_sketch.py", 
                "--target_file", file_,
                "--output_pref", output_pref,
                "--num_strokes", str(num_strokes),
                "--num_iter", str(num_iter),
                "--test_name", test_name,
                "--num_sketches", str(num_sketches),
                "--mask_object", "1",
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