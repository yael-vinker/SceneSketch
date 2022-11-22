import os
import argparse
import subprocess as sp
from shutil import copyfile


# CUDA_VISIBLE_DEVICES=5 python scripts/mlp_mask/mlp_clipasso.py --image "face.png" --mlp 0 --num_strokes 16
# CUDA_VISIBLE_DEVICES=1 python scripts/mlp_mask/mlp_clipasso.py --image "face.png" --mlp 1 --num_strokes 16
# CUDA_VISIBLE_DEVICES=6 python scripts/mlp_mask/mlp_clipasso.py --image "the_thinker.jpeg" --mlp 0 --num_strokes 16
# CUDA_VISIBLE_DEVICES=3 python scripts/mlp_mask/mlp_clipasso.py --image "the_thinker.jpeg" --mlp 1 --num_strokes 16

# CUDA_VISIBLE_DEVICES=1 python scripts/mlp_mask/mlp_clipasso.py --image "easy-background-crop.jpeg" --mlp 0 --num_strokes 8
# CUDA_VISIBLE_DEVICES=5 python scripts/mlp_mask/mlp_clipasso.py --image "easy-background-crop.jpeg" --mlp 1 --num_strokes 32

# 
# CUDA_VISIBLE_DEVICES=5 python scripts/mlp_mask/mlp_clipasso.py --image "semi-complex.jpeg" --mlp 1



# CUDA_VISIBLE_DEVICES=6 python scripts/mask_loss_new_test.py --image "semi-complex.jpeg" --mlp 1 --loss_mask "for"
# CUDA_VISIBLE_DEVICES=4 python scripts/mask_loss_new_test.py --image "semi-complex.jpeg" --mlp 1 --loss_mask "for" --num_strokes 64

# CUDA_VISIBLE_DEVICES=2 python scripts/mask_loss_new_test.py --image "semi-complex_mask.png" --mlp 1

# CUDA_VISIBLE_DEVICES=6 python scripts/mask_loss_new_test.py --image "semi-complex.jpeg" --mlp 1 --loss_mask "for_latent" --clip_mask_loss 1 --clip_conv_loss 0 --dilated_mask 1
# CUDA_VISIBLE_DEVICES=6 python scripts/mask_loss_new_test.py --image "semi-complex.jpeg" --mlp 1 --loss_mask "for" --clip_mask_loss 0 --clip_conv_loss 1 --dilated_mask 0


# path_to_files = "/home/vinker/dev/background_project/notebooks/complex_level_scenes/"
path_to_files = "/home/vinker/dev/backgroundCLIPasso/CLIPasso/target_images/"


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)
parser.add_argument("--mlp", type=int, default=0)
parser.add_argument("--num_strokes", type=int, default=32)

parser.add_argument("--clip_conv_loss", type=int, default=1)
parser.add_argument("--clip_mask_loss", type=int, default=0)

parser.add_argument("--loss_mask", type=str, default="none")
parser.add_argument("--dilated_mask", type=int, default=0)



args = parser.parse_args()
im_name = args.image
num_strokes = args.num_strokes
loss_mask = args.loss_mask
mlp_train = args.mlp
clip_mask_loss = args.clip_mask_loss
clip_conv_loss = args.clip_conv_loss

# use_wandb = 0
# wandb_project_name = "none"
# num_iter = 11
# num_sketches = 1

use_wandb = 1
wandb_project_name = "mlp_19_06"
num_iter = 1001
num_sketches = 2

model = "RN101"
clip_conv_layer_weights = "0,0,1.0,1.0,0"
clip_fc_loss_weight = 0.1
mask_object_attention = 0

lr = 1.0
if mlp_train:
    lr = 1e-4

output_pref = f"background_project/experiements/mlp_19_06"

file_ = f"{path_to_files}/{im_name}"

test_name = ""
if mlp_train:
    test_name += "mlp_"
if loss_mask != "none":
    test_name += f"mask_{loss_mask}_"
if args.dilated_mask:
    test_name += "dilated_"
test_name += f"clipasso_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}"
print(test_name)
sp.run(["python", 
        "scripts/mlp_mask/run_sketch.py", 
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
        "--loss_mask", str(loss_mask),
        "--mlp_train", str(mlp_train),
        "--lr", str(lr),
        "--clip_mask_loss", str(clip_mask_loss),
        "--clip_conv_loss", str(clip_conv_loss),
        "--dilated_mask", str(args.dilated_mask),
        "--mask_object_attention", str(mask_object_attention),
        "--use_wandb", str(use_wandb),
        "--wandb_project_name", str(wandb_project_name)])