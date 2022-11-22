import os
import argparse
import subprocess as sp
from shutil import copyfile

# ================
# MLP vs no MLP (no mask)
# ================
# CUDA_VISIBLE_DEVICES=1 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 0
# CUDA_VISIBLE_DEVICES=2 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 1
# CUDA_VISIBLE_DEVICES=3 python scripts/mlp_mask/mlp_vit.py --image "complex-scene-crop.png" --mlp 0
# CUDA_VISIBLE_DEVICES=5 python scripts/mlp_mask/mlp_vit.py --image "complex-scene-crop.png" --mlp 1

# ================
# inpainting
# ================
# CUDA_VISIBLE_DEVICES=1 python scripts/mlp_mask/mlp_vit.py --image "complex-scene-crop_mask.png" --mlp 1 --clip_conv_loss_type "Cos"
# CUDA_VISIBLE_DEVICES=5 python scripts/mlp_mask/mlp_vit.py --image "semi-complex_mask.png" --mlp 1 --clip_conv_loss_type "Cos"


# ================
# cos similarity
# ================
# CUDA_VISIBLE_DEVICES=5 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 0 --clip_conv_loss_type "Cos"

# ================
# latent dilated mask mlp vs np mlp
# ================
# CUDA_VISIBLE_DEVICES=1 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 0 --clip_conv_loss 0 --clip_mask_loss 1
# CUDA_VISIBLE_DEVICES=2 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 1 --clip_conv_loss 0 --clip_mask_loss 1
# CUDA_VISIBLE_DEVICES=5 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 0 --clip_conv_loss 0 --clip_mask_loss 1 --loss_mask "for_latent" --dilated_mask 1 --clip_conv_loss_type "Cos"
# CUDA_VISIBLE_DEVICES=5 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 1 --clip_conv_loss 0 --clip_mask_loss 1 --loss_mask "for_latent" --dilated_mask 1 --clip_conv_loss_type "Cos"
# CUDA_VISIBLE_DEVICES=7 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 1 --clip_conv_loss 0 --clip_mask_loss 1 --loss_mask "for_latent" --dilated_mask 1 --clip_conv_loss_type "Cos" --mask_object_attention 0 --mask_attention 1

# CUDA_VISIBLE_DEVICES=5 python scripts/mlp_mask/mlp_vit.py --image "complex-scene-crop.png" --mlp 1 --clip_conv_loss 0 --clip_mask_loss 1 --loss_mask "for_latent" --dilated_mask 1 --clip_conv_loss_type "Cos" --num_strokes 64
# CUDA_VISIBLE_DEVICES=6 python scripts/mlp_mask/mlp_vit.py --image "complex-scene-crop.png" --mlp 1 --clip_conv_loss 0 --clip_mask_loss 1 --loss_mask "for_latent" --dilated_mask 1 --clip_conv_loss_type "Cos" --mask_object_attention 1 --mask_attention 0 
# CUDA_VISIBLE_DEVICES=7 python scripts/mlp_mask/mlp_vit.py --image "complex-scene-crop.png" --mlp 1 --clip_conv_loss 0 --clip_mask_loss 1 --loss_mask "for_latent" --dilated_mask 1 --clip_conv_loss_type "Cos" --mask_object_attention 0 --mask_attention 1 
# CUDA_VISIBLE_DEVICES=5 python scripts/mlp_mask/mlp_vit.py --image "complex-scene-crop.png" --mlp 1 --clip_conv_loss 0 --clip_mask_loss 1 --loss_mask "for_latent" --dilated_mask 1 --clip_conv_loss_type "Cos" --mask_object_attention 1 --mask_attention 1 


# ================
# mask cls test
# ================
# CUDA_VISIBLE_DEVICES=1 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 1 --clip_conv_loss 0 --clip_mask_loss 1 --mask_cls "only_cls" --clip_conv_loss_type "Cos"
# CUDA_VISIBLE_DEVICES=2 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 1 --clip_conv_loss 0 --clip_mask_loss 1 --mask_cls "only_cls" --loss_mask "for_latent" --dilated_mask 1 --clip_conv_loss_type "Cos"
# CUDA_VISIBLE_DEVICES=3 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 1 --clip_conv_loss 0 --clip_mask_loss 1 --mask_cls "cls_out" --clip_conv_loss_type "Cos"
# CUDA_VISIBLE_DEVICES=3 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 1 --clip_conv_loss 0 --clip_mask_loss 1 --mask_cls "none" --clip_conv_loss_type "Cos"
# CUDA_VISIBLE_DEVICES=5 python scripts/mlp_mask/mlp_vit.py --image "semi-complex.jpeg" --mlp 1 --clip_conv_loss 0 --clip_mask_loss 1 --mask_cls "cls_out" --loss_mask "for_latent" --dilated_mask 1 --clip_conv_loss_type "Cos"

# ================
# deletee strokes (inpainting images)
# ================
# CUDA_VISIBLE_DEVICES=1 python scripts/mlp_mask/mlp_vit.py --image "complex-scene-crop_mask.png" --mlp 1 --clip_conv_loss_type "Cos"

# CUDA_VISIBLE_DEVICES=5 python scripts/mlp_mask/mlp_vit.py --image "semi-complex_mask.png" --mlp 1 --clip_conv_loss_type "Cos"


path_to_files = "/home/vinker/dev/background_project/notebooks/complex_level_scenes/"

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)
parser.add_argument("--mlp", type=int, default=0)
parser.add_argument("--num_strokes", type=int, default=32)

parser.add_argument("--clip_conv_loss", type=int, default=1)
parser.add_argument("--clip_mask_loss", type=int, default=0)

parser.add_argument("--loss_mask", type=str, default="none")
parser.add_argument("--dilated_mask", type=int, default=0)
parser.add_argument("--mask_attention", type=int, default=0)

parser.add_argument("--clip_conv_loss_type", type=str, default="L2")
parser.add_argument("--mask_cls", type=str, default="none")
parser.add_argument("--mask_object_attention", type=int, default=0)
parser.add_argument("--width_optim", type=int, default=0)
parser.add_argument("--width_loss_weight", type=str, default=0)


args = parser.parse_args()
im_name = args.image
num_strokes = args.num_strokes
loss_mask = args.loss_mask
mlp_train = args.mlp
clip_mask_loss = args.clip_mask_loss
clip_conv_loss = args.clip_conv_loss

use_wandb = 0
wandb_project_name = "none"
layers = [4]
num_iter = 11
num_sketches = 1

# use_wandb = 1
# wandb_project_name = "mlp_19_06"
# layers = [2,3,4,5,6,7,8,9,10,11]
# # layers = [7]
# num_iter = 1001
# num_sketches = 2

model = "ViT-B/32"
clip_fc_loss_weight = 0
mask_object_attention = args.mask_object_attention

lr = 1.0
if mlp_train:
    lr = 1e-4

# output_pref = f"background_project/experiements/mlp_19_06"
output_pref = f"background_project/experiements/keys_demo_07_11"


file_ = f"{path_to_files}/{im_name}"
for i in layers:
    clip_conv_layer_weights_int = [0 for k in range(12)]
    clip_conv_layer_weights_int[i] = 1
    clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
    clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

    print(clip_conv_layer_weights)
    print(model)
    print(clip_fc_loss_weight)
    
    test_name = ""
    
    if mlp_train:
        test_name += "mlp_"
    if args.width_optim:
        test_name += f"sdel{args.width_loss_weight}_"
    if args.clip_conv_loss_type != "L2":
        test_name += f"{args.clip_conv_loss_type}_"
    if loss_mask != "none":
        test_name += f"mask_{loss_mask}_"
    if args.mask_attention:
        test_name += f"attn_"
    if args.mask_cls != "none":
        test_name += f"{args.mask_cls}_"
    if args.dilated_mask:
        test_name += "dilated_"
    test_name += f"{model[:3]}_l{i}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}"
    test_name+= "_2806"
    print(test_name)
    sp.run(["python", 
            "scripts/mlp_mask/run_sketch.py", 
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
            "--loss_mask", str(loss_mask),
            "--mlp_train", str(mlp_train),
            "--lr", str(lr),
            "--clip_mask_loss", str(clip_mask_loss),
            "--clip_conv_loss", str(clip_conv_loss),
            "--dilated_mask", str(args.dilated_mask),
            "--mask_object_attention", str(mask_object_attention),
            "--use_wandb", str(use_wandb),
            "--wandb_project_name", str(wandb_project_name),
            "--clip_conv_loss_type", str(args.clip_conv_loss_type),
            "--mask_cls", args.mask_cls,
            "--width_optim", str(args.width_optim),
            "--width_loss_weight", str(args.width_loss_weight),
            "--mask_attention", str(args.mask_attention)])