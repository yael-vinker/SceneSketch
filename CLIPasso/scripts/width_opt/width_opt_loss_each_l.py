import os
import argparse
import subprocess as sp
from shutil import copyfile
import time

def get_svg_file(path):
    files = os.listdir(path)
    files = [f for f in files if ".svg" in f]
    return files[0]

# ================
# width_optim
# ================
# CUDA_VISIBLE_DEVICES=1 python scripts/width_opt/width_opt_loss.py --image "house_layer4.png" --clip_conv_loss_type "Cos" --width_loss_type "L1_hinge"
# if mlp 0 than svg path needs to be the final output, if 1 then it should be the initialization

# ================
# cos vs l2
# ================
# CUDA_VISIBLE_DEVICES=6 python scripts/width_opt/width_opt_loss_each_l.py --image "house_layer4.png" --clip_conv_loss_type "Cos" --width_loss_type "L1_hinge"


source_im_name="semi-complex_mask"
path_to_files = "/home/vinker/dev/backgroundCLIPasso/CLIPasso/notebooks/"
output_pref = f"/home/vinker/dev/background_project/experiements/width_05_07"
path_res_pref = "/home/vinker/dev/background_project/experiements/mlp_19_06/"


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)
# parser.add_argument("--layer", type=int)
# parser.add_argument("--width_loss_weight", type=str, default="0")
parser.add_argument("--width_loss_type", type=str, default="L1")
parser.add_argument("--clip_conv_loss_type", type=str, default="L2")

parser.add_argument("--mlp", type=int, default=0)
parser.add_argument("--width_optim", type=int, default=1)
parser.add_argument("--optimize_points", type=int, default=0)
parser.add_argument("--num_strokes", type=int, default=32)

parser.add_argument("--clip_conv_loss", type=int, default=1)
parser.add_argument("--clip_mask_loss", type=int, default=0)
parser.add_argument("--loss_mask", type=str, default="none")
parser.add_argument("--dilated_mask", type=int, default=0)
parser.add_argument("--mask_attention", type=int, default=0)
parser.add_argument("--mask_cls", type=str, default="none")

args = parser.parse_args()
num_strokes = args.num_strokes
loss_mask = args.loss_mask
mlp_train = args.mlp
clip_mask_loss = args.clip_mask_loss
clip_conv_loss = args.clip_conv_loss
model = "ViT-B/32"
mask_object_attention = 0
lr = 1.0
if mlp_train:
    lr = 1e-4


use_wandb = 1
wandb_project_name = "width_05_07"
layers = [3,4,5,6,7,8,9,10,11]
weights = [30, 16, 8, 4]
if args.width_loss_type == "L1":
    weights = [0.0001, 0.0004, 0.0008, 0.001]
num_iter = 151
num_sketches = 1

# use_wandb = 1
# wandb_project_name = "mlp_19_06"
# layers = [2,3,4,5,6,7,8,9,10,11]
# num_iter = 1001
# num_sketches = 2



for j, layer in enumerate(layers):
    layer_opt = layer
    clip_conv_layer_weights_int = [0 for k in range(12)]
    clip_conv_layer_weights_int[layer_opt] = 1
    clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
    clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

    file_ = f"{path_to_files}/house_layer{layer}.png"
    start_l = time.time()
    path_res = f"{path_res_pref}/Cos_mlp_ViT_l{layer}_32s_{source_im_name}/"
    svg_filename = get_svg_file(path_res)
    path_svg = f"{path_res}/{svg_filename}"

    for i, w in enumerate(weights):
        start_w = time.time()
        test_name_pref = ""
        test_name_pref += f"_l{layer_opt}{args.clip_conv_loss_type}_"
        test_name = f"l{layer}__width_{args.width_loss_type}{w}_{test_name_pref}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}"
        print("**** test_name ****")
        print(test_name)

        if i == 0:
            mlp_width_weights_path = "none"
        else:
            mlp_width_weights_path = f"{output_pref}/l{layer}__width_{args.width_loss_type}{weights[i-1]}_{test_name_pref}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}/width_mlp.pt"
            print("**** mlp_width_weights_path ****")
            print(mlp_width_weights_path)
            assert os.path.exists(mlp_width_weights_path)
        

        sp.run(["python", 
                "scripts/width_opt/run_sketch.py", 
                "--target_file", file_,
                "--output_pref", output_pref,
                "--num_strokes", str(num_strokes),
                "--num_iter", str(num_iter),
                "--test_name", test_name,
                "--num_sketches", str(num_sketches),
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
                "--width_loss_weight", str(w),
                "--mask_attention", str(args.mask_attention),
                "--optimize_points", str(args.optimize_points),
                "--width_loss_type", str(args.width_loss_type),
                "--path_svg", path_svg,
                "--mlp_width_weights_path", mlp_width_weights_path])
        print("=" * 50)
        print("time per w: ", time.time() - start_w)
        print("=" * 50)

    print("=" * 50)
    print("time per layer: ", time.time() - start_l)
    print("=" * 50)