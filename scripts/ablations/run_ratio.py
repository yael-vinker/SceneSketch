import os
import argparse
import subprocess as sp
from shutil import copyfile
import time
import scripts_utils

# ====================================================
# ========= visual simplification script =============
# ====================================================
# This script is to run the visual simplification (ratio based).
# The script is suitable for objects and background (specified under "object_or_background")
# The script recieves the name of the desired image, and the layer of interest.
# The set of ratios are automatically calculated as part of this script.
# Example of a running command:
# CUDA_VISIBLE_DEVICES=2 python scripts/ablations/run_ratio.py --im_name "man_flowers" --layer_opt 11 --object_or_background "background" --min_div 0.9


parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer_opt", type=int, default=4)
parser.add_argument("--object_or_background", type=str, default="background")
parser.add_argument("--min_div", type=float, default=0)
parser.add_argument("--resize_obj", type=int, default=0)
parser.add_argument("--num_strokes", type=int, default=64)
parser.add_argument("--ablation_name", type=str, default="ablation")
parser.add_argument("--optimize_points", type=int, default=1)
parser.add_argument("--ratios_str", type=str, default="")


args = parser.parse_args()

# =============================
# ====== default params =======
# =============================
path_to_files = "/home/vinker/dev/input_images/background_sketching/"  # where the input images are located
output_pref = f"/home/vinker/dev/background_project/experiements/{args.ablation_name}" # path to output the results
path_res_pref = "/home/vinker/dev/background_project/experiements/all_together_09_09" # path to take semantic trained models from
filename = f"{args.im_name}_mask.png" if args.object_or_background == "background" else f"{args.im_name}.jpg"
# filename = f"{args.im_name}.jpg"
file_ = f"{path_to_files}/{filename}"

res_filename = f"{args.object_or_background}_l{args.layer_opt}_{args.num_strokes}s_{os.path.splitext(filename)[0]}"
if args.resize_obj:
    res_filename = f"{res_filename}_resize{args.resize_obj}"

model = "ViT-B/32"
num_strokes=args.num_strokes
# change this to 1, it's 0 for Lsparse ablation
gradnorm = 1
width_optim = 1
load_points_opt_weights = 1
width_weight = 1
mask_object = 0
if args.object_or_background == "object":
    mask_object = 1

# set the weights
clip_conv_layer_weights_int = [0 for k in range(12)]
if args.object_or_background == "object":
    clip_conv_layer_weights_int[4] = 0.5
clip_conv_layer_weights_int[args.layer_opt] = 1
clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)
mlp_points_weights_path = "none"


# =============================
# ======= demo params =========
# =============================
# use_wandb = 0
# wandb_project_name = "none"
# num_iter = 51
# save_interval = 10
# num_sketches = 2
# output_pref = f"/home/vinker/dev/background_project/experiements/{args.ablation_name}_demo/"
# =============================

# =============================
# =========== real ============
# =============================
use_wandb = 0
wandb_project_name = args.ablation_name
num_iter = 501
save_interval = 100
num_sketches = 2
# =============================


# load the semantic MLP and its input
path_res = f"{path_res_pref}/{res_filename}/"
if not os.path.isdir(path_res):
    res_filename = f"{res_filename}_resize{args.resize_obj}"
    path_res = f"{path_res_pref}/{res_filename}/"
svg_filename = scripts_utils.get_svg_file(path_res)    
best_svg_folder = svg_filename[:-9]
path_svg = f"{path_res}/{best_svg_folder}/svg_logs/init_svg.svg"

mlp_points_weights_path = f"{path_res}/{best_svg_folder}/points_mlp.pt"
assert os.path.exists(mlp_points_weights_path)
print(mlp_points_weights_path)


# get the ratios for im_name at the given layer_opt
if args.ratios_str == "":
    ratios_str = scripts_utils.get_ratios_dict3(path_res_pref, folder_name_l=res_filename, 
                                            layer=args.layer_opt, im_name=args.im_name, 
                                            object_or_background=args.object_or_background,
                                            step_size_l=args.min_div)
else:
    ratios_str = args.ratios_str
                             
ratios = [float(item) for item in ratios_str.split(',')]
print(ratios)


# train for each ratio
for i, ratio in enumerate(ratios):
    # i = j + 3
    print(i)
    start_w = time.time()
    test_name_pref = f"l{args.layer_opt}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}_{args.min_div}"#_MLPloc{args.optimize_points}"
    if args.resize_obj:
        test_name_pref += f"_resize{args.resize_obj}"
    test_name = f"ratio{ratio}_{test_name_pref}"
    print("**** test_name ****")
    print(test_name)
    if i == 0:
        # in this case we use the semantic mlp (first row) and we don't want its optimizer
        mlp_width_weights_path = "none"
        load_points_opt_weights = 0
    else:
        mlp_width_weights_path = f"{output_pref}/ratio{ratios[i-1]}_{test_name_pref}/width_mlp_n.pt"
        print("**** mlp_width_weights_path ****")
        print(mlp_width_weights_path)
        assert os.path.exists(mlp_width_weights_path)

        mlp_points_weights_path = f"{output_pref}/ratio{ratios[i-1]}_{test_name_pref}/points_mlp_n.pt"
        print("**** mlp_points_weights_path ****")
        print(mlp_points_weights_path)
        assert os.path.exists(mlp_points_weights_path)

        load_points_opt_weights = 1

    sp.run(["python", 
            "scripts/ablations/run_sketch.py", 
            "--target_file", file_,
            "--output_pref", output_pref,
            "--num_strokes", str(num_strokes),
            "--num_iter", str(num_iter),
            "--test_name", test_name,
            "--num_sketches", str(num_sketches),
            "--clip_conv_layer_weights", clip_conv_layer_weights,
            "--clip_model_name", model,
            "--use_wandb", str(use_wandb),
            "--wandb_project_name", str(wandb_project_name),
            "--width_optim", str(width_optim),
            "--width_loss_weight", str(width_weight),
            "--path_svg", path_svg,
            "--mlp_width_weights_path", mlp_width_weights_path,
            "--save_interval", str(save_interval),
            "--mlp_points_weights_path", mlp_points_weights_path,
            "--gradnorm", str(gradnorm),
            "--load_points_opt_weights", str(load_points_opt_weights),
            "--width_weights_lst", ratios_str,
            "--ratio_loss", str(ratio),
            "--mask_object", str(mask_object),
            "--resize_obj", str(args.resize_obj),
            "--optimize_points", str(args.optimize_points)])
    print("=" * 50)
    print("time per w: ", time.time() - start_w)
    print("=" * 50)
