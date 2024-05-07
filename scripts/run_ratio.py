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
# CUDA_VISIBLE_DEVICES=7 python scripts/run_ratio.py --im_name "man_flowers" --layer_opt 8 --object_or_background "background" --min_div 0.5
# CUDA_VISIBLE_DEVICES=6 python scripts/run_ratio.py --im_name "man_flowers" --layer_opt 8 --object_or_background "object" --min_div 0.5 --resize 1
# ====================================================

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer_opt", type=int, default=4)
parser.add_argument("--simp_level", type=int, default=0)
parser.add_argument("--object_or_background", type=str, default="background")
parser.add_argument("--min_div", type=float, default=0)
parser.add_argument("--resize_obj", type=int, default=0)
parser.add_argument("--num_sketches", type=int, default=2)
parser.add_argument("--fg_bg_separation", type=int, default=1)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--process_id", type=int, default=0)
parser.add_argument("--num_strokes", type=int, default=64,
                    help="number of strokes used to generate the sketch, this defines the level of abstraction.")

args = parser.parse_args()

# =============================
# ====== default params =======
# =============================
path_to_files = "./target_images"  # where the input images are located
output_pref = f"./results_sketches/{args.im_name}/runs" # path to output the results
path_res_pref = f"./results_sketches/{args.im_name}/runs" # path to take semantic trained models from
filename = f"{args.im_name}_mask.png" if args.object_or_background == "background" else f"{args.im_name}.png"
folder_ = "background" if args.object_or_background == "background" else "scene"

if not args.fg_bg_separation:
    filename = f"{args.im_name}.png"
    folder_ = "scene"
    res_filename = f"l{args.layer_opt}_{os.path.splitext(filename)[0]}"
else:
    res_filename = f"{args.object_or_background}_l{args.layer_opt}_{os.path.splitext(filename)[0]}"

file_ = f"{path_to_files}/{folder_}/{filename}"


num_strokes=args.num_strokes
gradnorm = 1
mask_object = 0
if args.object_or_background == "object":
    mask_object = 1

# =============================
# =========== real ============
# =============================
num_iter = 401
num_sketches = args.num_sketches
# =============================


# set the weights
clip_conv_layer_weights_int = [0 for k in range(12)]
if args.object_or_background == "object":
    clip_conv_layer_weights_int[4] = 0.5
clip_conv_layer_weights_int[args.layer_opt] = 1
clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)



# load the semantic MLP and its input
path_res = f"{path_res_pref}/{res_filename}/"
path_svg = f"{path_res}/init.svg"
mlp_points_weights_path = f"{path_res}/points_mlp.pt"
assert os.path.exists(mlp_points_weights_path)


# get the ratios for im_name at the given layer_opt
ratios_str = scripts_utils.get_ratios_dict(path_res_pref, folder_name_l=res_filename, 
                                            layer=args.layer_opt, im_name=args.im_name, 
                                            object_or_background=args.object_or_background,
                                            step_size_l=args.min_div)                         
ratios = [float(item) for item in ratios_str.split(',')]
print(ratios)


# train for each ratio
for i, ratio in enumerate(ratios):
    if args.simp_level != 0 and args.simp_level != i: # in case we want to skip to a specific simplicity level
        continue 
    start = time.time()
    if args.fg_bg_separation:
        test_name_pref = f"l{args.layer_opt}_{os.path.splitext(os.path.basename(file_))[0]}_{args.min_div}"
    else:
        test_name_pref = f"l{args.layer_opt}_{os.path.splitext(os.path.basename(file_))[0]}_scene_{args.min_div}"
    test_name = f"ratio{ratio}_{test_name_pref}"
    if not os.path.exists(f"{output_pref}/{test_name}/width_mlp.pt"):
        print("**** test_name ****")
        print(test_name)
        if i == 0 or (args.simp_level != 0 and args.simp_level == i):
            # in this case we use the semantic mlp (first row) and we don't want its optimizer
            mlp_width_weights_path = "none"
            load_points_opt_weights = 0
        else:
            mlp_width_weights_path = f"{output_pref}/ratio{ratios[i-1]}_{test_name_pref}/width_mlp.pt"
            print("**** mlp_width_weights_path ****")
            print(mlp_width_weights_path)
            assert os.path.exists(mlp_width_weights_path)

            mlp_points_weights_path = f"{output_pref}/ratio{ratios[i-1]}_{test_name_pref}/points_mlp.pt"
            print("**** mlp_points_weights_path ****")
            print(mlp_points_weights_path)
            assert os.path.exists(mlp_points_weights_path)

            load_points_opt_weights = 1

        sp.run(["python", 
                "scripts/run_sketch.py", 
                "--target_file", file_,
                "--output_pref", output_pref,
                "--num_strokes", str(num_strokes),
                "--num_iter", str(num_iter),
                "--test_name", test_name,
                "--num_sketches", str(num_sketches),
                "--clip_conv_layer_weights", clip_conv_layer_weights,
                "--width_optim", str(1),
                "--width_loss_weight", str(1),
                "--path_svg", path_svg,
                "--mlp_width_weights_path", mlp_width_weights_path,
                "--mlp_points_weights_path", mlp_points_weights_path,
                "--gradnorm", str(gradnorm),
                "--load_points_opt_weights", str(load_points_opt_weights),
                "--width_weights_lst", ratios_str,
                "--ratio_loss", str(ratio),
                "--mask_object", str(mask_object),
                "--gpu_id",str(args.gpu_id),
                "--process_id", str(args.process_id),
                "--resize_obj", str(args.resize_obj)])
        print("=" * 50)
        print(f"time per w {test_name}: ", time.time() - start)
        print("=" * 50)
