import os
import argparse
import subprocess as sp
from shutil import copyfile
import time

# ===========================================
# ======= fidelity axis generation ==========
# ===========================================
# This script is to run the baseline sketching method with the ViT clip model.
# We use this to create the first row in the abstraciton matrix.
# You can use this to create both the objects and background. 
# The default parameters are set for the background case.
# Example of a running command:
# CUDA_VISIBLE_DEVICES=1 python scripts/generate_fidelity_levels.py --im_name "man_flowers" --layer_opt 7 --object_or_background "object" --resize_obj 1
# CUDA_VISIBLE_DEVICES=3 python scripts/generate_fidelity_levels.py --im_name "man_flowers" --layer_opt 7 --object_or_background "background"
# CUDA_VISIBLE_DEVICES=2 python scripts/generate_fidelity_levels.py --im_name "man_flowers" --layer_opt 8 --object_or_background "background"


parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer_opt", type=int, default=4)
parser.add_argument("--object_or_background", type=str, default="background")
parser.add_argument("--resize_obj", type=int, default=0)
parser.add_argument("--num_iter", type=int, default=1501)
parser.add_argument("--num_sketches", type=int, default=2)
parser.add_argument("--fg_bg_separation", type=int, default=1)
parser.add_argument("--num_strokes", type=int, default=64,
                    help="number of strokes used to generate the sketch, this defines the level of abstraction.")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--process_id", type=int, default=0)
args = parser.parse_args()


path_to_input_images = "./target_images" # where the input images are located
output_pref = f"./results_sketches/{args.im_name}/runs"

# if you run on objects, this need to be changed:
im_filename = f"{args.im_name}_mask.png"
folder_ = "background"
gradnorm = 0
mask_object = 0
if args.object_or_background == "object":
    if args.layer_opt != 4:
        gradnorm = 1
    mask_object = 1
    im_filename = f"{args.im_name}.png"
    folder_ = "scene"

# ===================
# ====== demo =======
# ===================
num_strokes = args.num_strokes
num_sketches = args.num_sketches
num_iter = args.num_iter
# ===================

# set the weights for each layer
clip_conv_layer_weights_int = [0 for k in range(12)]
if args.object_or_background == "object":
    # we combine two layers if we train on objects
    clip_conv_layer_weights_int[4] = 0.5
clip_conv_layer_weights_int[args.layer_opt] = 1
clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)
if not args.fg_bg_separation:
    im_filename = f"{args.im_name}.png"
    folder_ = "scene"
    test_name = f"l{args.layer_opt}_{os.path.splitext(im_filename)[0]}"
else:
    test_name = f"{args.object_or_background}_l{args.layer_opt}_{os.path.splitext(im_filename)[0]}"
file_ = f"{path_to_input_images}/{folder_}/{im_filename}"
    
print(test_name)
start_time = time.time()
sp.run(["python", 
        "scripts/run_sketch.py", 
        "--target_file", file_,
        "--output_pref", output_pref,
        "--num_iter", str(num_iter),
        "--test_name", test_name,
        "--num_sketches", str(num_sketches),
        "--mask_object", str(mask_object),
        "--clip_conv_layer_weights", clip_conv_layer_weights,
        "--gradnorm", str(gradnorm),
        "--resize_obj", str(args.resize_obj),
        "--eval_interval", str(50),
        "--num_strokes", str(num_strokes),
        "--gpu_id",str(args.gpu_id),
        "--process_id",str(args.process_id),
        "--min_eval_iter", str(400)])
total_time = time.time() - start_time
print(f"Time taken to generate fidelity for {test_name}: [{total_time:.2f}] seconds")
