import os
import argparse
import subprocess as sp
from shutil import copyfile
import glob

# =====================================================
# ========= baseline vit script for video =============
# =====================================================
# Runs the first frame in the given folder
# Example of a running command:
# CUDA_VISIBLE_DEVICES=1 python video_clipasso/video_clipasso_scripts/run_video_sketch.py --im_name "horse_vid" --layer_opt 4

path_to_files = "/home/vinker/dev/input_images/videos_input/frames/" # should cnotain the frames from the preprocess stage
output_pref_ = f"/home/vinker/dev/background_project/experiements/video_res"

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer_opt", type=int, default=4)
args = parser.parse_args()

# ===================
# ====== real =======
# ===================
use_wandb = 0
wandb_project_name = "none"
im_filename = f"{args.im_name}.jpg"
if args.object_or_background == "background":
    im_filename = f"{args.im_name}_mask.png"
# ===================
path_to_objects = f"{path_to_files}/{args.im_name}/object/"
frames_lst = sorted(os.listdir(path_to_objects))
first_frame = frames_lst[0]
first_frame_path = f"{path_to_objects}/{first_frame}"
output_pref = f"{output_pref_}/{args.im_name}/object"
if not os.path.exists(output_pref):
    os.makedirs(output_pref)
print(f"Your results will be saved to [{output_pref}]")

# ===================
# ====== demo =======
# ===================
# output_pref_ = f"/home/vinker/dev/background_project/experiements/obj_vit_demo"
# num_strokes = 64
# num_sketches = 2
# num_iter = 51
# use_wandb = 0
# wandb_project_name = "none"
# ===================

model = "ViT-B/32" # for background it's vit
num_strokes = 64
num_sketches = 2

object_or_background="object"
resize_obj=0

# if you run on objects, this need to be changed:
gradnorm = 0
mask_object = 0
if object_or_background == "object":
    mask_object = 1
    if args.layer_opt != 4:
        gradnorm = 1
    # change the images as well
# set the weights for each layer
clip_conv_layer_weights_int = [0 for k in range(12)]
if object_or_background == "object":
    # we combine two layers if we train on objects
    clip_conv_layer_weights_int[4] = 0.5
clip_conv_layer_weights_int[args.layer_opt] = 1
clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

# =============================================
# =========== run the first frame =============
# =============================================
num_iter = 1501
test_name = f"l{args.layer_opt}_{os.path.splitext(first_frame)[0]}"
sp.run(["python", 
        "video_clipasso/video_clipasso_scripts/run_sketch.py", 
        "--target_file", first_frame_path,
        "--output_pref", output_pref,
        "--num_strokes", str(num_strokes),
        "--num_iter", str(num_iter),
        "--test_name", test_name,
        "--num_sketches", str(num_sketches),
        "--mask_object", str(mask_object),
        "--fix_scale", "0",
        "--clip_conv_layer_weights", clip_conv_layer_weights,
        "--clip_model_name", model,
        "--use_wandb", str(use_wandb),
        "--wandb_project_name", wandb_project_name,
        "--gradnorm", str(gradnorm),
        "--resize_obj", str(resize_obj)])


# =============================================
# =========== run the next frames =============
# =============================================
def get_frmae_results(test_name_):
    # find the init svg path, and the previous MLP_loc
    # the svg path is always the first svg of the first frame, we only load the prev mlp each time
    path_res = f"{output_pref}/{test_name_}/"
    if not os.path.isdir(path_res):
        print(f"path [{path_res}] does not exists!")
    svg_filename = [f for f in os.listdir(path_res) if ".svg" in f][0]
    best_svg_folder = svg_filename[:-9]
    path_svg = f"{path_res}/{best_svg_folder}/svg_logs/init_svg.svg"
    mlp_points_weights_path = f"{path_res}/{best_svg_folder}/points_mlp.pt"
    assert os.path.exists(mlp_points_weights_path)
    return mlp_points_weights_path, path_svg


num_iter = 101
load_points_opt_weights = 1
for f_index, f_name in enumerate(frames_lst[1:]):
    cur_frame_path = f"{path_to_objects}/{f_name}"
    test_name_cur = f"l{args.layer_opt}_{os.path.splitext(f_name)[0]}"
    print(f_index, f_name)
    f_name_prev = frames_lst[f_index]
    test_name_prev = f"l{args.layer_opt}_{os.path.splitext(f_name_prev)[0]}"
    mlp_points_weights_path_prev, path_svg_prev = get_frmae_results(test_name_prev)
    sp.run(["python", 
            "video_clipasso/video_clipasso_scripts/run_sketch.py", 
            "--target_file", cur_frame_path,
            "--output_pref", output_pref,
            "--num_strokes", str(num_strokes),
            "--num_iter", str(num_iter),
            "--test_name", test_name_cur,
            "--num_sketches", str(num_sketches),
            "--clip_conv_layer_weights", clip_conv_layer_weights,
            "--clip_model_name", model,
            "--use_wandb", str(use_wandb),
            "--wandb_project_name", str(wandb_project_name),
            "--path_svg", path_svg_prev,
            "--mlp_points_weights_path", mlp_points_weights_path_prev,
            "--gradnorm", str(gradnorm),
            "--load_points_opt_weights", str(load_points_opt_weights),
            "--mask_object", str(mask_object),
            "--resize_obj", str(resize_obj)])