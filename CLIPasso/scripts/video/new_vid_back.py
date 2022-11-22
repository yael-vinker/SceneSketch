import os
import argparse
import subprocess as sp
from shutil import copyfile



# CUDA_VISIBLE_DEVICES=5 python scripts/video/new_vid_back.py --dir_name "horse_vid_back" --num_strokes 64

parser = argparse.ArgumentParser()
parser.add_argument("--dir_name", type=str, default="horse_vid")
parser.add_argument("--num_strokes", type=int, default=16)
args = parser.parse_args()
dir_name = args.dir_name

# dir_name = "horse_vid"

# ==========================
# ====== first frame =======
# ==========================
num_strokes = args.num_strokes
num_sketches = 2
num_iter = 2001
use_wandb = 0
wandb_project_name = "none"
images = ["000_mask.png"]
# images = ["face.png", "semi-complex.jpeg", "complex-scene-crop.png", "flamingo.png", "horse.png"]

output_pref = f"/home/vinker/dev/video_clipasso/experiements/video_31_07/{dir_name}/"
loss_mask = "none"
mask_object_attention = 0
# ===================

path_to_files = f"/home/vinker/dev/video_clipasso/video_frames/{dir_name}/"
layer = 4
model = "ViT-B/32"
clip_conv_layer_weights_int = [0 for k in range(12)]
clip_conv_layer_weights_int[layer] = 1
clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)
clip_fc_loss_weight = 0
mlp_train = 0
lr = 1.0


for im_name in images:
    file_ = f"{path_to_files}/{im_name}"
    # test_name = f"{dir_name}_l{layer}_{os.path.splitext(os.path.basename(file_))[0]}_{num_strokes}s"
    # print(test_name)
    # sp.run(["python", 
    #         "scripts/video/run_sketch.py", 
    #         "--target_file", file_,
    #         "--output_pref", output_pref,
    #         "--num_strokes", str(num_strokes),
    #         "--num_iter", str(num_iter),
    #         "--test_name", test_name,
    #         "--num_sketches", str(num_sketches),
    #         "--mask_object", "0",
    #         "--fix_scale", "0",
    #         "--clip_fc_loss_weight", str(clip_fc_loss_weight),
    #         "--clip_conv_layer_weights", clip_conv_layer_weights,
    #         "--clip_model_name", model,
    #         "--use_wandb", str(use_wandb),
    #         "--wandb_project_name", wandb_project_name,
    #         "--loss_mask",loss_mask,
    #         "--mask_object_attention", str(mask_object_attention),
    #         "--mlp_train", str(mlp_train),
    #         "--lr", str(lr)])



# frames = range(0, 100)
frames = range(10, 254)
num_iter = 101
for i in frames:
    print(i)
    # f"{dir_name}_l{layer}_{os.path.splitext(os.path.basename(file_))[0]}_{num_strokes}s"
    path_svg_ = f"{output_pref}/{dir_name}_l{layer}_{i:03d}_mask_{num_strokes}s"
    files = os.listdir(path_svg_)
    files = [f for f in files if "svg" in f]
    path_svg = f"{path_svg_}/{files[0]}"
    # print(path_svg)
    # print(os.path.isfile(path_svg))
    j = i + 1
    im_name = f"{j:03d}_mask.png"
    print(im_name)
    file_ = f"{path_to_files}/{im_name}"
    test_name = f"{dir_name}_l{layer}_{os.path.splitext(os.path.basename(file_))[0]}_{num_strokes}s"
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
        "--lr", str(lr),
        "--path_svg", path_svg])

# parser.add_argument("--loss_mask", type=str, default="none", 
#                         help="mask the object during training, can be none|back|for, if you want to mask out the background choose back")
# parser.add_argument("--mask_object_attention", type=int, default=0)

# loss_mask = for