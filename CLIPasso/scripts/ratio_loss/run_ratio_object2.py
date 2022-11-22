import os
import argparse
import subprocess as sp
from shutil import copyfile
import time

# switch_loss = 0


# 8,9,10,11

# =========================
# ======= commands ========
# =========================
# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio_object.py --im_name "woman_city" --layer_opt 11 --run_ratio 1
# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio_object.py --im_name "woman_city" --layer_opt 3 --run_ratio 1
# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio_object2.py --im_name "woman_city" --layer_opt 11 --run_ratio 1 --weihts_str "11.67,8.716,7.32,4.69,2.6,1.837,1.255,0.744"


parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer_opt", type=int, default=4)
parser.add_argument("--run_ratio", type=int, default=0)
parser.add_argument("--weihts_str", type=str, default="11.67,8.716,7.32,4.69,2.6,1.837,1.255")

parser.add_argument("--width_loss_type", type=str, default="L1")
parser.add_argument("--clip_conv_loss_type", type=str, default="L2")

parser.add_argument("--mlp", type=int, default=1)
parser.add_argument("--width_optim", type=int, default=1)
parser.add_argument("--optimize_points", type=int, default=1)
parser.add_argument("--switch_loss", type=int, default=0)
parser.add_argument("--gradnorm", type=int, default=1)
parser.add_argument("--load_points_opt_weights", type=int, default=0)


args = parser.parse_args()

# =========================
# ====== run params =======
# =========================
use_wandb = 1
wandb_project_name = "ratio_loss_objects_09_04"
width_weight = 1
weihts_str = args.weihts_str
ratios = [float(item) for item in weihts_str.split(',')]
num_iter = 501
save_interval = 100
num_sketches = 2
output_pref = f"/home/vinker/dev/background_project/experiements/ratio_loss_objects_09_04/"

# # # =========================
# # # ====== debug =======
# # # =========================
# use_wandb = 0
# wandb_project_name = "none"
# width_weight = 1
# weihts_str = "5.2"
# ratios = [float(item) for item in weihts_str.split(',')]
# num_iter = 11
# save_interval = 10
# num_sketches = 1
# output_pref = f"/home/vinker/dev/background_project/experiements/ratio_loss_30_08_demo/"


# =========================
# ====== set params =======
# =========================
path_to_files = "/home/vinker/dev/input_images/background_sketching/"
model = "ViT-B/32"
num_strokes=64

layer_opt = args.layer_opt
layer_opt_weight = 0.33
layer_set = 4
layer_set_weight = 1
clip_conv_layer_weights_int = [0 for k in range(12)]
clip_conv_layer_weights_int[layer_opt] = layer_opt_weight
clip_conv_layer_weights_int[layer_set] = layer_set_weight
clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)
mlp_points_weights_path = "none"


# =========================
# ==== per run params =====
# =========================
path_res_pref = "/home/vinker/dev/background_project/experiements/object_yuval_0904/"
filename = f"{args.im_name}.jpg"
res_filename = f"fc-0_ViT_l4-w1_l{args.layer_opt}-w0.33_64s_{args.im_name}"


mlp_train = args.mlp

lr = 1.0
if mlp_train:
    lr = 1e-4
width_lr = 0.00005

def get_svg_file(path):
    files = os.listdir(path)
    files = [f for f in files if ".svg" in f]
    return files[0]


start_l = time.time()
file_ = f"{path_to_files}/{filename}"

# use the mlp, start from init0
path_res = f"{path_res_pref}/{res_filename}/"
svg_filename = get_svg_file(path_res)    
# best_svg_folder = svg_filename[:-9]
path_svg = f"{path_res}/{svg_filename}"
if args.mlp:
    mlp_points_weights_path = f"{path_res}/points_mlp.pt"
    assert os.path.exists(mlp_points_weights_path)


for j, ratio in enumerate(ratios[-1:]):
    i = len(ratios) - 1 #j + 1
# for i, ratio in enumerate(ratios):
    print(i, ratio)
    start_w = time.time()
    test_name_pref = ""
    test_name_pref += f"_ViT_l{layer_set}-{layer_set_weight}_l{layer_opt}-{layer_opt_weight}_"
    test_name = f"ratio{ratio}_{test_name_pref}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}"
    print("**** test_name ****")
    print(test_name)
    if i == 0:
        mlp_width_weights_path = "none"
        load_points_opt_weights = 0
    else:
        mlp_width_weights_path = f"{output_pref}/ratio{ratios[i-1]}_{test_name_pref}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}/width_mlp_n.pt"
        print("**** mlp_width_weights_path ****")
        print(mlp_width_weights_path)
        assert os.path.exists(mlp_width_weights_path)

        # mlp_points_weights_path = f"{output_pref}/width{weights[i-1]}_{test_name_pref}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}/points_mlp_n.pt"
        # print("**** mlp_points_weights_path ****")
        # print(mlp_points_weights_path)
        # assert os.path.exists(mlp_points_weights_path)

        load_points_opt_weights = args.load_points_opt_weights

    sp.run(["python", 
            "scripts/ratio_loss/run_sketch.py", 
            "--target_file", file_,
            "--output_pref", output_pref,
            "--num_strokes", str(num_strokes),
            "--num_iter", str(num_iter),
            "--test_name", test_name,
            "--num_sketches", str(num_sketches),
            "--clip_conv_layer_weights", clip_conv_layer_weights,
            "--clip_model_name", model,
            "--mlp_train", str(mlp_train),
            "--lr", str(lr),
            "--use_wandb", str(use_wandb),
            "--wandb_project_name", str(wandb_project_name),
            "--clip_conv_loss_type", str(args.clip_conv_loss_type),
            "--width_optim", str(args.width_optim),
            "--width_loss_weight", str(width_weight),
            "--optimize_points", str(args.optimize_points),
            "--width_loss_type", str(args.width_loss_type),
            "--path_svg", path_svg,
            "--mlp_width_weights_path", mlp_width_weights_path,
            "--save_interval", str(save_interval),
            "--mlp_points_weights_path", mlp_points_weights_path,
            "--switch_loss", str(args.switch_loss),
            "--gradnorm", str(args.gradnorm),
            "--load_points_opt_weights", str(load_points_opt_weights),
            "--width_weights_lst", weihts_str,
            "--width_lr", str(width_lr),
            "--ratio_loss", str(ratio),
            "--mask_object", str(1)])
    print("=" * 50)
    print("time per w: ", time.time() - start_w)
    print("=" * 50)
