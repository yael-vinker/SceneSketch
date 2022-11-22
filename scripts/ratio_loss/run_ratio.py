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
# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio.py --im_name "semi-complex" --layer_opt 11 --run_ratio 1 --weihts_str "5.02,3.748,3.148,2.02,1.119,0.79,0.54,0.32"
# CUDA_VISIBLE_DEVICES=1 python scripts/ratio_loss/run_ratio.py --im_name "semi-complex" --layer_opt 4 --run_ratio 1 --weihts_str "50.2,37.48,31.48,20.2,11.19,7.9,5.4,3.2"


# 
  #1.72171655  1.02027647

# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio.py --im_name "man_flowers" --layer_opt 11 --run_ratio 1 --weihts_str "5.02,3.748,3.148,2.02,1.119,0.79,0.54,0.32"
# CUDA_VISIBLE_DEVICES=5 python scripts/ratio_loss/run_ratio.py --im_name "man_flowers" --layer_opt 4 --run_ratio 1 --weihts_str "50.2,37.48,31.48,20.2,11.19,7.9,5.4,3.2"

# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio.py --im_name "woman_city" --layer_opt 11 --run_ratio 1 --weihts_str "5.02,3.748,3.148,2.02,1.119,0.79,0.54,0.32"
#from here:
# CUDA_VISIBLE_DEVICES=5 python scripts/ratio_loss/run_ratio.py --im_name "woman_city" --layer_opt 4 --run_ratio 1 --weihts_str "50.2,37.48,31.48,20.2,11.19,7.9,5.4,3.2"

# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio.py --im_name "van" --layer_opt 11 --run_ratio 1 --weihts_str "5.02,3.748,3.148,2.02,1.119,0.79,0.54,0.32"
# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio.py --im_name "van" --layer_opt 4 --run_ratio 1 --weihts_str "50.2,37.48,31.48,20.2,11.19,7.9,5.4,3.2"

# =======
# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio.py --im_name "semi-complex" --layer_opt 8 --run_ratio 1 --weihts_str "16,11.95,10.03,6.44,3.567,2.518,1.721,1.02"
# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio.py --im_name "man_flowers" --layer_opt 8 --run_ratio 1 --weihts_str "16,11.95,10.03,6.44,3.567,2.518,1.721,1.02"
# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio.py --im_name "woman_city" --layer_opt 8 --run_ratio 1 --weihts_str "16,11.95,10.03,6.44,3.567,2.518,1.721,1.02"
# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio.py --im_name "van" --layer_opt 8 --run_ratio 1 --weihts_str "16,11.95,10.03,6.44,3.567,2.518,1.721,1.02"


# =======
# CUDA_VISIBLE_DEVICES=5 python scripts/ratio_loss/run_ratio.py --im_name "semi-complex" --layer_opt 6 --run_ratio 1 --weihts_str "38.95,29.08,24.427,15.674,8.682,6.13,4.19,2.483"
# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio.py --im_name "man_flowers" --layer_opt 6 --run_ratio 1 --weihts_str "38.95,29.08,24.427,15.674,8.682,6.13,4.19,2.483"
# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio.py --im_name "woman_city" --layer_opt 8 --run_ratio 1 --weihts_str "16,11.95,10.03,6.44,3.567,2.518,1.721,1.02"
# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio.py --im_name "van" --layer_opt 8 --run_ratio 1 --weihts_str "16,11.95,10.03,6.44,3.567,2.518,1.721,1.02"



# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio.py --im_name "semi-complex" --layer_opt 11 --run_ratio 1 --weihts_str "5.02,3.748,3.148,2.02,1.119,0.79,0.54,0.32"
# CUDA_VISIBLE_DEVICES=5 python scripts/ratio_loss/run_ratio.py --im_name "semi-complex" --layer_opt 11 --run_ratio 1 --weihts_str "5.02,3.748,3.148,2.02,1.119,0.79,0.54,0.32"

# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio.py --im_name "semi-complex" --layer_opt 4 --run_ratio 1 --weihts_str "50.2,37.48,31.48,20.2,11.19,7.9,5.4"
# CUDA_VISIBLE_DEVICES=1 python scripts/ratio_loss/run_ratio.py --im_name "semi-complex" --layer_opt 4 --run_ratio 1 --weihts_str "49.5,39.8,30,17.2,14.7,11.3,3,2.8"
# CUDA_VISIBLE_DEVICES=1 python scripts/ratio_loss/run_ratio.py --im_name "semi-complex" --layer_opt 4 --run_ratio 1 --weihts_str "50.2,37.48,31.48,20.2,11.19,7.9,5.4,3.2",2.5"

# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio.py --im_name "van" --layer_opt 11 --run_ratio 1
# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio.py --im_name "woman_city" --layer_opt 11 --run_ratio 1 
# CUDA_VISIBLE_DEVICES=6 python scripts/ratio_loss/run_ratio.py --im_name "semi-complex" --layer_opt 11 --run_ratio 1 --gradnorm 0
# CUDA_VISIBLE_DEVICES=0 python scripts/ratio_loss/run_ratio.py --im_name "man_flowers" --layer_opt 11 --run_ratio 1 

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer_opt", type=int, default=4)
parser.add_argument("--run_ratio", type=int, default=0)
parser.add_argument("--weihts_str", type=str, default="5.2,3.98,3.0,1.72,1.47,1.13,0.3,0.1")

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
# wandb_project_name = "ratio_loss_30_08"
wandb_project_name = "ratio_loss_objects_09_04"
width_weight = 1
# weihts_str = "4.95,3.97,3.62,1.716,1.46,1.128,0.3,0.28"
# weihts_str = "0.28,0.25,0.23,0.2"
weihts_str = args.weihts_str#"5.2,3.98,3.0,1.72,1.47,1.1,0.2,0.1"
ratios = [float(item) for item in weihts_str.split(',')]
num_iter = 501
save_interval = 100
num_sketches = 2
# output_pref = f"/home/vinker/dev/background_project/experiements/ratio_loss_30_08/"
output_pref = f"/home/vinker/dev/background_project/experiements/ratio_loss_objects_09_04/"

# # =========================
# # ====== debug =======
# # =========================
# use_wandb = 0
# wandb_project_name = "none"
# width_weight = 1
# # weihts_str = "4.95,3.98,3.62,1.72,1.47,1.13,0.3,0.28"
# # 4.95, 3.97, 3.62, 1.716, 1.46, 1.128, 0.3, 0.29
# weihts_str = "4.95,3.98"
# ratios = [float(item) for item in weihts_str.split(',')]
# num_iter = 51
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
clip_conv_layer_weights_int = [0 for k in range(12)]
clip_conv_layer_weights_int[layer_opt] = 1
clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)
mlp_points_weights_path = "none"


# =========================
# ==== per run params =====
# =========================
path_res_pref = "/home/vinker/dev/background_project/experiements/big_test_07_27/"
filename = f"{args.im_name}_mask.png"
res_filename = f"ViT_l{args.layer_opt}_64s_{args.im_name}_mask"


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
best_svg_folder = svg_filename[:-9]
path_svg = f"{path_res}/{best_svg_folder}/svg_logs/svg_iter0.svg" # change to svg_init.svg
if args.mlp:
    mlp_points_weights_path = f"{path_res}/{best_svg_folder}/points_mlp.pt"
    assert os.path.exists(mlp_points_weights_path)

for i, ratio in enumerate(ratios):
# for j, ratio in enumerate(ratios[-1:]):
    # i = len(ratios) - 1 #j + 1
    print(i, ratio)
    start_w = time.time()
    test_name_pref = f"c_width{width_weight}_lr{width_lr}"#"b_prev-w"
    if args.gradnorm:
        test_name_pref += f"_gradnorm"
    test_name_pref += f"_clip_l{layer_opt}{args.clip_conv_loss_type}_"
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
            "--ratio_loss", str(ratio)])
    print("=" * 50)
    print("time per w: ", time.time() - start_w)
    print("=" * 50)
