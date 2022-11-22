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
# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "semi-complex" --layer_opt 11 --run_ratio 1 --weihts_str "5.02,3.748,3.148,2.02,1.119,0.79,0.54,0.32"
# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "semi-complex" --layer_opt 4 --run_ratio 1 --weihts_str "50.2,37.48,31.48,20.2,11.19,7.9,5.4,3.2"
# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "semi-complex" --layer_opt 4 --run_ratio 1 --weihts_str "50.2,37.48,31.48,20.2,11.19,7.9,5.4,3.2" --load_points_opt_weights 1
# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "semi-complex" --layer_opt 8 --run_ratio 1 --weihts_str "16,11.95,10.03,6.44,3.567,2.518,1.721,1.02" --load_points_opt_weights 1

# CUDA_VISIBLE_DEVICES=5 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "semi-complex" --layer_opt 4 --run_ratio 1 --weihts_str "60,50,37.5,31.5,20,13,8,5.5,3.5" --load_points_opt_weights 1
# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "semi-complex" --layer_opt 4 --run_ratio 1 --weihts_str "48.038,35.878,30.114,19.335,10.709,7.56,5.167,3.062" --load_points_opt_weights 1


# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "woman_city" --layer_opt 4 --run_ratio 1 --weihts_str "51.966,38.812,32.576,20.917,11.585,8.178,5.59,3.313" --load_points_opt_weights 1
# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "woman_city" --layer_opt 6 --run_ratio 1 --weihts_str "41.826,31.238,26.219,16.835,9.324,6.582,4.499,2.666" --load_points_opt_weights 1
# CUDA_VISIBLE_DEVICES=5 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "semi-complex" --layer_opt 6 --run_ratio 1 --weihts_str "36.39,27.179,22.812,14.647,8.113,5.727,3.914,2.32" --load_points_opt_weights 1

# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "woman_city" --layer_opt 7 --run_ratio 1 --weihts_str "29.18,21.794,18.292,11.745,6.505,4.592,3.139,1.86" --load_points_opt_weights 1
# CUDA_VISIBLE_DEVICES=3 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "man_flowers" --layer_opt 7 --run_ratio 1 --weihts_str "29.295,21.879,18.364,11.791,6.531,4.61,3.151,1.868" --load_points_opt_weights 1
# CUDA_VISIBLE_DEVICES=5 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "semi-complex" --layer_opt 7 --run_ratio 1 --weihts_str "29.99,22.399,18.8,12.071,6.686,4.72,3.226,1.912" --load_points_opt_weights 1

# CUDA_VISIBLE_DEVICES=1 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "woman_city" --layer_opt 11 --run_ratio 1 --weihts_str "4.826,3.604,3.025,1.942,1.075,0.759,0.519,0.307" --load_points_opt_weights 1
# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "semi-complex" --layer_opt 11 --run_ratio 1 --weihts_str "5.818,4.345,3.647,2.341,1.297,0.915,0.625,0.37" --load_points_opt_weights 1
# CUDA_VISIBLE_DEVICES=5 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "man_flowers" --layer_opt 11 --run_ratio 1 --weihts_str "5.119,3.823,3.209,2.06,1.141,0.805,0.55,0.326" --load_points_opt_weights 1


# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "semi-complex" --layer_opt 5 --run_ratio 1 --weihts_str "40.544,30.282,25.416,16.319,9.039,6.381,4.361,2.585" --load_points_opt_weights 1
# CUDA_VISIBLE_DEVICES=2 python scripts/ratio_loss/run_ratio_load_mlp_points.py --im_name "man_flowers" --layer_opt 5 --run_ratio 1 --weihts_str "45.042,33.641,28.236,18.13,10.042,7.089,4.845,2.871" --load_points_opt_weights 1



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
path_svg = f"{path_res}/{best_svg_folder}/svg_logs/svg_iter0.svg"
if args.mlp:
    mlp_points_weights_path = f"{path_res}/{best_svg_folder}/points_mlp.pt"
    assert os.path.exists(mlp_points_weights_path)

for i, ratio in enumerate(ratios):
# for j, ratio in enumerate(ratios[-1:]):
    # i = len(ratios) - 1 #j + 1
    print(i, ratio)
    start_w = time.time()
    test_name_pref = f"points-mlp1_opt{args.load_points_opt_weights}"#"b_prev-w"
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

        mlp_points_weights_path = f"{output_pref}/ratio{ratios[i-1]}_{test_name_pref}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}/points_mlp_n.pt"
        print("**** mlp_points_weights_path ****")
        print(mlp_points_weights_path)
        assert os.path.exists(mlp_points_weights_path)

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
