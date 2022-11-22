import os
import argparse
import subprocess as sp
from shutil import copyfile
import time

# =========================
# ======= commands ========
# =========================
# CUDA_VISIBLE_DEVICES=1 python scripts/width_and_points/width_points.py --filename "woman_city.png" --res_filename "RN1_64s_woman_city"

# CUDA_VISIBLE_DEVICES=2 python scripts/width_and_points/width_points.py --filename "van2.png" --res_filename "RN1_64s_van2"
# CUDA_VISIBLE_DEVICES=4 python scripts/width_and_points/width_points.py --filename "lion.png" --res_filename "RN1_64s_lion"
# CUDA_VISIBLE_DEVICES=5 python scripts/width_and_points/width_points.py --filename "man_flowers.png" --res_filename "RN1_64s_man_flowers"
# CUDA_VISIBLE_DEVICES=6 python scripts/width_and_points/width_points.py --filename "bull.png" --res_filename "RN1_64s_bull"

# CUDA_VISIBLE_DEVICES=1 python scripts/width_and_points/width_points.py --filename "panda.png" --res_filename "RN1_64s_panda"
# CUDA_VISIBLE_DEVICES=2 python scripts/width_and_points/width_points.py --filename "zebra.png" --res_filename "RN1_64s_zebra"
# CUDA_VISIBLE_DEVICES=4 python scripts/width_and_points/width_points.py --filename "chicks.png" --res_filename "RN1_64s_chicks"
# CUDA_VISIBLE_DEVICES=5 python scripts/width_and_points/width_points.py --filename "japan.png" --res_filename "RN1_64s_japan"
# CUDA_VISIBLE_DEVICES=6 python scripts/width_and_points/width_points.py --filename "venice.png" --res_filename "RN1_64s_venice"

# CUDA_VISIBLE_DEVICES=6 python scripts/width_and_points/width_points.py --filename "face.png" --res_filename "RN1_32s_mask0_face"
# CUDA_VISIBLE_DEVICES=2 python scripts/width_and_points/width_points.py --filename "horse1.png" --res_filename "RN1_32s_mask0_horse"
# CUDA_VISIBLE_DEVICES=5 python scripts/width_and_points/width_points.py --filename "semi-complex.png" --res_filename "RN1_32s_mask0_semi-complex"

# CUDA_VISIBLE_DEVICES=2 python scripts/width_and_points/width_points.py --filename "face.png" --res_filename "RN1_64s_mask0_face"
# CUDA_VISIBLE_DEVICES=5 python scripts/width_and_points/width_points.py --filename "semi-complex.png" --res_filename "RN1_64s_mask0_semi-complex"
# CUDA_VISIBLE_DEVICES=2 python scripts/width_and_points/width_points.py --filename "flamingo.png" --res_filename "RN1_64s_mask0_flamingo"
# CUDA_VISIBLE_DEVICES=5 python scripts/width_and_points/width_points.py --filename "horse1.png" --res_filename "RN1_64s_mask0_horse"

# =====================
# switch
# =====================
# CUDA_VISIBLE_DEVICES=1 python scripts/width_and_points/width_points.py --filename "flamingo.png" --res_filename "RN1_32s_mask0_flamingo" --switch_loss 10
# CUDA_VISIBLE_DEVICES=6 python scripts/width_and_points/width_points.py --filename "semi-complex.png" --res_filename "RN1_32s_mask0_semi-complex" --switch_loss 10
# CUDA_VISIBLE_DEVICES=1 python scripts/width_and_points/width_points.py --filename "semi-complex.png" --res_filename "RN1_32s_mask0_semi-complex" --switch_loss 20

# CUDA_VISIBLE_DEVICES=4 python scripts/width_and_points/width_points.py --filename "flamingo.png" --res_filename "RN1_32s_mask0_flamingo" --switch_loss 10
# CUDA_VISIBLE_DEVICES=2 python scripts/width_and_points/width_points.py --filename "flamingo.png" --res_filename "RN1_32s_mask0_flamingo" --switch_loss 20


# CUDA_VISIBLE_DEVICES=2 python scripts/width_and_points/width_points.py --filename "horse1.png" --res_filename "RN1_32s_mask0_horse" --switch_loss 10
# CUDA_VISIBLE_DEVICES=7 python scripts/width_and_points/width_points.py --filename "horse1.png" --res_filename "RN1_32s_mask0_horse" --switch_loss 5
# CUDA_VISIBLE_DEVICES=5 python scripts/width_and_points/width_points.py --filename "semi-complex.png" --res_filename "RN1_32s_mask0_semi-complex"


parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="")
parser.add_argument("--res_filename", type=str, default="")

parser.add_argument("--width_loss_type", type=str, default="L1")
parser.add_argument("--clip_conv_loss_type", type=str, default="L2")

parser.add_argument("--mlp", type=int, default=1)
parser.add_argument("--width_optim", type=int, default=1)
parser.add_argument("--optimize_points", type=int, default=1)
parser.add_argument("--switch_loss", type=int, default=5)
args = parser.parse_args()

# =========================
# ====== run params =======
# =========================
use_wandb = 1
# wandb_project_name = "big_test_07_27"
wandb_project_name = "width_and_points_07_22"
weights = [0.0001, 0.0004, 0.001, 0.002, 0.01]
num_iter = 1001
save_interval = 100
num_sketches = 2
output_pref = f"/home/vinker/dev/background_project/experiements/width_22_07/"
# output_pref = f"/home/vinker/dev/background_project/experiements/big_test_07_27/"

# # =========================
# # ====== debug =======
# # =========================
# use_wandb = 0
# wandb_project_name = "width_and_points_07_21"
# weights = [0.0001, 0.0004, 0.0008, 0.001, 0.002, 0.01]
# num_iter = 1001
# save_interval = 100
# num_sketches = 1
# output_pref = f"/home/vinker/dev/background_project/experiements/width_21_07_demo/"


# =========================
# ====== set params =======
# =========================
path_to_files = "/home/vinker/dev/input_images/output_sketches/"
model = "ViT-B/32"
num_strokes=32
layer_opt = 4
clip_conv_layer_weights_int = [0 for k in range(12)]
clip_conv_layer_weights_int[layer_opt] = 1
clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)
mlp_points_weights_path = "none"


# =========================
# ==== per run params =====
# =========================
# filename = f"house_layer4.png"
# path_res_pref = "/home/vinker/dev/background_project/experiements/big_test_07_27/"
path_res_pref = "/home/vinker/dev/background_project/experiements/width_20_07/"
filename = args.filename
res_filename = args.res_filename
#"horse_easy.png"

# source_im_name="semi-complex_mask"

#"mlp_clipasso_32s_easy-background-crop"
# res_filename = f"Cos_mlp_ViT_l4_32s_{source_im_name}"

mlp_train = args.mlp

lr = 1.0
if mlp_train:
    lr = 1e-4


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

for i, w in enumerate(weights):
    start_w = time.time()
    test_name_pref = "prev-mlp"
    if args.switch_loss:
        test_name_pref += f"_switch{args.switch_loss}"
    test_name_pref += f"_clip_l{layer_opt}{args.clip_conv_loss_type}_"
    test_name = f"points{args.optimize_points}_width_{args.width_loss_type}_{w}_{test_name_pref}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}"
    print("**** test_name ****")
    print(test_name)
    if i == 0:
        mlp_width_weights_path = "none"
    else:
        mlp_width_weights_path = f"{output_pref}/points{args.optimize_points}_width_{args.width_loss_type}_{weights[i-1]}_{test_name_pref}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}/width_mlp.pt"
        print("**** mlp_width_weights_path ****")
        print(mlp_width_weights_path)
        assert os.path.exists(mlp_width_weights_path)

        if args.mlp:
            mlp_points_weights_path = f"{output_pref}/points{args.optimize_points}_width_{args.width_loss_type}_{weights[i-1]}_{test_name_pref}_{num_strokes}s_{os.path.splitext(os.path.basename(file_))[0]}/points_mlp.pt"
            print("**** mlp_points_weights_path ****")
            print(mlp_points_weights_path)
            assert os.path.exists(mlp_points_weights_path)

    sp.run(["python", 
            "scripts/width_and_points/run_sketch.py", 
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
            "--width_loss_weight", str(w),
            "--optimize_points", str(args.optimize_points),
            "--width_loss_type", str(args.width_loss_type),
            "--path_svg", path_svg,
            "--mlp_width_weights_path", mlp_width_weights_path,
            "--save_interval", str(save_interval),
             "--mlp_points_weights_path", mlp_points_weights_path,
                "--switch_loss", str(args.switch_loss)])
    print("=" * 50)
    print("time per w: ", time.time() - start_w)
    print("=" * 50)

# print("=" * 50)
# print("time per layer: ", time.time() - start_l)
# print("=" * 50)