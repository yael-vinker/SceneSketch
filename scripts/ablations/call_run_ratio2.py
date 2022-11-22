import subprocess as sp
import argparse

# ===================================================
# ================= call run ratio ==================
# ===================================================
# This script is to call the run_ratio.py
# Instead of changing run_ratio, you can loop  over your 
# parameters (images and layers per image).
# and then call the existing script from here.
# Example of a running command:
# CUDA_VISIBLE_DEVICES=2 python scripts/ablations/call_run_ratio.py --im_name "man_flowers" --optimize_points 0 --ablation_name "noMLPloc"
# CUDA_VISIBLE_DEVICES=5 python scripts/ablations/call_run_ratio.py --im_name "semi-complex" --optimize_points 0 --ablation_name "noMLPloc"
# CUDA_VISIBLE_DEVICES=6 python scripts/ablations/call_run_ratio.py --im_name "woman_city" --optimize_points 0 --ablation_name "noMLPloc"

# CUDA_VISIBLE_DEVICES=2 python scripts/ablations/call_run_ratio.py --im_name "man_flowers" --optimize_points 1 --ablation_name "noLsparse"
# CUDA_VISIBLE_DEVICES=5 python scripts/ablations/call_run_ratio.py --im_name "semi-complex" --optimize_points 1 --ablation_name "noLsparse"
# CUDA_VISIBLE_DEVICES=6 python scripts/ablations/call_run_ratio.py --im_name "woman_city" --optimize_points 1 --ablation_name "noLsparse"

# CUDA_VISIBLE_DEVICES=2 python scripts/ablations/call_run_ratio.py --im_name "man_flowers" --optimize_points 1 --ablation_name "fk_step1"
# CUDA_VISIBLE_DEVICES=5 python scripts/ablations/call_run_ratio.py --im_name "semi-complex" --optimize_points 1 --ablation_name "fk_step1"
# CUDA_VISIBLE_DEVICES=6 python scripts/ablations/call_run_ratio.py --im_name "woman_city" --optimize_points 1 --ablation_name "fk_step1"

# CUDA_VISIBLE_DEVICES=2 python scripts/ablations/call_run_ratio.py --im_name "man_flowers" --optimize_points 1 --ablation_name "gradnorm0"
# CUDA_VISIBLE_DEVICES=5 python scripts/ablations/call_run_ratio.py --im_name "semi-complex" --optimize_points 1 --ablation_name "gradnorm0"
# CUDA_VISIBLE_DEVICES=6 python scripts/ablations/call_run_ratio.py --im_name "woman_city" --optimize_points 1 --ablation_name "gradnorm0"
# ===================================================

# CUDA_VISIBLE_DEVICES=3 python scripts/ablations/call_run_ratio.py --im_name "man_flowers" --optimize_points 1 --ablation_name "gradnorm_bug" --object_or_background "object" --layer 11 --div 0.9
# CUDA_VISIBLE_DEVICES=3 python scripts/ablations/call_run_ratio.py --im_name "woman_city" --optimize_points 1 --ablation_name "gradnorm_bug" --object_or_background "object" --layer 11 --div 0.9

# CUDA_VISIBLE_DEVICES=3 python scripts/ablations/call_run_ratio.py --im_name "woman_city" --optimize_points 1 --ablation_name "gradnorm_bug"
# CUDA_VISIBLE_DEVICES=1 python scripts/ablations/call_run_ratio.py --im_name "man_flowers" --optimize_points 1 --ablation_name "gradnorm_bug" --layer 2 --div 0.35
# CUDA_VISIBLE_DEVICES=5 python scripts/ablations/call_run_ratio.py --im_name "semi-complex" --optimize_points 1 --ablation_name "gradnorm_bug" --layer 2 --div 0.35
# CUDA_VISIBLE_DEVICES=1 python scripts/ablations/call_run_ratio.py --im_name "woman_city" --optimize_points 1 --ablation_name "gradnorm_bug" --layer 2 --div 0.9

# CUDA_VISIBLE_DEVICES=6 python scripts/ablations/call_run_ratio.py --im_name "semi-complex" --optimize_points 1 --ablation_name "gradnorm_bug" --layer 2 --div 0.9
# CUDA_VISIBLE_DEVICES=1 python scripts/ablations/call_run_ratio.py --im_name "man_flowers" --optimize_points 1 --ablation_name "gradnorm_bug" --layer 2 --div 0.9
# layers = [2,4,7,8,11]
# divs = [0.45,0.4,0.4,0.5,0.9]


# CUDA_VISIBLE_DEVICES=2 python scripts/ablations/call_run_ratio.py --im_name "man_flowers" --optimize_points 1 --ablation_name "directLs_fixA"
# CUDA_VISIBLE_DEVICES=1 python scripts/ablations/call_run_ratio.py --im_name "semi-complex" --optimize_points 1 --ablation_name "directLs_fixA"
# CUDA_VISIBLE_DEVICES=3 python scripts/ablations/call_run_ratio.py --im_name "woman_city" --optimize_points 1 --ablation_name "directLs_fixA"

# CUDA_VISIBLE_DEVICES=7 python scripts/ablations/call_run_ratio2.py --im_name "man_flowers" --optimize_points 1 --ablation_name "sameR8" --layer 2 --div 0.5
# CUDA_VISIBLE_DEVICES=2 python scripts/ablations/call_run_ratio2.py --im_name "man_flowers" --optimize_points 1 --ablation_name "sameR8" --layer 7 --div 0.5
# CUDA_VISIBLE_DEVICES=3 python scripts/ablations/call_run_ratio2.py --im_name "man_flowers" --optimize_points 1 --ablation_name "sameR8" --layer 8 --div 0.5
# CUDA_VISIBLE_DEVICES=5 python scripts/ablations/call_run_ratio2.py --im_name "man_flowers" --optimize_points 1 --ablation_name "sameR8" --layer 11 --div 0.5


parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--ablation_name", type=str, default="ablation")
parser.add_argument("--optimize_points", type=int, default=1)
parser.add_argument("--layer", type=int, default=11)
parser.add_argument("--div", type=float, default=0.9)
parser.add_argument("--object_or_background", type=str, default="background")


args = parser.parse_args()
# layers = [11]
# divs = [0.9]
layers = [args.layer]
divs = [args.div]
resize_obj = 0
if args.object_or_background == "object":
    resize_obj = 1
# layers = [11]
# divs = [0.9]
for l,div in zip(layers,divs):
    # ratios_str = ratios_str_dict[l]
    print(f"=================== layer{l} ===================")
    sp.run(["python", "scripts/ablations/run_ratio.py", 
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background", args.object_or_background,
            "--min_div", str(div),
            "--num_strokes", str(64),
            "--ablation_name", args.ablation_name,
            "--optimize_points", str(args.optimize_points),
            "--resize_obj", str(resize_obj),
            "--ratios_str", "16.661,11.781,8.331,5.891,4.165,2.945,2.083,1.473"])

# ratios_str_dict = {
#     2: "1.0,0.785,0.616,0.483,0.379,0.297,0.233,0.183",
#     # 11: "1.0,0.536,0.287,0.154,0.082,0.044,0.024,0.013",
#     11: "1.0,0.5,0.25,0.125"
# }
# layers = [11]
# divs = [0.9]
# for l,div in zip(layers,divs):
#     ratios_str = ratios_str_dict[l]
#     print(f"=================== layer{l} ===================")
#     sp.run(["python", "scripts/ablations/run_ratio.py", 
#             "--im_name", args.im_name,
#             "--layer_opt", str(l),
#             "--object_or_background", "background",
#             "--min_div", str(div),
#             "--num_strokes", str(64),
#             "--ablation_name", args.ablation_name,
#             "--optimize_points", str(args.optimize_points),
#             "--ratios_str", ratios_str])

# fk step 1
# layers = [11,2]
# divs = [1,1]
# for l,div in zip(layers,divs):
#     # ratios_str = ratios_str_dict[l]
#     print(f"=================== layer{l} ===================")
#     sp.run(["python", "scripts/ablations/run_ratio.py", 
#             "--im_name", args.im_name,
#             "--layer_opt", str(l),
#             "--object_or_background", "background",
#             "--min_div", str(div),
#             "--num_strokes", str(64),
#             "--ablation_name", args.ablation_name,
#             "--optimize_points", str(args.optimize_points)])