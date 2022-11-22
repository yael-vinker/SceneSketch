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
# CUDA_VISIBLE_DEVICES=2 python scripts/ablations/call_run_ratio_step.py --im_name "man_flowers" --optimize_points 1 --div 0.9
# CUDA_VISIBLE_DEVICES=5 python scripts/ablations/call_run_ratio_step.py --im_name "semi-complex" --optimize_points 1 --div 0.9
# CUDA_VISIBLE_DEVICES=6 python scripts/ablations/call_run_ratio_step.py --im_name "woman_city" --optimize_points 1 --div 0.9

# CUDA_VISIBLE_DEVICES=1 python scripts/ablations/call_run_ratio_step.py --im_name "man_flowers" --optimize_points 1 --div 0.35
# CUDA_VISIBLE_DEVICES=3 python scripts/ablations/call_run_ratio_step.py --im_name "semi-complex" --optimize_points 1 --div 0.35
# CUDA_VISIBLE_DEVICES=7 python scripts/ablations/call_run_ratio_step.py --im_name "woman_city" --optimize_points 1 --div 0.35
# ===================================================

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--ablation_name", type=str, default="ablation")
parser.add_argument("--optimize_points", type=int, default=1)
parser.add_argument("--div", type=float, default=1.0)


args = parser.parse_args()
ablation_name = f"step{args.div}_fix"
# layers = [2,11]
# divs = [1.0,1.0]
layers = [2,11]
for l in layers:
    # ratios_str = ratios_str_dict[l]
    print(f"=================== layer{l} ===================")
    sp.run(["python", "scripts/ablations/run_ratio.py", 
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background", "background",
            "--min_div", str(args.div),
            "--num_strokes", str(64),
            "--ablation_name", ablation_name,
            "--optimize_points", str(args.optimize_points)])

# layers = [2,11]
# divs = [0.5,0.5]
# # layers = [11]
# # divs = [0.9]
# for l,div in zip(layers,divs):
#     # ratios_str = ratios_str_dict[l]
#     print(f"=================== layer{l} ===================")
#     sp.run(["python", "scripts/ablations/run_ratio.py", 
#             "--im_name", args.im_name,
#             "--layer_opt", str(l),
#             "--object_or_background", "background",
#             "--min_div", str(div),
#             "--num_strokes", str(64),
#             "--ablation_name", "step0.5",
#             "--optimize_points", str(args.optimize_points)])

# ratios_str_dict = {
#     2: "1.0,0.785,0.616,0.483,0.379,0.297,0.233,0.183",
#     11: "1.0,0.536,0.287,0.154,0.082,0.044,0.024,0.013",
# }
# layers = [2,11]
# divs = [0.35,0.9]
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