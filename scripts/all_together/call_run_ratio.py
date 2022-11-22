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
# CUDA_VISIBLE_DEVICES=2 python scripts/all_together/call_run_ratio.py --im_name "ballerina"
# CUDA_VISIBLE_DEVICES=2 python scripts/all_together/call_run_ratio.py --im_name "dog"
# CUDA_VISIBLE_DEVICES=3 python scripts/all_together/call_baseline_vit.py --im_name "man_camera"
# CUDA_VISIBLE_DEVICES=5 python scripts/all_together/call_baseline_vit.py --im_name "basketball-5"
# CUDA_VISIBLE_DEVICES=1 python scripts/all_together/call_run_ratio.py --im_name "house3"
# CUDA_VISIBLE_DEVICES=6 python scripts/all_together/call_run_ratio.py --im_name "house4"

# CUDA_VISIBLE_DEVICES=6 python scripts/all_together/call_baseline_vit.py --im_name "bull"
# CUDA_VISIBLE_DEVICES=2 python scripts/all_together/call_run_ratio.py --im_name "dog"

# CUDA_VISIBLE_DEVICES=1 python scripts/all_together/call_run_ratio.py --im_name "house3"
# CUDA_VISIBLE_DEVICES=6 python scripts/all_together/call_run_ratio.py --im_name "house4"

# CUDA_VISIBLE_DEVICES=1 python scripts/all_together/call_baseline_vit.py --im_name "ballerina"
# CUDA_VISIBLE_DEVICES=2 python scripts/all_together/call_baseline_vit.py --im_name "assaf"
# CUDA_VISIBLE_DEVICES=3 python scripts/all_together/call_baseline_vit.py --im_name "man_camera"
# CUDA_VISIBLE_DEVICES=5 python scripts/all_together/call_baseline_vit.py --im_name "basketball-5"
# CUDA_VISIBLE_DEVICES=6 python scripts/all_together/call_baseline_vit.py --im_name "bull"
# CUDA_VISIBLE_DEVICES=7 python scripts/all_together/call_baseline_vit.py --im_name "yael"
# ===================================================

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
args = parser.parse_args()

# this is for background
layers = [2,7,8,11]
divs = [0.35,0.45,0.5,0.9]
for l,div in zip(layers,divs):
    print(f"=================== layer{l} ===================")
    sp.run(["python", "scripts/all_together/run_ratio.py", 
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background", "background",
            "--min_div", str(div)])

# this is for objects
layers = [2,7,8,11]
divs = [0.45,0.4,0.5,0.9]
for l,div in zip(layers,divs):
    print(f"=================== layer{l} ===================")
    sp.run(["python", "scripts/all_together/run_ratio.py", 
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background", "object",
            "--min_div", str(div),
            "--resize_obj", str(1)])