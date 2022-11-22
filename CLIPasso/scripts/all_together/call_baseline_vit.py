import subprocess as sp
import argparse
# ===================================================
# =============== call baseline ViT =================
# ===================================================
# This script is to call the baseline_vit.py
# Instead of changing baseline_vit, you can loop over your 
# parameters (images and layers per image).
# and then call the existing script from here.
# Example of a running command:
# CUDA_VISIBLE_DEVICES=6 python scripts/all_together/call_baseline_vit.py --im_name "man_flowers"
# ===================================================

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
args = parser.parse_args()

# sp.run(["python", "scripts/all_together/baseline_vit.py", 
#         "--im_name", "woman_city",
#         "--layer_opt", "8",
#         "--object_or_background", "object"])

layers = [2,7,8,11]
for l in layers:
    sp.run(["python", "scripts/all_together/baseline_vit.py", 
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background", "object",
            "--resize_obj", str(1)])


# example:
# layers = [8,2,3,4,5,6,7,9,10,11]
# for l in layers:
#     sp.run(["python", "scripts/all_together/baseline_vit.py", 
#             "--im_name", "woman_city",
#             "--layer_opt", str(l),
#             "--object_or_background", "background"])

# layers = [2,7,8,10,11]
# for l in layers:
#     sp.run(["python", "scripts/all_together/baseline_vit.py", 
#             "--im_name", "woman_city",
#             "--layer_opt", str(l),
#             "--object_or_background", "object"])
