import subprocess as sp
import argparse
import time

# ===================================================
# ================= run all ==================
# ===================================================
# This script runs both foreground and background one after the other
# If you have a single GPU, you should use this script.
# Example of running commands:
# CUDA_VISIBLE_DEVICES=6 python scripts/run_all.py --im_name "man_flowers"
# CUDA_VISIBLE_DEVICES=2 python scripts/run_all.py --im_name "hummingbird"
# CUDA_VISIBLE_DEVICES=3 python scripts/run_all.py --im_name "boat"
# ===================================================

# list of divs per layer
# background : layers = [2, 3, 4, 7, 8, 11] divs = [0.35, 0.45, 0.45, 0.45, 0.5, 0.9]
# foreground : layers = [2, 3, 4, 7, 8, 11] divs = [0.45, 0.45, 0.45, 0.4,  0.5, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
args = parser.parse_args()

sp.run(["python", "scripts/run_background.py", "--im_name", args.im_name])
sp.run(["python", "scripts/run_foreground.py", "--im_name", args.im_name])