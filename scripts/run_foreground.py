import subprocess as sp
import argparse
import time
import os

# ===================================================
# ================= run foreground ==================
# ===================================================
# This script calls the foreground sketches generation loop.
# You can specify different layers and divs using the --layers and
# --divs parameters. Default is "2,8,11"
# Example of running commands:
# CUDA_VISIBLE_DEVICES=6 python scripts/run_foreground.py --im_name "ballerina"
# CUDA_VISIBLE_DEVICES=2 python scripts/run_foreground.py --im_name "hummingbird"
# CUDA_VISIBLE_DEVICES=3 python scripts/run_foreground.py --im_name "boat"
# ===================================================

# list of divs per layer
# background : layers = [2, 3, 4, 7, 8, 11] divs = [0.35, 0.45, 0.45, 0.45, 0.5, 0.9]
# foreground : layers = [2, 3, 4, 7, 8, 11] divs = [0.35, 0.45, 0.45, 0.4,  0.5, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layers", type=str, default="2,8,11")
parser.add_argument("--divs", type=str, default="0.4,0.5,0.9")
args = parser.parse_args()

layers = [int(l) for l in args.layers.split(",")]
divs = [float(d) for d in args.divs.split(",")]

# run the first row (fidelity axis)
start_time_fidelity_o = time.time()
for l in layers:
    if not os.path.exists(f"./results_sketches/{args.im_name}/runs/object_l{l}_{args.im_name}/points_mlp.pt"):
        num_iter = 1000
        if l < 8: # converge fater for shallow layers
            num_iter = 600
        sp.run(["python", "scripts/generate_fidelity_levels.py", 
                "--im_name", args.im_name,
                "--layer_opt", str(l),
                "--object_or_background", "object",
                "--resize_obj", str(1),
                "--num_iter", str(num_iter)])
end_time_fidelity_o = time.time() - start_time_fidelity_o

# run the columns (simplicity axis)
# this is for objects
start_time_simp_o = time.time()
for l,div in zip(layers,divs):
    print(f"=================== layer{l} ===================")
    sp.run(["python", "scripts/run_ratio.py", 
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background", "object",
            "--min_div", str(div),
            "--resize_obj", str(1)])
end_time_simp_o = time.time() - start_time_simp_o

print("=" * 50)
print(f"end_time_fidelity_o [{end_time_fidelity_o:.3f}]")
print(f"end_time_simp_o [{end_time_simp_o:.3f}]")
total_time = end_time_fidelity_o + end_time_simp_o
print(f"total [{total_time:.3f}]")
print("=" * 50)