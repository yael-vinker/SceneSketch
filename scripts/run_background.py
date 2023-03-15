import subprocess as sp
import argparse
import time
import os
# ===================================================
# ================= run background ==================
# ===================================================
# This script calls the background sketches generation loop.
# You can specify different layers and divs using the --layers and
# --divs parameters. Default is "2,8,11"
# Example of running commands:
# CUDA_VISIBLE_DEVICES=6 python scripts/run_background.py --im_name "ballerina"
# CUDA_VISIBLE_DEVICES=2 python scripts/run_background.py --im_name "hummingbird"
# CUDA_VISIBLE_DEVICES=3 python scripts/run_background.py --im_name "boat"
# ===================================================

# list of divs per layer
# background : layers = [2, 3, 4, 7, 8, 11] divs = [0.35, 0.45, 0.45, 0.45, 0.5, 0.9]
# foreground : layers = [2, 3, 4, 7, 8, 11] divs = [0.45, 0.45, 0.45, 0.4,  0.5, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layers", type=str, default="2,8,11")
parser.add_argument("--divs", type=str, default="0.35,0.5,0.85")
args = parser.parse_args()

layers = [int(l) for l in args.layers.split(",")]
divs = [float(d) for d in args.divs.split(",")]

# run the first row (fidelity axis)
start_time_fidelity_b = time.time()
for l in layers:
    if not os.path.exists(f"./results_sketches/{args.im_name}/runs/background_l{l}_{args.im_name}_mask/points_mlp.pt"):
        sp.run(["python", "scripts/generate_fidelity_levels.py", 
                "--im_name", args.im_name,
                "--layer_opt", str(l),
                "--object_or_background", "background"])
end_time_fidelity_b = time.time() - start_time_fidelity_b

# run the columns (simplicity axis)
# this is for background
start_time_simp_b = time.time()
for l,div in zip(layers,divs):
    
    print(f"=================== layer{l} ===================")
    sp.run(["python", "scripts/run_ratio.py", 
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background", "background",
            "--min_div", str(div)])
end_time_simp_b = time.time() - start_time_simp_b

print("=" * 50)
print(f"end_time_fidelity_b [{end_time_fidelity_b:.3f}]")
print(f"end_time_simp_b [{end_time_simp_b:.3f}]")
total_time = end_time_fidelity_b + end_time_simp_b
print(f"total [{total_time:.3f}]")
print("=" * 50)