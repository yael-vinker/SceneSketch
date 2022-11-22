import subprocess as sp
import argparse
import time

# ===================================================
# ================= call run ratio ==================
# ===================================================
# This script is to call the run_ratio.py
# Instead of changing run_ratio, you can loop  over your 
# parameters (images and layers per image).
# and then call the existing script from here.
# Example of a running command:
# CUDA_VISIBLE_DEVICES=6 python scripts/run_all.py --im_name "ballerina"
# ===================================================

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
args = parser.parse_args()

# run the first row (fidelity axis)
layers = [2,7,8,11]

start_time_fidelity_b = time.time()
for l in layers:
    sp.run(["python", "scripts/generate_fidelity_levels.py", 
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background", "background"])
end_time_fidelity_b = time.time() - start_time_fidelity_b

start_time_fidelity_o = time.time()
for l in layers:
    sp.run(["python", "scripts/generate_fidelity_levels.py", 
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background", "object",
            "--resize_obj", str(1)])
end_time_fidelity_o = time.time() - start_time_fidelity_o

# run the columns (simplicity axis)
# this is for background
divs = [0.35,0.45,0.5,0.9]
start_time_simp_b = time.time()
for l,div in zip(layers,divs):
    print(f"=================== layer{l} ===================")
    sp.run(["python", "scripts/run_ratio.py", 
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background", "background",
            "--min_div", str(div)])
end_time_simp_b = time.time() - start_time_simp_b

# this is for objects
divs = [0.45,0.4,0.5,0.9]
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
print(f"end_time_fidelity_b [{end_time_fidelity_b:.3f}]")
print(f"end_time_fidelity_o [{end_time_fidelity_o:.3f}]")
print(f"end_time_simp_b [{end_time_simp_b:.3f}]")
print(f"end_time_simp_o [{end_time_simp_o:.3f}]")
total_time = end_time_fidelity_b + end_time_fidelity_o + end_time_simp_b + end_time_simp_o
print(f"total [{total_time:.3f}]")
print("=" * 50)