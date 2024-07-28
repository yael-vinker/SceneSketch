import subprocess as sp
import argparse
import time
import os
import multiprocessing as mp

# ===================================================
# ================= run background ==================
# ===================================================
# The script generates single sketch of different levels of fidelity.
# You can specify different layers using the --layers
# Example of running commands:
# python scripts/run_single_sketch.py --im_name "ballerina" --layer_opt 4
# ===================================================

start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer_opt", type=int, default=4)
parser.add_argument("--fg_bg_separation", type=int, default=1)
parser.add_argument("--num_strokes", type=int, default=64,
                    help="number of strokes used to generate the sketch, this defines the level of abstraction.")
args = parser.parse_args()


def run(object_or_background, resize_obj, num_iter, process_id):
    exit_code = sp.run(["python", "scripts/generate_fidelity_levels.py",
                            "--im_name", args.im_name,
                            "--layer_opt", str(args.layer_opt),
                            "--object_or_background", object_or_background,
                            "--num_iter", str(num_iter),
                            "--resize_obj", resize_obj,
                            "--num_sketches", str(1),
                            "--num_strokes",str(args.num_strokes),
                            "--process_id",str(process_id),
                            "--fg_bg_separation", str(args.fg_bg_separation)])


if __name__ == "__main__":
        mp.set_start_method("spawn")
        ncpus = 10
        P = mp.Pool(ncpus)
        num_iter = 1501      
        
        if not ((args.fg_bg_separation and os.path.exists(f"./results_sketches/{args.im_name}/runs/background_l{str(args.layer_opt)}_{args.im_name}_mask/points_mlp.pt")) \
                or ((not args.fg_bg_separation) and os.path.exists(f"./results_sketches/{args.im_name}/runs/l{str(args.layer_opt)}_{args.im_name}/points_mlp.pt"))):
                P.apply_async(run, ("background", str(0), num_iter, 0))

        if args.fg_bg_separation:
                num_iter = 1000
                if args.layer_opt < 8: # converge fater for shallow layers
                        num_iter = 600
                if not os.path.exists(f"./results_sketches/{args.im_name}/runs/object_l{str(args.layer_opt)}_{args.im_name}/points_mlp.pt"):
                        P.apply_async(run, ("object", str(1), num_iter, 1))

        P.close()
        P.join()  # start processes

        sp.run(["python", "scripts/combine_matrix.py", 
                "--im_name", args.im_name,
                "--layers", str(args.layer_opt),
                "--rows", "1",
                "--is_single", "1",
                "--fg_bg_separation", str(args.fg_bg_separation)])

        total_time = time.time() - start_time
        print(f"Time run single sketch [{total_time:.3f}] seconds")