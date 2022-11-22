import sys
sys.stdout.flush()

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
import multiprocessing as mp
import os
import subprocess as sp
from shutil import copyfile

import numpy as np
import torch
from IPython.display import Image as Image_colab
from IPython.display import display

parser = argparse.ArgumentParser()
parser.add_argument("--target_file", type=str,
                    help="target image file, located in <target_images>")
parser.add_argument("--output_pref", type=str, default="for_arik",
                    help="target image file, located in <target_images>")
parser.add_argument("--num_strokes", type=int, default=16,
                    help="number of strokes used to generate the sketch, this defines the level of abstraction.")
parser.add_argument("--num_iter", type=int, default=2001,
                    help="number of iterations")
parser.add_argument("--test_name", type=str, default="test",
                    help="target image file, located in <target_images>")
                    
parser.add_argument("--fix_scale", type=int, default=0,
                    help="if the target image is not squared, it is recommended to fix the scale")
parser.add_argument("--mask_object", type=int, default=0,
                    help="if the target image contains background, it's better to mask it out")
parser.add_argument("--num_sketches", type=int, default=1,
                    help="it is recommended to draw 3 sketches and automatically chose the best one")
parser.add_argument("--multiprocess", type=int, default=1,
                    help="recommended to use multiprocess if your computer has enough memory")
parser.add_argument("--mask_object_attention", type=int, default=0,
                    help="if the target image contains background, it's better to mask it out")

parser.add_argument("--clip_fc_loss_weight", type=str, default="0")
parser.add_argument("--clip_conv_layer_weights", type=str, default="0,0,1.0,1.0,0")
parser.add_argument("--clip_conv_loss_type", type=str, default="L2")

parser.add_argument("--clip_model_name", type=str, default="RN101")
parser.add_argument("--loss_mask", type=str, default="none")
parser.add_argument("--mlp_train", type=int, default=0)
parser.add_argument("--lr", type=float, default=1.0)
parser.add_argument("--clip_mask_loss", type=int, default=0)
parser.add_argument("--clip_conv_loss", type=int, default=0)
parser.add_argument("--dilated_mask", type=int, default=0)
parser.add_argument("--mask_attention", type=int, default=0)

parser.add_argument("--use_wandb", type=int, default=0)
parser.add_argument("--wandb_project_name", type=str, default="")
parser.add_argument("--mask_cls", type=str, default="none")
parser.add_argument("--width_optim", type=int, default=0)
parser.add_argument("--width_loss_weight", type=str, default="0")
parser.add_argument("--optimize_points", type=int, default=1)
parser.add_argument("--width_loss_type", type=str, default="L1")
parser.add_argument("--path_svg", type=str, default="none")
parser.add_argument("--mlp_width_weights_path", type=str, default="none")


parser.add_argument('-colab', action='store_true')
parser.add_argument('-cpu', action='store_true')
args = parser.parse_args()

multiprocess = not args.colab and args.num_sketches > 1 and args.multiprocess
abs_path = os.path.abspath(os.getcwd())

target = args.target_file
assert os.path.isfile(target), f"{target} does not exists!"

if not os.path.isfile(f"{abs_path}/U2Net_/saved_models/u2net.pth"):
    sp.run(["gdown", "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
           "-O", "U2Net_/saved_models/"])

output_all = "/home/vinker/dev/"

output_dir = f"{args.output_pref}/{args.test_name}/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=" * 50)
print(f"Processing [{args.target_file}] ...")
if args.colab:
    img_ = Image_colab(target)
    display(img_)
print(f"Results will be saved to \n[{output_dir}] ...")
print("=" * 50)

num_iter = args.num_iter
# save_interval = 100
save_interval = 5
use_gpu = not args.cpu
if not torch.cuda.is_available():
    use_gpu = False
    print("CUDA is not configured with GPU, running with CPU instead.")
    print("Note that this will be very slow, it is recommended to use colab.")
print(f"GPU: {use_gpu}")
seeds = list(range(0, args.num_sketches * 1000, 1000))
# seeds = list(range(2000, args.num_sketches * 4000, 1000))


exit_codes = []
manager = mp.Manager()
losses_all = manager.dict()


def run(seed, wandb_name):
    exit_code = sp.run(["python", "painterly_rendering.py", target,
                            "--num_paths", str(args.num_strokes),
                            "--output_dir", output_dir,
                            "--wandb_name", wandb_name,
                            "--num_iter", str(num_iter),
                            "--save_interval", str(save_interval),
                            "--seed", str(seed),
                            "--use_gpu", str(int(use_gpu)),
                            "--fix_scale", str(args.fix_scale),
                            "--mask_object", str(args.mask_object),
                            "--mask_object_attention", str(
                                args.mask_object_attention),
                            "--display_logs", str(int(args.colab)),
                            "--use_wandb", str(args.use_wandb),
                            "--wandb_project_name", str(args.wandb_project_name),
                            "--clip_fc_loss_weight", args.clip_fc_loss_weight,
                            "--clip_conv_layer_weights", args.clip_conv_layer_weights,
                            "--clip_model_name", args.clip_model_name,
                            "--loss_mask", str(args.loss_mask),
                            "--clip_conv_loss", str(args.clip_conv_loss),
                            "--mlp_train", str(args.mlp_train),
                            "--lr", str(args.lr),
                            "--clip_mask_loss", str(args.clip_mask_loss),
                            "--dilated_mask", str(args.dilated_mask),
                            "--clip_conv_loss_type", str(args.clip_conv_loss_type),
                            "--mask_cls", args.mask_cls,
                            "--width_optim", str(args.width_optim),
                            "--width_loss_weight", str(args.width_loss_weight),
                            "--mask_attention", str(args.mask_attention),
                            "--optimize_points", str(args.optimize_points),
                            "--width_loss_type", args.width_loss_type,
                            "--path_svg", args.path_svg,
                            "--mlp_width_weights_path", args.mlp_width_weights_path])
    if exit_code.returncode:
        sys.exit(1)

    config = np.load(f"{output_dir}/{wandb_name}/config.npy",
                     allow_pickle=True)[()]
    loss_eval = np.array(config['loss_eval'])
    inds = np.argsort(loss_eval)
    losses_all[wandb_name] = loss_eval[inds][0]

# num_sketches
print("multiprocess", multiprocess)
if multiprocess:
    ncpus = 10
    P = mp.Pool(ncpus)  # Generate pool of workers

for j in range(args.num_sketches):
    seed = seeds[j]
    wandb_name = f"{args.test_name}_seed{seed}"
    if multiprocess:
        # run simulation and ISF analysis in each process
        P.apply_async(run, (seed, wandb_name))
    else:
        run(seed, wandb_name)

if multiprocess:
    P.close()
    P.join()  # start processes

sorted_final = dict(sorted(losses_all.items(), key=lambda item: item[1]))
copyfile(f"{output_dir}/{list(sorted_final.keys())[0]}/best_iter.svg",
         f"{output_dir}/{list(sorted_final.keys())[0]}_best.svg")
if args.width_optim:
    copyfile(f"{output_dir}/{list(sorted_final.keys())[0]}/width_mlp.pt",
             f"{output_dir}/width_mlp.pt")
