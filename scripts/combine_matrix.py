import subprocess as sp
import argparse
import time
import torch
import os
import pydiffvg
import torch.nn as nn
import sys 
import matplotlib.pyplot as plt
from shutil import copyfile
import numpy as np
import imageio
from skimage.transform import resize
import scripts_utils

# ===================================================
# ================= call run ratio ==================
# ===================================================
# This script combines the object and background matrices, and saves the results
# example command:
# CUDA_VISIBLE_DEVICES=5 python scripts/combine_matrix.py --im_name "man_flowers"
# ===================================================

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
args = parser.parse_args()

def copy_files(folders_arr, col_, output_subdir):
    for j, folder_ in enumerate(folders_arr):
        cur_f = f"{runs_dir}/{folder_}"
        
        svg_filename_lst = [s_ for s_ in os.listdir(cur_f) if s_.endswith("best.svg")]#[0]
        if len(svg_filename_lst):
            svg_filename = svg_filename_lst[0]
            svg_path = f"{cur_f}/{svg_filename}"
            new_filename = f"row{j}_col{col_}.svg"
            print(cur_f, new_filename)
            copyfile(svg_path, f"{output_dir}/{output_subdir}/{new_filename}")
        else:
            print(f"{folder_} failed!! using previous")
            print(new_filename)
            
            old_filename = f"row{j - 1}_col{col_}.svg"
            for k in range(j,9):
                new_filename = f"row{k}_col{col_}.svg"
                copyfile(f"{output_dir}/{output_subdir}/{old_filename}", f"{output_dir}/{output_subdir}/{new_filename}")
            

def gen_matrix(output_dir, im_name):    
    runs_folders = os.listdir(runs_dir)
    layers = [2,7,8,11]
    rows_inds = [0,1,2,3]
    # layers = [2,7,11]
    # rows_inds = [0,1,3]
    for layer, col_ in zip(layers, rows_inds):
        layer_paths = [path for path in runs_folders if f"l{layer}" in f"_{str(path)}_" and "ratio" in path]
        object_paths = [path for path in layer_paths if "mask" not in path]
        background_paths = [path for path in layer_paths if "mask" in path]
        
        # print(background_paths)
        sorted_layer_paths_o = sorted(object_paths, key=lambda x: float(x.split("_")[0].replace("ratio", "")), reverse=True)
        sorted_layer_paths_b = sorted(background_paths, key=lambda x: float(x.split("_")[0].replace("ratio", "")), reverse=True)
        # print(sorted_layer_paths_o)
        sorted_layer_paths_o = [f"object_l{layer}_{im_name}"] + sorted_layer_paths_o
        sorted_layer_paths_b = [f"background_l{layer}_{im_name}_mask"] + sorted_layer_paths_b
    
        copy_files(sorted_layer_paths_b, col_, "background_matrix")
        copy_files(sorted_layer_paths_o, col_, "object_matrix")
    
    params_path = f"{output_dir}/runs/object_l11_{im_name}/resize_params.npy"
    if os.path.exists(params_path):
        copyfile(params_path, f"{output_dir}/object_matrix/resize_params.npy")
    mask_path = f"{output_dir}/runs/object_l11_{im_name}/mask.png"
    if os.path.exists(mask_path):
        copyfile(mask_path, f"{output_dir}/object_matrix/mask.png")

def plot_matrix_svg(svgs_path, rows, cols, resize_obj, output_dir, output_name):
    params_path = f"{svgs_path}/resize_params.npy"
    params = None
    if os.path.exists(params_path):
        params = np.load(params_path, allow_pickle=True)[()]
    plt.figure(figsize=(len(cols) * 2, len(rows) * 2))
    for i, row_ in enumerate(rows):
        for j, col_ in enumerate(cols):
            cur_svg = f"{svgs_path}/row{row_}_col{col_}.svg"
            im = scripts_utils.read_svg(cur_svg, resize_obj=resize_obj, params=params, multiply=False, device=device)
            plt.subplot(len(rows),len(cols), i + 1 + len(cols) * j)
            plt.imshow(im)
            plt.axis("off")
    plt.savefig(f"{output_dir}/{output_name}_matrix.png")
    plt.show()
    

def plot_matrix_raster(im_path, rows, cols, output_dir, output_name):
    plt.figure(figsize=(len(cols) * 2, len(rows) * 2))
    for i, row_ in enumerate(rows):
        for j, col_ in enumerate(cols):
            cur_p = f"{im_path}/row{row_}_col{col_}.png"
            im = imageio.imread(cur_p)
            plt.subplot(len(rows),len(cols), i + 1 + len(cols) * j)
            plt.imshow(im)
            plt.axis("off")
    plt.savefig(f"{output_dir}/{output_name}_matrix.png")
    plt.show()
    
        

def combine_matrix(output_dir, rows, cols, output_size = 448):  
    obj_matrix_path = f"{output_dir}/object_matrix"
    back_matrix_path = f"{output_dir}/background_matrix"
    
    params_path = f"{obj_matrix_path}/resize_params.npy"
    params = None
    if os.path.exists(params_path):
        params = np.load(params_path, allow_pickle=True)[()]       
    mask_path = f"{obj_matrix_path}/mask.png"
    mask = imageio.imread(mask_path)
    mask = resize(mask, (output_size, output_size), anti_aliasing=False)
    
    for i, row_ in enumerate(rows):
        for j, col_ in enumerate(cols):
            cur_svg_o = f"{output_dir}/object_matrix/row{row_}_col{col_}.svg"
            raster_o = scripts_utils.read_svg(cur_svg_o, resize_obj=1, params=params, multiply=True, device=device)
            imageio.imsave(f"{output_dir}/object_matrix/row{row_}_col{col_}.png", raster_o)

            cur_svg_b = f"{output_dir}/background_matrix/row{row_}_col{col_}.svg"
            raster_b = scripts_utils.read_svg(cur_svg_b, resize_obj=0, params=params, multiply=True, device=device)
            imageio.imsave(f"{output_dir}/background_matrix/row{row_}_col{col_}.png", raster_b)

            raster_b[mask == 1] = 1
            raster_b[raster_o == 0] = 0
            imageio.imsave(f"{output_dir}/combined_matrix/row{row_}_col{col_}.png", raster_b)


output_dir = f"./results_sketches/{args.im_name}"
if not os.path.exists(f"{output_dir}/object_matrix"):
    os.mkdir(f"{output_dir}/object_matrix")
if not os.path.exists(f"{output_dir}/background_matrix"):
    os.mkdir(f"{output_dir}/background_matrix")
if not os.path.exists(f"{output_dir}/combined_matrix"):
    os.mkdir(f"{output_dir}/combined_matrix")
if not os.path.exists(f"{output_dir}/all_sketches"):
    os.mkdir(f"{output_dir}/all_sketches")


device = torch.device("cuda:0" if (
            torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
runs_dir = f"{output_dir}/runs"
gen_matrix(output_dir, args.im_name)
            
rows = range(9)
cols = [0,1,2,3]
# cols = [0,1,3]

svg_path = f"{output_dir}/background_matrix"
resize_obj=0
rows = range(9)[::2][:-1]
plot_matrix_svg(svg_path, rows, cols, resize_obj, output_dir, "background")


svg_path = f"{output_dir}/object_matrix"
resize_obj=1
rows = range(9)[::2][:-1]
plot_matrix_svg(svg_path, rows, cols, resize_obj, output_dir, "obj")

combine_matrix(output_dir, rows, cols)

plot_matrix_raster(f"{output_dir}/combined_matrix", rows, cols, output_dir, "combined")
