import os
import numpy as np
from scipy.optimize import curve_fit


def get_svg_file(path):
    files = os.listdir(path)
    files = [f for f in files if ".svg" in f]
    return files[0]

def get_seed(filename):
    filename = filename[:-9]
    keyword = 'seed'
    before_keyword, keyword, after_keyword = filename.partition(keyword)
    return after_keyword

def get_clip_loss(path, layer):
    path_config = f"{path}/config.npy"
    config = np.load(path_config, allow_pickle=True)[()]
    loss_clip = np.array(config[f"loss_eval"])
    best_iter = np.argsort(loss_clip)[0]
    loss_clip_layer = np.array(config[f"clip_vit_l{layer}_original_eval"])
    return loss_clip, best_iter, loss_clip_layer

def ratios_to_str(ratios):
    ratios_str = ""
    for r_ in ratios:
        r_str = f"{r_:.3f}"
        ratios_str += f"{float(r_str)},"
    ratios_str = ratios_str[:-1]
    return ratios_str

def get_ratios_dict(path_to_initial_sketches, rel_layer, folder_name_rel, folder_name_l, layer, seed1, ratios_rel, im_name):        
    svg_filename = get_svg_path(f"{path_to_initial_sketches}/{folder_name_rel}")
    seed = get_seed(svg_filename)
    path_l_rel = f"{path_to_initial_sketches}/{folder_name_rel}/{folder_name_rel}_seed{seed}"
    loss_clip_l_rel, best_iter_rel, loss_clip_layer_rel = get_clip_loss(path_l_rel, rel_layer)
    best_lclip_rel = loss_clip_l_rel[best_iter_rel]
    best_lclip_layer_rel = loss_clip_layer_rel[best_iter_rel]

    svg_filename = get_svg_path(f"{path_to_initial_sketches}/{folder_name_l}")
    seed = get_seed(svg_filename)
    path_li = f"{path_to_initial_sketches}/{folder_name_l}/{folder_name_l}_seed{seed}"
    loss_clip_li, best_iter, loss_clip_layer = get_clip_loss(path_li, layer)
    best_lclip_i = loss_clip_li[best_iter]
    best_lclip_layer = loss_clip_layer[best_iter]

    div1 = best_lclip_layer / best_lclip_layer_rel  
    div2 = best_lclip_i / best_lclip_rel   

    ratios_l1 = ratios_rel / div1
    ratios_l = ratios_rel / div2

    ratios_str1 = ratios_to_str(ratios_l1)
    ratios_str2 = ratios_to_str(ratios_l)
    # ratios_str1 (layer), ratios_str2 (L4 + layer)
    return ratios_str1, ratios_str2


def func(x, a, c, d):
    return a*np.exp(c*x)

def func_inv(y,a,c,d):
    return np.log(y / a) * (1 / c)

def get_func(ratios_rel, start_x, start_ys):
    target_ys = ratios_rel[start_ys:]
    x = np.linspace(start_x, start_x + len(target_ys) - 1, len(target_ys))
    # calculate exponent
    popt, pcov = curve_fit(func, x, target_ys, maxfev=3000)
    return popt


def get_clip_loss2(path, layer, object_or_background):
    path_config = f"{path}/config.npy"
    config = np.load(path_config, allow_pickle=True)[()]
    loss_clip = np.array(config[f"loss_eval"])
    best_iter = np.argsort(loss_clip)[0]
    loss_clip_layer = np.array(config[f"clip_vit_l{layer}_original_eval"])
    if object_or_background == "object":
        loss_clip_layer4 = np.array(config[f"clip_vit_l4_original_eval"])
        loss_clip_layer = 1*loss_clip_layer4 + loss_clip_layer
    return best_iter, loss_clip_layer
       
def get_ratios_dict3(path_to_initial_sketches, folder_name_l, layer, im_name, object_or_background, step_size_l, num_ratios=8):        
    # get the sketch of the given layer, and get L_clip_i 
    svg_filename = get_svg_file(f"{path_to_initial_sketches}/{folder_name_l}")
    seed = get_seed(svg_filename)
    path_li = f"{path_to_initial_sketches}/{folder_name_l}/{folder_name_l}_seed{seed}"
    best_iter, loss_clip_layer = get_clip_loss2(path_li, layer, object_or_background)
    best_lclip_layer = loss_clip_layer[best_iter]
    r_1_k = 1 / best_lclip_layer
    
    # get the next ratios by jumping by 2
    r_j_k = r_1_k
    ratios_k = [r_1_k]
    for j in range(4):
        r_j_k = r_j_k / 2
        ratios_k.append(r_j_k)
    start_ys, start_x, end_x_addition = 0, 0, 0
    popt = get_func(ratios_k, start_x=0, start_ys=0) # fit the function to ratios_k
    x_1_k = func_inv([r_1_k], *popt)

    step_size = step_size_l
    num_steps = num_ratios - start_x + end_x_addition
    start_ = x_1_k[0]
    end = num_steps * step_size
    # sample the function from the initial scaled r_1 with the corresponding step size
    new_xs_layer_l = np.linspace(start_, end - step_size + start_, num_steps) 
    # print("new_xs_layer_l", new_xs_layer_l)
    ratios_li = func(new_xs_layer_l, *popt)
    ratios_str = ratios_to_str(ratios_li)
    xs_layer_l_str = ratios_to_str(new_xs_layer_l)
    print(f"layer {layer} r_1_k {r_1_k} \n new {ratios_str} \n x {xs_layer_l_str}\n")
    return ratios_str


