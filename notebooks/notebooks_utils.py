import os
import pydiffvg
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import warp
from skimage import transform as tf
import imageio 
import matplotlib.font_manager as font_manager
import sys 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import PIL
from skimage.transform import resize
import torch.nn.functional as F
import matplotlib.cm as cm
from scipy import ndimage
from skimage import morphology
from skimage.measure import label   
from skimage.transform import resize


p = os.path.abspath('..')
sys.path.insert(1, p)
import u2net_utils
import sketch_utils as utils
from U2Net_.model import U2NET
from scipy import ndimage
from torchvision.utils import make_grid
from scipy.optimize import curve_fit

def read_svg(path_svg, multiply=False, resize_obj=False, params=None, opacity=1, device=None):
    
    if device is None:
        print("setting device")
        device = torch.device("cuda" if (
            torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    pydiffvg.set_device(device)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        path_svg)
    for group in shape_groups:
        group.stroke_color = torch.tensor([0,0,0,opacity])
    if resize_obj and params:
        w, h = params["scale_w"], params["scale_h"]
        for path in shapes:
            path.points = path.points / canvas_width
            path.points = 2 * path.points - 1
            path.points[:,0] /= (w)# / canvas_width)
            path.points[:,1] /= (h)# / canvas_height)
            path.points = 0.5 * (path.points + 1.0) * canvas_width
            center_x, center_y = canvas_width / 2, canvas_height / 2
            path.points[:,0] += (params["original_center_x"] * canvas_width - center_x)
            path.points[:,1] += (params["original_center_y"] * canvas_height - center_y)
    if multiply:
        canvas_width *= 2
        canvas_height *= 2
        for path in shapes:
            path.points *= 2
            path.stroke_width *= 2
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,   # num_samples_x
                  2,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + \
        torch.ones(img.shape[0], img.shape[1], 3,
                   device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3].cpu().numpy()
    return img

def get_seed(filename):
    filename = filename[:-9]
    keyword = 'seed'
    before_keyword, keyword, after_keyword = filename.partition(keyword)
    return after_keyword

def get_svg_path(path):
    files = os.listdir(f"{path}")
    path_svg_ = [f for f in files if ".svg" in f][0]
    return path_svg_


def get_size_of_largest_cc(binary_im):
    labels, num = label(binary_im, background=0, return_num=True)
    (unique, counts) = np.unique(labels, return_counts=True)
    args = np.argsort(counts)[::-1]
    largest_cc_label = unique[args][1] # without background
    return counts[args][1]
    # unique_sort_top = unique[args][1:num_c] # without background
    # seg1 = mask_np.copy()
    # labels = labels.copy()
    # seg1[~np.isin(labels, largest_cc_label)] = 0

def get_num_cc(binary_im):
    labels, num = label(binary_im, background=0, return_num=True)
    return num

def get_obj_bb(binary_im):
    y = np.where(binary_im != 0)[0]
    x = np.where(binary_im != 0)[1]
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    return x0, x1, y0, y1
    # np.min(a[0])
    print(x0)
    return [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]

def cut_and_resize(im, x0, x1, y0, y1, new_height, new_width):
    cut_obj = im[y0 : y1, x0 : x1]
    resized_obj = resize(cut_obj, (new_height, new_width))
    new_mask = np.zeros(im.shape)
    center_y_new = int(new_height / 2)
    center_x_new = int(new_width / 2)
    center_targ_y = int(new_mask.shape[0] / 2)
    center_targ_x = int(new_mask.shape[1] / 2)
    startx, starty = center_targ_x - center_x_new, center_targ_y - center_y_new
    new_mask[starty: starty + resized_obj.shape[0], startx: startx + resized_obj.shape[1]] = resized_obj
    return new_mask, starty + center_y_new, startx + center_x_new
    


def get_target(target_, device, image_scale=224, resize_obj=False, return_params=False):
    target = Image.open(target_)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")

    masked_im, mask, original_mask, params = get_mask_u2net(target, device, resize_obj=resize_obj, return_params=return_params)
    # masked_im = masked_im.convert("RGB")

    transforms_ = []
    transforms_.append(transforms.Resize(
        image_scale, interpolation=PIL.Image.BICUBIC))
    transforms_.append(transforms.CenterCrop(image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    
    target_ = data_transforms(target).unsqueeze(0).to(device)
    masked_im = data_transforms(masked_im).unsqueeze(0).to(device)
    
    mask = Image.fromarray((mask*255).astype(np.uint8)).convert('RGB')
    mask = data_transforms(mask).unsqueeze(0).to(device)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    
    original_mask = Image.fromarray((original_mask*255).astype(np.uint8)).convert('RGB')
    original_mask = data_transforms(original_mask).unsqueeze(0).to(device)
    original_mask[original_mask < 0.5] = 0
    original_mask[original_mask >= 0.5] = 1
    
    return target_, mask, masked_im, original_mask, params


def get_mask_u2net(pil_im, device, use_gpu=True, resize_obj=False, return_params=False):
    w, h = pil_im.size[0], pil_im.size[1]

    test_salobj_dataset = u2net_utils.SalObjDataset(imgs_list=[pil_im],
                                                    lbl_name_list=[],
                                                    transform=transforms.Compose([u2net_utils.RescaleT(320),
                                                                                  u2net_utils.ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    input_im_trans = next(iter(test_salobj_dataloader))

    model_dir = os.path.join("/home/vinker/dev/backgroundCLIPasso/CLIPasso/U2Net_/saved_models/u2net.pth")
    # model_dir = '/nfs/private/yuval/scene_clipasso/weights/u2net.pth'
    net = U2NET(3, 1)
    if torch.cuda.is_available() and use_gpu:
        net.load_state_dict(torch.load(model_dir))
        net.to(device)
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    with torch.no_grad():
        input_im_trans = input_im_trans.type(torch.FloatTensor)
        d1, d2, d3, d4, d5, d6, d7 = net(input_im_trans.detach())
    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    predict = pred
    
    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1

    # opposite mask (mask the object insteadof background)
    # predict_dilated_back = 1 - torch.tensor(ndimage.binary_dilation(predict[0].cpu().numpy(), structure=np.ones((11,11))).astype(np.int)).unsqueeze(0)
    
    mask = torch.cat([predict, predict, predict], axis=0).permute(1, 2, 0)
    mask = mask.cpu().numpy()
    mask = resize(mask, (h, w), anti_aliasing=False)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    original_mask = mask.copy()
    
    im = Image.fromarray((mask[:, :, 0]*255).astype(np.uint8)).convert('RGB')
    im_np = np.array(pil_im)
    
    params = {}
    if resize_obj:
        mask_np = mask[:,:,0].astype(int)
        target_np = im_np
        min_size = int(get_size_of_largest_cc(mask_np) / 3)
        mask_np2 = morphology.remove_small_objects((mask_np > 0), min_size=min_size).astype(int)
        num_cc = get_num_cc(mask_np2)

        mask_np3 = np.ones((h, w ,3))
        mask_np3[:,:,0] = mask_np2
        mask_np3[:,:,1] = mask_np2
        mask_np3[:,:,2] = mask_np2

        x0, x1, y0, y1 = get_obj_bb(mask_np2)

        im_width, im_height = x1 - x0, y1 - y0
        max_size = max(im_width, im_height)
        target_size = int(min(h, w) * 0.7)

        if max_size < target_size and num_cc == 1:
            if im_width > im_height:
                new_width, new_height = target_size, int((target_size / im_width) * im_height)
            else:
                new_width, new_height = int((target_size / im_height) * im_width), target_size
            mask, new_center_y, new_center_x = cut_and_resize(mask_np3, x0, x1, y0, y1, new_height, new_width)
            target_np = target_np / target_np.max()
            im_np, new_center_y, new_center_x = cut_and_resize(target_np, x0, x1, y0, y1, new_height, new_width)
            params["original_center_y"] = (y0 + (y1 - y0) / 2) / h
            params["original_center_x"] = (x0 + (x1 - x0) / 2) / w
            params["scale_w"] = new_width / im_width
            params["scale_h"] = new_height / im_height
            
    im_np = im_np / im_np.max()
    im_np = mask * im_np
    im_np[mask == 0] = 1
    im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    im_final = Image.fromarray(im_final)
    
    if return_params:
        return im_final, mask, original_mask, params
    
    return im_final, mask, original_mask, None
