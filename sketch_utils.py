import os

import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydiffvg
import skimage
import skimage.io
import torch
import wandb
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from scipy import ndimage

import u2net_utils
from U2Net_.model import U2NET
from skimage.transform import resize
import PIL
from skimage import morphology
from skimage.measure import label 
from models.painter_params import MLP, WidthMLP
from shutil import copyfile
import warnings
warnings.filterwarnings("ignore", message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.")



def imwrite(img, filename, gamma=2.2, normalize=False, use_wandb=False, wandb_name="", step=0, input_im=None):
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(img, np.ndarray):
        img = img.data.numpy()
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        # repeat along the third dimension
        img = np.expand_dims(img, 2)
    img[:, :, :3] = np.power(img[:, :, :3], 1.0/gamma)
    img = (img * 255).astype(np.uint8)

    skimage.io.imsave(filename, img, check_contrast=False)
    images = [wandb.Image(Image.fromarray(img), caption="output")]
    if input_im is not None and step == 0:
        images.append(wandb.Image(input_im, caption="input"))
    if use_wandb:
        wandb.log({wandb_name + "_": images}, step=step)


def plot_batch(inputs, outputs, output_dir, step, use_wandb, title):
    plt.ioff()
    plt.figure(figsize=(3,6))
    plt.subplot(2, 1, 1)
    grid = make_grid(inputs.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("inputs")

    plt.subplot(2, 1, 2)
    grid = make_grid(outputs, normalize=False, pad_value=2)
    npgrid = grid.detach().cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("outputs")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"output": wandb.Image(plt)}, step=step)
    plt.savefig("{}/{}".format(output_dir, title))
    plt.close()


def log_input(use_wandb, epoch, inputs, output_dir):
    grid = make_grid(inputs.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.tight_layout()
    if use_wandb:
        wandb.log({"input": wandb.Image(plt)}, step=epoch)
    plt.close()
    input_ = inputs[0].cpu().clone().detach().permute(1, 2, 0).numpy()
    input_ = (input_ - input_.min()) / (input_.max() - input_.min())
    input_ = (input_ * 255).astype(np.uint8)
    imageio.imwrite("{}/{}.png".format(output_dir, "input"), input_)


def log_sketch_summary_final(path_svg, use_wandb, device, epoch, loss, title):
    canvas_width, canvas_height, shapes, shape_groups = load_svg(path_svg)
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
    img = img[:, :, :3]
    plt.imshow(img.cpu().numpy())
    plt.axis("off")
    plt.title(f"{title} best res [{epoch}] [{loss}.]")
    if use_wandb:
        wandb.log({title: wandb.Image(plt)})
    plt.close()


def log_best_normalised_sketch(configs_to_save, output_dir, use_wandb, device, eval_interval, min_eval_iter):
    np.save(f"{output_dir}/config.npy", configs_to_save)
    losses_eval = {}
    for k in configs_to_save.keys():
        if "_original_eval" in k and "normalization" not in k:
            cur_arr = np.array(configs_to_save[k])
            mu = cur_arr.mean()
            std = cur_arr.std()
            losses_eval[k] = (cur_arr - mu) / std

    final_normalise_losses = sum(list(losses_eval.values()))
    sorted_iters = np.argsort(final_normalise_losses)
    index = 0
    best_iter = sorted_iters[index]
    best_normalised_loss = final_normalise_losses[best_iter]
    best_num_strokes = configs_to_save["num_strokes"][best_iter]
    
    iter_ = best_iter * eval_interval + min_eval_iter
    configs_to_save["best_normalised_iter"] = iter_
    configs_to_save["best_normalised_loss"] = best_normalised_loss
    configs_to_save["best_normalised_num_strokes"] = best_num_strokes
    copyfile(f"{output_dir}/mlps/points_mlp{iter_}.pt",
                 f"{output_dir}/points_mlp.pt")
    copyfile(f"{output_dir}/mlps/width_mlp{iter_}.pt",
                 f"{output_dir}/width_mlp.pt")

    if use_wandb:
        wandb.run.summary["best_normalised_loss"] = best_normalised_loss
        wandb.run.summary["best_normalised_iter"] = configs_to_save["best_normalised_iter"]
        wandb.run.summary["best_normalised_num_strokes"] = best_num_strokes
    


def log_sketch_summary(sketch, title, use_wandb):
    plt.figure()
    grid = make_grid(sketch.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    if use_wandb:
        wandb.run.summary["best_loss_im"] = wandb.Image(plt)
    plt.close()


def load_svg(path_svg):
    svg = os.path.join(path_svg)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        svg)
    return canvas_width, canvas_height, shapes, shape_groups


def read_svg(path_svg, device, multiply=False):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        path_svg)
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
    img = img[:, :, :3]
    return img


def plot_attn_dino(attn, threshold_map, inputs, inds, use_wandb, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(2, attn.shape[0] + 2, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, 2)
    plt.imshow(attn.sum(0).numpy(), interpolation='nearest')
    plt.title("atn map sum")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 3)
    plt.imshow(threshold_map[-1].numpy(), interpolation='nearest')
    plt.title("prob sum")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 4)
    plt.imshow(threshold_map[:-1].sum(0).numpy(), interpolation='nearest')
    plt.title("thresh sum")
    plt.axis("off")

    for i in range(attn.shape[0]):
        plt.subplot(2, attn.shape[0] + 2, i + 3)
        plt.imshow(attn[i].numpy())
        plt.axis("off")
        plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 1 + i + 4)
        plt.imshow(threshold_map[i].numpy())
        plt.axis("off")
    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()


def plot_attn_clip(attn, threshold_map, inputs, inds, use_wandb, output_path, display_logs):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("atn map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    threshold_map_ = (threshold_map - threshold_map.min()) / \
        (threshold_map.max() - threshold_map.min())
    plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
    plt.title("prob softmax")
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()


def plot_atten(attn, threshold_map, inputs, inds, use_wandb, output_path, saliency_model, display_logs):
    if saliency_model == "dino":
        plot_attn_dino(attn, threshold_map, inputs,
                       inds, use_wandb, output_path)
    elif saliency_model == "clip":
        plot_attn_clip(attn, threshold_map, inputs, inds,
                       use_wandb, output_path, display_logs)


def fix_image_scale(im):
    im_np = np.array(im) / 255
    height, width = im_np.shape[0], im_np.shape[1]
    max_len = max(height, width) + 20
    new_background = np.ones((max_len, max_len, 3))
    y, x = max_len // 2 - height // 2, max_len // 2 - width // 2
    new_background[y: y + height, x: x + width] = im_np
    new_background = (new_background / new_background.max()
                      * 255).astype(np.uint8)
    new_im = Image.fromarray(new_background)
    return new_im


def get_size_of_largest_cc(binary_im):
    labels, num = label(binary_im, background=0, return_num=True)
    (unique, counts) = np.unique(labels, return_counts=True)
    args = np.argsort(counts)[::-1]
    largest_cc_label = unique[args][1] # without background
    return counts[args][1]

def get_num_cc(binary_im):
    labels, num = label(binary_im, background=0, return_num=True)
    return num

def get_obj_bb(binary_im):
    y = np.where(binary_im != 0)[0]
    x = np.where(binary_im != 0)[1]
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    return x0, x1, y0, y1

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
    return new_mask


# u2net source : https://github.com/xuebinqin/U-2-Net
def get_mask_u2net(args, pil_im):
    # return : numpy binary mask, with 1 where the salient object is and 0 in the background
    # return the masked image in it's original size
    # assume that input image is squre
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

    model_dir = os.path.join("U2Net_/saved_models/u2net.pth")
    net = U2NET(3, 1)
    if torch.cuda.is_available() and args.use_gpu:
        net.load_state_dict(torch.load(model_dir))
        net.to(args.device)
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    with torch.no_grad():
        input_im_trans = input_im_trans.type(torch.FloatTensor)
        d1, d2, d3, d4, d5, d6, d7 = net(input_im_trans.cuda(args.gpu_id))

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
    
    im = Image.fromarray((mask[:, :, 0]*255).astype(np.uint8)).convert('RGB')
    im.save(f"{args.output_dir}/mask.png")
    im_np = np.array(pil_im)
    im_np = im_np / im_np.max()

    params = {}
    if args.resize_obj:
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
            mask = cut_and_resize(mask_np3, x0, x1, y0, y1, new_height, new_width)
            target_np = target_np / target_np.max()
            im_np = cut_and_resize(target_np, x0, x1, y0, y1, new_height, new_width)

            params["original_center_y"] = (y0 + (y1 - y0) / 2) / h
            params["original_center_x"] = (x0 + (x1 - x0) / 2) / w
            params["scale_w"] = new_width / im_width
            params["scale_h"] = new_height / im_height

    im_np = mask * im_np
    im_np[mask == 0] = 1
    im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    im_final = Image.fromarray(im_final)

    np.save(f"{args.output_dir}/resize_params.npy", params)
    return im_final, mask


def get_init_points(path_svg):
    points_init = []
    canvas_width, canvas_height, shapes, shape_groups = load_svg(path_svg)
    for path in shapes:
        points_init.append(path.points)
    return points_init, canvas_width, canvas_height


def is_in_canvas(canvas_width, canvas_height, path, device):
    shapes, shape_groups = [], []
    stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
    shapes.append(path)
    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                        fill_color = None,
                                        stroke_color = stroke_color)
    shape_groups.append(path_group) 
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
    img = img[:, :, :3].detach().cpu().numpy()
    return (1 - img).sum()



def inference_sketch(args, eps=1e-4):
    output_dir = args.output_dir
    mlp_points_weights_path = f"{output_dir}/points_mlp.pt"
    mlp_width_weights_path = f"{output_dir}/width_mlp.pt"
    sketch_init_path = f"{output_dir}/svg_logs/init_svg.svg"
    output_path = f"{output_dir}/"
    device = args.device

    num_paths = args.num_paths
    control_points_per_seg = args.control_points_per_seg
    width_ = 1.5
    num_control_points = torch.zeros(1, dtype = torch.int32) + (control_points_per_seg - 2)
    init_widths = torch.ones((num_paths)).to(device) * width_
    
    mlp = MLP(num_strokes=num_paths, num_cp=control_points_per_seg).to(device)
    checkpoint = torch.load(mlp_points_weights_path)
    mlp.load_state_dict(checkpoint['model_state_dict'])

    if args.width_optim:
        mlp_width = WidthMLP(num_strokes=num_paths, num_cp=control_points_per_seg).to(device)
        checkpoint = torch.load(mlp_width_weights_path)
        mlp_width.load_state_dict(checkpoint['model_state_dict'])
    
    points_vars, canvas_width, canvas_height = get_init_points(sketch_init_path)
    points_vars = torch.stack(points_vars).unsqueeze(0).to(device)
    points_vars = points_vars / canvas_width
    points_vars = 2 * points_vars - 1
    points = mlp(points_vars)
    
    all_points = 0.5 * (points + 1.0) * canvas_width
    all_points = all_points + eps * torch.randn_like(all_points)
    all_points = all_points.reshape((-1, num_paths, control_points_per_seg, 2))

    if args.width_optim: #first iter use just the location mlp
        widths_  = mlp_width(init_widths).clamp(min=1e-8)
        mask_flipped = (1 - widths_).clamp(min=1e-8)
        v = torch.stack((torch.log(widths_), torch.log(mask_flipped)), dim=-1)
        hard_mask = torch.nn.functional.gumbel_softmax(v, 0.2, False)
        stroke_probs = hard_mask[:, 0]
        widths = stroke_probs * init_widths   
        
    shapes = []
    shape_groups = []
    for p in range(num_paths):
        width = torch.tensor(width_)
        if args.width_optim:
            width = widths[p]
        w = width / 1.5 
        path = pydiffvg.Path(
            num_control_points=num_control_points, points=all_points[:,p].reshape((-1,2)),
            stroke_width=width, is_closed=False)
        # if mode == "init":
        #     # do once at the begining, define a mask for strokes that are outside the canvas
        is_in_canvas_ = is_in_canvas(canvas_width, canvas_height, path, device)
        if is_in_canvas_ and w > 0.7:
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=torch.tensor([0,0,0,1]))
            shape_groups.append(path_group)
    pydiffvg.save_svg(f"{output_path}/best_iter.svg", canvas_width, canvas_height, shapes, shape_groups)
