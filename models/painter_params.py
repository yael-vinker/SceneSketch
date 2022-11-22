import random
import CLIP_.clip as clip
import numpy as np
import pydiffvg
import sketch_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms
import time


class Painter(torch.nn.Module):
    def __init__(self, args,
                num_strokes=4,
                num_segments=4,
                imsize=224,
                device=None,
                target_im=None,
                mask=None):
        super(Painter, self).__init__()

        self.args = args
        self.num_paths = num_strokes
        self.num_segments = num_segments
        self.width = args.width
        self.control_points_per_seg = args.control_points_per_seg
        self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
        self.opacity_optim = args.force_sparse
        self.num_stages = args.num_stages
        self.add_random_noise = "noise" in args.augemntations
        self.noise_thresh = args.noise_thresh
        self.softmax_temp = args.softmax_temp
        self.optimize_points = args.optimize_points
        self.optimize_points_global = args.optimize_points

        self.shapes = []
        self.shape_groups = []
        self.device = device
        self.canvas_width, self.canvas_height = imsize, imsize
        self.points_vars = []
        self.points_init = [] # for mlp training
        self.color_vars = []
        self.color_vars_threshold = args.color_vars_threshold

        self.path_svg = args.path_svg
        self.strokes_per_stage = self.num_paths
        self.optimize_flag = []

        # attention related for strokes initialisation
        self.attention_init = args.attention_init
        self.target_path = args.target
        self.saliency_model = args.saliency_model
        self.xdog_intersec = args.xdog_intersec
        self.mask_object_attention = args.mask_object_attention
        
        self.text_target = args.text_target # for clip gradients
        self.saliency_clip_model = args.saliency_clip_model
        self.define_attention_input(target_im)
        self.mask = mask
        if "for" in args.loss_mask:
            # default for the mask is to mask out the background
            # if mask loss is for it means we want to maskout the foreground
            self.mask = 1 - mask
        self.attention_map = self.set_attention_map() if self.attention_init else None
        
        self.thresh = self.set_attention_threshold_map() if self.attention_init else None
        self.strokes_counter = 0 # counts the number of calls to "get_path"        
        self.epoch = 0
        self.final_epoch = args.num_iter - 1
        self.mlp_train = args.mlp_train
        self.width_optim = args.width_optim
        self.width_optim_global = args.width_optim
        
        if self.width_optim:
            self.init_widths = torch.ones((self.num_paths)).to(device) * 1.5
            self.mlp_width = WidthMLP(num_strokes=self.num_paths, num_cp=self.control_points_per_seg, width_optim=self.width_optim).to(device)
            self.mlp_width_weights_path = args.mlp_width_weights_path
            self.mlp_width_weight_init()
            # self.mlp_width.apply(init_weights)
        self.gumbel_temp = args.gumbel_temp
        self.mlp = MLP(num_strokes=self.num_paths, num_cp=self.control_points_per_seg, width_optim=self.width_optim).to(device) if self.mlp_train else None
        self.mlp_points_weights_path = args.mlp_points_weights_path
        self.mlp_points_weight_init()
        self.out_of_canvas_mask = torch.ones((self.num_paths)).to(self.device)

    def turn_off_points_optim(self):
        self.optimize_points = False

    def switch_opt(self):
        self.width_optim = not self.width_optim
        self.optimize_points = not self.optimize_points


    def mlp_points_weight_init(self):
        if self.mlp_points_weights_path != "none":
            checkpoint = torch.load(self.mlp_points_weights_path)
            self.mlp.load_state_dict(checkpoint['model_state_dict'])
            print("mlp checkpoint loaded from ", self.mlp_points_weights_path)


    def mlp_width_weight_init(self):
        if self.mlp_width_weights_path == "none":
            self.mlp_width.apply(init_weights)
        else:
            checkpoint = torch.load(self.mlp_width_weights_path)
            self.mlp_width.load_state_dict(checkpoint['model_state_dict'])
            print("mlp checkpoint loaded from ", self.mlp_width_weights_path)


        # net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        # net.apply(init_weights)
        
    def init_image(self, stage=0):
        if stage > 0:
            # if multi stages training than add new strokes on existing ones
            # don't optimize on previous strokes
            self.optimize_flag = [False for i in range(len(self.shapes))]
            for i in range(self.strokes_per_stage):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)
                self.optimize_flag.append(True)

        else:
            num_paths_exists = 0
            if self.path_svg != "none":
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = utils.load_svg(self.path_svg)
                # if you want to add more strokes to existing ones and optimize on all of them
                num_paths_exists = len(self.shapes)
                # if self.width_optim:
                for path in self.shapes:
                    self.points_init.append(path.points)
                #         path.stroke_width.data.to(self.device)
                #         print("stroke_width_init", path.stroke_width.is_cuda)
                
                # if self.width_optim:
                #     for path in self.shapes:
                #         path.stroke_width.data.to(self.device)
                #         print("stroke_width_init", path.stroke_width.is_cuda)

            for i in range(num_paths_exists, self.num_paths):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                    fill_color = None,
                                                    stroke_color = stroke_color)
                self.shape_groups.append(path_group)        
            self.optimize_flag = [True for i in range(len(self.shapes))]
        
        # img = self.render_warp()
        # img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - img[:, :, 3:4])
        # img = img[:, :, :3]
        # # Convert img from HWC to NCHW
        # img = img.unsqueeze(0)
        # img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
        # return img
        # utils.imwrite(img.cpu(), '{}/init.png'.format(args.output_dir), gamma=args.gamma, use_wandb=args.use_wandb, wandb_name="init")

    def get_image(self, mode="train"):
        if self.mlp_train:
            img = self.mlp_pass(mode)
        else:
            img = self.render_warp(mode)
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
        return img
    
    def mlp_pass(self, mode, eps=1e-4):
        """
        update self.shapes etc through mlp pass instead of directly (should be updated with the optimizer as well).
        """
        if self.optimize_points_global:
            points_vars = self.points_init
            # reshape and normalise to [-1,1] range
            points_vars = torch.stack(points_vars).unsqueeze(0).to(self.device)
            points_vars = points_vars / self.canvas_width
            points_vars = 2 * points_vars - 1
            if self.optimize_points:
                points = self.mlp(points_vars)
            else:
                with torch.no_grad():
                    points = self.mlp(points_vars)
            
        else:
            points = torch.stack(self.points_init).unsqueeze(0).to(self.device)

        if self.width_optim and mode != "init": #first iter use just the location mlp
            widths_  = self.mlp_width(self.init_widths).clamp(min=1e-8)
            mask_flipped = (1 - widths_).clamp(min=1e-8)
            v = torch.stack((torch.log(widths_), torch.log(mask_flipped)), dim=-1)
            hard_mask = torch.nn.functional.gumbel_softmax(v, self.gumbel_temp, False)
            self.stroke_probs = hard_mask[:, 0] * self.out_of_canvas_mask
            self.widths = self.stroke_probs * self.init_widths            
        
        # normalize back to canvas size [0, 224] and reshape
        all_points = 0.5 * (points + 1.0) * self.canvas_width
        all_points = all_points + eps * torch.randn_like(all_points)
        all_points = all_points.reshape((-1, self.num_paths, self.control_points_per_seg, 2))

        if self.width_optim_global and not self.width_optim:
            self.widths = self.widths.detach()
            # all_points = all_points.detach()
        
        # define new primitives to render
        shapes = []
        shape_groups = []
        for p in range(self.num_paths):
            width = torch.tensor(self.width)
            if self.width_optim_global and mode != "init":
                width = self.widths[p]
            path = pydiffvg.Path(
                num_control_points=self.num_control_points, points=all_points[:,p].reshape((-1,2)),
                stroke_width=width, is_closed=False)
            if mode == "init":
                # do once at the begining, define a mask for strokes that are outside the canvas
                is_in_canvas_ = self.is_in_canvas(self.canvas_width, self.canvas_height, path)
                if not is_in_canvas_:
                    self.out_of_canvas_mask[p] = 0
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=torch.tensor([0,0,0,1]))
            shape_groups.append(path_group)
        
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.canvas_width, self.canvas_height, shapes, shape_groups)
        img = _render(self.canvas_width, # width
                    self.canvas_height, # height
                    2,   # num_samples_x
                    2,   # num_samples_y
                    0,   # seed
                    None,
                    *scene_args)
        self.shapes = shapes.copy()
        self.shape_groups = shape_groups.copy()
        return img
        

    def get_path(self):
        points = []
        p0 = self.inds_normalised[self.strokes_counter] if self.attention_init else (random.random(), random.random())
        points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        
        self.points_init.append(points)

        path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(self.width).to(self.device),
                                is_closed = False)
        self.strokes_counter += 1
        return path

    def render_warp(self, mode):
        if not self.mlp_train:
            if self.opacity_optim:
                for group in self.shape_groups:
                    group.stroke_color.data[:3].clamp_(0., 0.) # to force black stroke
                    group.stroke_color.data[-1].clamp_(0., 1.) # opacity
                    # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
            # uncomment if you want to add random noise
            if self.add_random_noise:
                if random.random() > self.noise_thresh:
                    eps = 0.01 * min(self.canvas_width, self.canvas_height)
                    for path in self.shapes:
                        path.points.data.add_(eps * torch.randn_like(path.points))
        
        if self.width_optim and mode != "init":
            widths_  = self.mlp_width(self.init_widths).clamp(min=1e-8)
            mask_flipped = 1 - widths_
            v = torch.stack((torch.log(widths_), torch.log(mask_flipped)), dim=-1)
            hard_mask = torch.nn.functional.gumbel_softmax(v, self.gumbel_temp, False)
            self.stroke_probs = hard_mask[:, 0] * self.out_of_canvas_mask
            self.widths = self.stroke_probs * self.init_widths  
        
        if self.optimize_points:
            _render = pydiffvg.RenderFunction.apply
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
            img = _render(self.canvas_width, # width
                        self.canvas_height, # height
                        2,   # num_samples_x
                        2,   # num_samples_y
                        0,   # seed
                        None,
                        *scene_args)
        else:
            points = torch.stack(self.points_init).unsqueeze(0).to(self.device)
            shapes = []
            shape_groups = []
            for p in range(self.num_paths):
                width = torch.tensor(self.width)
                if self.width_optim:
                    width = self.widths[p]
                path = pydiffvg.Path(
                    num_control_points=self.num_control_points, points=points[:,p].reshape((-1,2)),
                    stroke_width=width, is_closed=False)
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=None,
                    stroke_color=torch.tensor([0,0,0,1]))
                shape_groups.append(path_group)
            
            _render = pydiffvg.RenderFunction.apply
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                self.canvas_width, self.canvas_height, shapes, shape_groups)
            img = _render(self.canvas_width, # width
                        self.canvas_height, # height
                        2,   # num_samples_x
                        2,   # num_samples_y
                        0,   # seed
                        None,
                        *scene_args)
            self.shapes = shapes.copy()
            self.shape_groups = shape_groups.copy()

        return img

    def parameters(self):
        if self.optimize_points:
            if self.mlp_train:
                self.points_vars = self.mlp.parameters()
            else:
                self.points_vars = []
                # storkes' location optimization
                for i, path in enumerate(self.shapes):
                    if self.optimize_flag[i]:
                        path.points.requires_grad = True
                        self.points_vars.append(path.points)
                        self.optimize_flag[i] = False

        if self.width_optim:
            return self.points_vars, self.mlp_width.parameters()    
        return self.points_vars
    
    def get_mlp(self):
        return self.mlp
    
    def get_width_mlp(self):
        if self.width_optim_global:
            return self.mlp_width
        else:
            return None

    def get_points_parans(self):
        return dict(list(self.mlp.named_parameters()))
        # return self.points_vars
    
    def set_color_parameters(self):
        # for storkes' color optimization (opacity)
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            if self.optimize_flag[i]:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)
        return self.color_vars

    def get_color_parameters(self):
        return self.color_vars
    
    def get_widths(self):
        if self.width_optim_global:
            return self.stroke_probs
        return None
    
    def get_strokes_in_canvas_count(self):
        return self.out_of_canvas_mask.sum()

    def get_strokes_count(self):
        if self.width_optim_global:
            with torch.no_grad():
                return torch.sum(self.stroke_probs)
        return self.num_paths
        
    def is_in_canvas(self, canvas_width, canvas_height, path):
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
                    device=self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3].detach().cpu().numpy()
        return (1 - img).sum()


    def save_svg(self, output_dir, name):
        if not self.width_optim:
            pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        else:
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            new_shapes, new_shape_groups = [], []
            for path in self.shapes:
                # is_in_canvas_ = self.is_in_canvas(self.canvas_width, self.canvas_height, path)
                is_in_canvas_ = True
                w = path.stroke_width / 1.5
                if w > 0.7 and is_in_canvas_:
                    new_shapes.append(path)
                    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(new_shapes) - 1]),
                                                fill_color = None,
                                                stroke_color = stroke_color)
                    new_shape_groups.append(path_group)
            pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, new_shapes, new_shape_groups)
        


    def dino_attn(self):
        patch_size=8 # dino hyperparameter
        threshold=0.6

        # for dino model
        mean_imagenet = torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None].to(self.device)
        std_imagenet = torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None].to(self.device)
        totens = transforms.Compose([
            transforms.Resize((self.canvas_height, self.canvas_width)),
            transforms.ToTensor()
            ])

        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').eval().to(self.device)
        
        self.main_im = Image.open(self.target_path).convert("RGB")
        main_im_tensor = totens(self.main_im).to(self.device)
        img = (main_im_tensor.unsqueeze(0) - mean_imagenet) / std_imagenet
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size
        
        with torch.no_grad():
            attn = dino_model.get_last_selfattention(img).detach().cpu()[0]

        nh = attn.shape[0]
        attn = attn[:,0,1:].reshape(nh,-1)
        val, idx = torch.sort(attn)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
        
        attn = attn.reshape(nh, w_featmap, h_featmap).float()
        attn = nn.functional.interpolate(attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
        
        return attn

    def define_attention_input(self, target_im):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([
                    preprocess.transforms[-1],
                ])
        self.image_input_attn_clip = data_transforms(target_im).to(self.device)

    def clip_attn(self):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        text_input = clip.tokenize([self.text_target]).to(self.device)

        if "RN" in self.saliency_clip_model:
            saliency_layer = "layer4"
            attn_map = gradCAM(
                model.visual,
                self.image_input_attn_clip,
                model.encode_text(text_input).float(),
                getattr(model.visual, saliency_layer)
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        else:
            # attn_map = interpret(self.image_input_attn_clip, text_input, model, device=self.device, index=0).astype(np.float32)
            attn_map = interpret(self.image_input_attn_clip, text_input, model, device=self.device)
            
        del model
        return attn_map

    def set_attention_map(self):
        assert self.saliency_model in ["dino", "clip"]
        if self.saliency_model == "dino":
            return self.dino_attn()
        elif self.saliency_model == "clip":
            return self.clip_attn()
        

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum() 

    def set_inds_clip(self):
        attn_map = (self.attention_map - self.attention_map.min()) / (self.attention_map.max() - self.attention_map.min())
        if self.xdog_intersec:
            xdog = XDoG_()
            im_xdog = xdog(self.image_input_attn_clip[0].permute(1,2,0).cpu().numpy(), k=10)
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map
        if self.mask_object_attention:
            attn_map = attn_map * self.mask[0,0].cpu().numpy()
        
            
        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)
        
        k = self.num_stages * self.num_paths
        self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
        self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
    
        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] =  self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] =  self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return attn_map_soft

    def set_inds_dino(self):
        k = max(3, (self.num_stages * self.num_paths) // 6 + 1) # sample top 3 three points from each attention head
        num_heads = self.attention_map.shape[0]
        self.inds = np.zeros((k * num_heads, 2))
        # "thresh" is used for visualisaiton purposes only
        thresh = torch.zeros(num_heads + 1, self.attention_map.shape[1], self.attention_map.shape[2])
        softmax = nn.Softmax(dim=1)
        for i in range(num_heads):
            # replace "self.attention_map[i]" with "self.attention_map" to get the highest values among
            # all heads. 
            topk, indices = np.unique(self.attention_map[i].numpy(), return_index=True)
            topk = topk[::-1][:k]
            cur_attn_map = self.attention_map[i].numpy()
            # prob function for uniform sampling
            prob = cur_attn_map.flatten()
            prob[prob > topk[-1]] = 1
            prob[prob <= topk[-1]] = 0
            prob = prob / prob.sum()
            thresh[i] = torch.Tensor(prob.reshape(cur_attn_map.shape))

            # choose k pixels from each head            
            inds = np.random.choice(range(cur_attn_map.flatten().shape[0]), size=k, replace=False, p=prob)
            inds = np.unravel_index(inds, cur_attn_map.shape)
            self.inds[i * k: i * k + k, 0] = inds[0]
            self.inds[i * k: i * k + k, 1] = inds[1]
        
        # for visualisaiton
        sum_attn = self.attention_map.sum(0).numpy()
        mask = np.zeros(sum_attn.shape)
        mask[thresh[:-1].sum(0) > 0] = 1
        sum_attn = sum_attn * mask
        sum_attn = sum_attn / sum_attn.sum()
        thresh[-1] = torch.Tensor(sum_attn)

        # sample num_paths from the chosen pixels.
        prob_sum = sum_attn[self.inds[:,0].astype(np.int), self.inds[:,1].astype(np.int)]
        prob_sum = prob_sum / prob_sum.sum()
        new_inds = []
        for i in range(self.num_stages):
            new_inds.extend(np.random.choice(range(self.inds.shape[0]), size=self.num_paths, replace=False, p=prob_sum))
        self.inds = self.inds[new_inds]
        print("self.inds",self.inds.shape)
    
        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] =  self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] =  self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return thresh

    def set_attention_threshold_map(self):
        assert self.saliency_model in ["dino", "clip"]
        if self.saliency_model == "dino":
            return self.set_inds_dino()
        elif self.saliency_model == "clip":
            return self.set_inds_clip()
        
    def get_attn(self):
        return self.attention_map
    
    def get_thresh(self):
        return self.thresh

    def get_inds(self):
        return self.inds
    
    def get_mask(self):
        return self.mask

    def set_random_noise(self, epoch):
        if epoch % self.args.save_interval == 0:
            self.add_random_noise = False
        else:
            self.add_random_noise = "noise" in self.args.augemntations


class PainterOptimizer:
    def __init__(self, args, renderer):
        self.renderer = renderer
        self.points_lr = args.lr
        self.color_lr = args.color_lr
        self.args = args
        self.optim_color = args.force_sparse
        self.width_optim = args.width_optim
        self.width_optim_global = args.width_optim
        self.width_lr = args.width_lr
        self.optimize_points = args.optimize_points
        self.optimize_points_global = args.optimize_points
        self.points_optim = None
        self.width_optimizer = None
        self.mlp_width_weights_path = args.mlp_width_weights_path
        self.mlp_points_weights_path = args.mlp_points_weights_path
        self.load_points_opt_weights = args.load_points_opt_weights
        # self.only_width = args.only_width

    def turn_off_points_optim(self):
        self.optimize_points = False

    def switch_opt(self):
        self.width_optim = not self.width_optim
        self.optimize_points = not self.optimize_points

    def init_optimizers(self):
        if self.width_optim:
            points_params, width_params = self.renderer.parameters()
            self.width_optimizer = torch.optim.Adam(width_params, lr=self.width_lr)
            if self.mlp_width_weights_path != "none":
                checkpoint = torch.load(self.mlp_width_weights_path)
                self.width_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("optimizer checkpoint loaded from ", self.mlp_width_weights_path)
        else:
            points_params = self.renderer.parameters()
        
        if self.optimize_points:
            self.points_optim = torch.optim.Adam(points_params, lr=self.points_lr)
            if self.mlp_points_weights_path != "none" and self.load_points_opt_weights:
                checkpoint = torch.load(self.mlp_points_weights_path)
                self.points_optim.load_state_dict(checkpoint['optimizer_state_dict'])
                print("optimizer checkpoint loaded from ", self.mlp_points_weights_path)
        
        if self.optim_color:
            self.color_optim = torch.optim.Adam(self.renderer.set_color_parameters(), lr=self.color_lr)
            

    def update_lr(self, counter):
        if self.optimize_points:
            new_lr = utils.get_epoch_lr(counter, self.args)
            for param_group in self.points_optim.param_groups:
                param_group["lr"] = new_lr
    
    def zero_grad_(self):
        if self.optimize_points:
            self.points_optim.zero_grad()
        if self.width_optim:
            self.width_optimizer.zero_grad()
        if self.optim_color:
            self.color_optim.zero_grad()
    
    def step_(self):
        if self.optimize_points:
            self.points_optim.step()
        if self.width_optim:
            self.width_optimizer.step()
        if self.optim_color:
            self.color_optim.step()
    
    def get_lr(self, optim="points"):
        if optim == "points" and self.optimize_points_global:
            return self.points_optim.param_groups[0]['lr']
        if optim == "width" and self.width_optim_global:
            return self.width_optimizer.param_groups[0]['lr']
        else:
            return None

    def get_points_optim(self):
        return self.points_optim

    def get_width_optim(self):
        return self.width_optimizer
    
    


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


def interpret(image, texts, model, device):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images, mode="saliency")
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = [] # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_probs.detach() #attn_probs shape is 12, 50, 50
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1) # mean of the 12 something
        cams.append(cam)  
        R = R + torch.bmm(cam, R)
              
    cams_avg = torch.cat(cams) # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam    


class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma=0.98
        self.phi=200
        self.eps=-0.1
        self.sigma=0.8
        self.binarize=True
        
    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0  + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')
        return imdiff


class MLP(nn.Module):
    def __init__(self, num_strokes, num_cp, width_optim=False):
        super().__init__()
        outdim = 1000
        self.width_optim = width_optim
        self.layers_points = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_strokes * num_cp * 2, outdim),
            nn.SELU(inplace=True),
            nn.Linear(outdim, outdim),
            nn.SELU(inplace=True),
            # nn.ReLU(),
            # nn.Linear(1000, 1000),
            # nn.ReLU(),
            nn.Linear(outdim, num_strokes * num_cp * 2),
            # nn.Tanh()
        )

        # if width_optim:
        #     self.layers_width = nn.Sequential(
        #         nn.Linear(num_strokes, num_strokes),
        #         nn.ReLU(),
        #         nn.Linear(num_strokes, num_strokes),
        #         nn.ReLU(),
        #         nn.Linear(num_strokes, num_strokes),
        #         nn.Sigmoid()
        #     )


    def forward(self, x, widths=None):
        '''Forward pass'''
        deltas = self.layers_points(x)
        # if self.width_optim:
        #     return x.flatten() + 0.1 * deltas, self.layers_width(widths)
        return x.flatten() + 0.1 * deltas

class WidthMLP(nn.Module):
    def __init__(self, num_strokes, num_cp, width_optim=False):
        super().__init__()
        outdim = 1000
        self.width_optim = width_optim

        self.layers_width = nn.Sequential(
            nn.Linear(num_strokes, outdim),
            nn.SELU(inplace=True),
            nn.Linear(outdim, outdim),
            nn.SELU(inplace=True),
            nn.Linear(outdim, num_strokes),
            nn.Sigmoid()
        )


    def forward(self, widths=None):
        '''Forward pass'''
        return self.layers_width(widths)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)