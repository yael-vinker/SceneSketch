import argparse
import os
import random

import numpy as np
import pydiffvg
import torch
import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # =================================
    # ============ general ============
    # =================================
    parser.add_argument("target", help="target image path")
    parser.add_argument("--output_dir", type=str,
                        help="directory to save the output images and loss")
    parser.add_argument("--path_svg", type=str, default="none",
                        help="if you want to load an svg file and train from it")
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_object", type=int, default=0)
    parser.add_argument("--resize_obj", type=int, default=0)
    parser.add_argument("--fix_scale", type=int, default=0)
    parser.add_argument("--display_logs", type=int, default=0)
    parser.add_argument("--display", type=int, default=0)

    # =================================
    # ============ wandb ============
    # =================================
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_user", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="test")
    parser.add_argument("--wandb_project_name", type=str, default="none")

    # =================================
    # =========== training ============
    # =================================
    parser.add_argument("--num_iter", type=int, default=500,
                        help="number of optimization iterations")
    parser.add_argument("--num_stages", type=int, default=1,
                        help="training stages, you can train x strokes, then freeze them and train another x strokes etc.")
    parser.add_argument("--lr_scheduler", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0)
    # parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--color_lr", type=float, default=0.01)
    parser.add_argument("--width_lr", type=float, default=1e-4)
    parser.add_argument("--color_vars_threshold", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="for optimization it's only one image")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--min_eval_iter", type=int, default=100)
    parser.add_argument("--image_scale", type=int, default=224)
    parser.add_argument("--loss_mask", type=str, default="none", 
                        help="mask the object during training, can be none|back|for, if you want to mask out the background choose back")
    parser.add_argument("--dilated_mask", type=int, default=0)
    parser.add_argument("--mask_cls", type=str, default="none", help="can be none|only_cls|cls_out")
    parser.add_argument("--mask_attention", type=int, default=0, help="mask during initialisation")
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--process_id", type=int, default=0)


    # =================================
    # ======== strokes params =========
    # =================================
    parser.add_argument("--num_paths", type=int,
                        default=16, help="number of strokes")
    parser.add_argument("--width", type=float,
                        default=1.5, help="stroke width")
    parser.add_argument("--control_points_per_seg", type=int, default=4)
    parser.add_argument("--num_segments", type=int, default=1,
                        help="number of segments for each stroke, each stroke is a bezier curve with 4 control points")
    parser.add_argument("--attention_init", type=int, default=1,
                        help="if True, use the attention heads of Dino model to set the location of the initial strokes")
    parser.add_argument("--saliency_model", type=str, default="clip")
    parser.add_argument("--saliency_clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--xdog_intersec", type=int, default=1)
    parser.add_argument("--mask_object_attention", type=int, default=0)
    parser.add_argument("--softmax_temp", type=float, default=0.3)
    parser.add_argument("--mlp_train", type=int, default=0)
    parser.add_argument("--width_optim", type=int, default=0)
    parser.add_argument("--mlp_width_weights_path", type=str, default="none")
    parser.add_argument("--mlp_points_weights_path", type=str, default="none")
    parser.add_argument("--switch_loss", type=int, default=0)
    parser.add_argument("--gumbel_temp", type=float, default=0.2)
    parser.add_argument("--width_loss_weight", type=str, default="0")
    parser.add_argument("--width_loss_type", type=str, default="L1", help="can be L1|L1_hinge")
    parser.add_argument("--optimize_points", type=int, default=1)
    parser.add_argument("--load_points_opt_weights", type=int, default=0, 
                        help="if True, load the optimizer of the points mlp as well")
    parser.add_argument("--gradnorm", type=int, default=0, help="balance width and location loss")
    parser.add_argument("--width_weights_lst", type=str, default="", help="the weights series used for width loss")
    parser.add_argument("--ratio_loss", type=float, default=0, help="use ratio loss and if so define the ratio")
    
    

    # =================================
    # ============= loss ==============
    # =================================
    parser.add_argument("--percep_loss", type=str, default="none",
                        help="the type of perceptual loss to be used (L2/LPIPS/none)")
    parser.add_argument("--perceptual_weight", type=float, default=0,
                        help="weight the perceptual loss")
    parser.add_argument("--train_with_clip", type=int, default=0)
    parser.add_argument("--clip_weight", type=float, default=0)
    parser.add_argument("--start_clip", type=int, default=0)
    parser.add_argument("--num_aug_clip", type=int, default=4)
    parser.add_argument("--include_target_in_aug", type=int, default=0)
    parser.add_argument("--augment_both", type=int, default=1,
                        help="if you want to apply the affine augmentation to both the sketch and image")
    parser.add_argument("--augemntations", type=str, default="affine",
                        help="can be any combination of: 'affine_noise_eraserchunks_eraser_press'")
    parser.add_argument("--noise_thresh", type=float, default=0.5)
    parser.add_argument("--aug_scale_min", type=float, default=0.7)
    parser.add_argument("--force_sparse", type=float, default=0,
                        help="if True, use L1 regularization on stroke's opacity to encourage small number of strokes")
    parser.add_argument("--clip_conv_loss", type=float, default=1)
    parser.add_argument("--clip_mask_loss", type=float, default=0)
    
    parser.add_argument("--clip_conv_loss_type", type=str, default="L2")
    parser.add_argument("--clip_conv_layer_weights",
                        type=str, default="0,0,1.0,1.0,0")
    parser.add_argument("--clip_model_name", type=str, default="RN101")
    parser.add_argument("--clip_fc_loss_weight", type=float, default=0.1)
    parser.add_argument("--clip_text_guide", type=int, default=0)
    parser.add_argument("--clip_text_fc", type=int, default=0)
    parser.add_argument("--clip_text_layer", type=int, default=0)
    parser.add_argument("--clip_text_guide_loss_type", type=str, default="Cos")
    parser.add_argument("--clip_text_fc_loss_type", type=str, default="L2")
    parser.add_argument("--clip_text_layer_loss_type", type=str, default="L2")
    parser.add_argument("--text_target", type=str, default="none")
    

    args = parser.parse_args()
    set_seed(args.seed)

    assert args.mask_cls in ["none", "only_cls", "cls_out"]
    args.width_loss_weight = float(args.width_loss_weight)
    args.clip_conv_layer_weights = [
        float(item) for item in args.clip_conv_layer_weights.split(',')]

    args.output_dir = os.path.join(args.output_dir, args.wandb_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    jpg_logs_dir = f"{args.output_dir}/jpg_logs"
    svg_logs_dir = f"{args.output_dir}/svg_logs"
    mlps_logs_dir = f"{args.output_dir}/mlps"
    if not os.path.exists(jpg_logs_dir):
        os.mkdir(jpg_logs_dir)
    if not os.path.exists(svg_logs_dir):
        os.mkdir(svg_logs_dir)
    if not os.path.exists(mlps_logs_dir):
        os.mkdir(mlps_logs_dir)

    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                   config=args, name=args.wandb_name, id=wandb.util.generate_id())

    if args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu_id}" if (
            torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
        print(f"Using cuda: {args.gpu_id}")
    else:
        args.device = torch.device("cpu")
    pydiffvg.set_use_gpu(torch.cuda.is_available() and args.use_gpu)
    pydiffvg.set_device(args.device)
    return args


if __name__ == "__main__":
    # for cog predict
    args = parse_arguments()
    final_config = vars(args)
    np.save(f"{args.output_dir}/config_init.npy", final_config)