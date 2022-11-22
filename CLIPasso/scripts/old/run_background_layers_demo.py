import os
import argparse
import subprocess as sp
from shutil import copyfile


# CUDA_VISIBLE_DEVICES=4 python scripts/run_background_layers_demo.py --image "semi-complex.jpeg" --mlp 0
# CUDA_VISIBLE_DEVICES=5 python scripts/run_background_layers_demo.py --image "semi-complex.jpeg" --mlp 1
# CUDA_VISIBLE_DEVICES=6 python scripts/run_background_layers_demo.py --image "semi-complex.jpeg" --mlp 1 --loss_mask "for"
# CUDA_VISIBLE_DEVICES=2 python scripts/run_background_layers_demo.py --image "semi-complex_mask.png" --mlp 1

path_to_files = "/home/vinker/dev/background_project/notebooks/complex_level_scenes/"
# clip_mask_loss

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)
parser.add_argument("--mlp", type=int, default=0)
parser.add_argument("--loss_mask", type=str, default="none")
args = parser.parse_args()
im_name = args.image

num_strokes = 32
num_sketches = 1
num_iter = 1001
loss_mask = args.loss_mask
model = "ViT-B/32"
layers = [2,3,4,5,6,7,8,9,10,11]
clip_fc_loss_weight = 0

mlp_train = args.mlp
lr = 1.0
if mlp_train:
    lr = 1e-4
mask_object_attention = 0
output_pref = f"background_project/experiements/mlp_13_06"

file_ = f"{path_to_files}/{im_name}"
for i in layers:
    clip_conv_layer_weights_int = [0 for k in range(12)]
    clip_conv_layer_weights_int[i] = 1
    clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
    clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

    print(clip_conv_layer_weights)
    print(model)
    print(clip_fc_loss_weight)
    
    test_name = ""
    if mlp_train:
        test_name += "mlp_"
    test_name += f"mask_{loss_mask}_{os.path.splitext(os.path.basename(file_))[0]}_{model[:3]}_l{i}_{num_strokes}s"
    print(test_name)
    sp.run(["python", 
            "scripts/run_sketch_demo.py", 
            "--target_file", file_,
            "--output_pref", output_pref,
            "--num_strokes", str(num_strokes),
            "--num_iter", str(num_iter),
            "--test_name", test_name,
            "--num_sketches", str(num_sketches),
            "--mask_object", "0",
            "--fix_scale", "0",
            "--clip_fc_loss_weight", str(clip_fc_loss_weight),
            "--clip_conv_layer_weights", clip_conv_layer_weights,
            "--clip_model_name", model,
            "--loss_mask", str(loss_mask),
            "--mlp_train", str(mlp_train),
            "--lr", str(lr),
            "--mask_object_attention", str(mask_object_attention)])