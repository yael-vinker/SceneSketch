# input folder
# run on all files in folders and save everything
# loop over best iter for each seed
# need 10 seeds, 2 parallel runs each
import os
import argparse
import subprocess as sp
from shutil import copyfile
# CUDA_VISIBLE_DEVICES=7 python scripts/background/image_divide.py --image "assaf_cleanup.jpg"
# CUDA_VISIBLE_DEVICES=6 python scripts/background/image_divide.py --image "yael_cleanup.jpg"
# CUDA_VISIBLE_DEVICES=5 python scripts/background/image_divide.py --image "man1_cleanup.jpg"
# CUDA_VISIBLE_DEVICES=4 python scripts/background/image_divide.py --image "horse_cleanup.png"

# CUDA_VISIBLE_DEVICES=5 python scripts/background/image_divide.py --image "yael_crop.png"
# CUDA_VISIBLE_DEVICES=6 python scripts/background/image_divide.py --image "man1_crop.png"
# CUDA_VISIBLE_DEVICES=7 python scripts/background/image_divide.py --image "horse_crop.png"
# CUDA_VISIBLE_DEVICES=5 python scripts/background/image_divide.py --image "yael.jpg"
# CUDA_VISIBLE_DEVICES=2 python scripts/background/image_divide.py --image "horse.png"

# CUDA_VISIBLE_DEVICES=5 python scripts/background/image_divide_background.py --image "complex-scene-crop_mask.png"
# CUDA_VISIBLE_DEVICES=4 python scripts/background/image_divide_background.py --image "easy-background-crop_mask.png"
# CUDA_VISIBLE_DEVICES=4 python scripts/run_background_layers.py --image "semi-complex.jpeg"

# path_to_files = "/home/vinker/dev/input_images/background/"
path_to_files = "/home/vinker/dev/background_project/notebooks/complex_level_scenes/"


# images = ["mountains3.jpg", "venice.jpg"]
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)
args = parser.parse_args()
im_name = args.image
num_strokes = 32
num_sketches = 1
num_iter = 1001
loss_mask = 'none'


model = "ViT-B/32"
# layers = [0,1,2,3,4,5,6,7,8,9,10,11]
layers = [4]

clip_fc_loss_weight = 0

output_pref = f"background_project/experiements/mlp_12_06"

file_ = f"{path_to_files}/{im_name}"
for i in layers:
    clip_conv_layer_weights_int = [0 for k in range(12)]
    clip_conv_layer_weights_int[i] = 1
    clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
    clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

    print(clip_conv_layer_weights)
    print(model)
    print(clip_fc_loss_weight)
    
    
    test_name = f"mask{loss_mask}_{os.path.splitext(os.path.basename(file_))[0]}_{model[:3]}_l{i}_{num_strokes}s"
    print(test_name)
    sp.run(["python", 
            "scripts/run_sketch.py", 
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
            "--loss_mask", str(loss_mask)])