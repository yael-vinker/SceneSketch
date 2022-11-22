# input folder
# run on all files in folders and save everything
# loop over best iter for each seed
# need 10 seeds, 2 parallel runs each
import os
import argparse
import subprocess as sp
from shutil import copyfile

# CUDA_VISIBLE_DEVICES=6 python scripts/object_sketch/baseline.py --image "horse_small.png"
# CUDA_VISIBLE_DEVICES=7 python scripts/object_sketch/baseline.py --image "horse_reg.png"
#  easy-background-crop.jpeg
# path_to_files = "/home/vinker/dev/input_images/background/"
path_to_files = "/home/vinker/dev/background_project/notebooks/complex_level_scenes/"


# images = ["mountains3.jpg", "venice.jpg"]
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)
args = parser.parse_args()
im_name = args.image
num_strokes = 32
num_sketches = 2
# num_iter = 11
num_iter = 2001
loss_mask = "none"

#horse_reg.png

model = "RN101"
clip_conv_layer_weights = "0,0,1,1,0"
# layers = [0,1,2,3,4,5,6,7,8,9,10,11]
layers = [4]

clip_fc_loss_weight = 0.1

output_pref = f"background_project/experiements/object_sketch_29_06"

file_ = f"{path_to_files}/{im_name}"

    
test_name = f"baseline_{os.path.splitext(os.path.basename(file_))[0]}_{model[:3]}_{num_strokes}s"
print(test_name)
sp.run(["python", 
        "scripts/object_sketch/run_sketch.py", 
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