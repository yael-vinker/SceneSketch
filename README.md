# Project Title

[![arXiv](https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/abs/2202.05822)
[[Project Website](https://clipasso.github.io/clipasso/)]
<br>
<br>
This is the official implementation of <title>, a method for converting a scene image into a sketch with different types of abstractions by disentangling abstraction into two axes of control: fidelity and simplicity. <br>


<img src="repo_images/teaser_4.png" width="800">

At a high level, we define a sketch as a set of Bézier curves and train an MLP network to ... bla <br>
We combine the final matrices into one ..
<br>

## Installation
### Installation via Docker [Recommended]
You can simply pull the docker image from docker hub, containing all the required libraries and packages:
```bash
docker pull yaelvinker/clipasso_docker
docker run --name clipsketch -it yaelvinker/clipasso_docker /bin/bash
```
Now you should have a running container.
Inside the container, clone the repository:

```bash
cd /home
git https://github.com/yael-vinker/SceneSketch.git
cd SceneSketch/
```
Now you are all set and ready to move to the next stage (Run Demo).

### Installation via pip
Note that it is recommended to use the provided docker image, as we rely on diffvg which has specific requirements and does not compile smoothly on every environment.
1.  Clone the repo:
```bash
git clone https://github.com/yael-vinker/SceneSketch.git
cd SceneSketch
```
2. Create a new environment and install the libraries:
```bash
python3.7 -m venv clipsketch
source clipsketch/bin/activate
pip install -r requirements.txt
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/openai/CLIP.git
```
3. Install diffvg:
```bash
git clone https://github.com/BachiLi/diffvg
cd diffvg
git submodule update --init --recursive
python setup.py install
```

<br>

## Run Demo

The input images to be drawn should be located under "target_images".
The expected format should be:
* <image_name>.jpg - the original scene image
* <image_name>_mask.png - the inpainted masked background, <br>
You can automatically produce the inpainted image with [this easy to use LAMA demo](https://huggingface.co/spaces/akhaliq/lama). <br>
Also note that both images need to be square.

As mentioned in the paper, we first generate the first row (fidelity axis) and then for each sketch in the row we generate ...
To run this ppieline, use the "run_all.py" script (under scripts), by simply running:
```bash
python scripts/run_all.py --im_name <im_name>
```
For example, on the ballerina image:
```bash
python scripts/run_all.py --im_name "ballerina"
```
The resulting sketches will be saved to the "results_sketches/<im_name>" folder, in SVG and png format.


Once the script have finished running (this can take up to few hours, for faster version and layer selection see "Play with the scripts" below), you can visualize the results using:
```bash
python scripts/combine_matrix.py --im_name <im_name>
```
<br>
The resulting matrixes and SVGs for the "ballerina" image are provided under "results_sketches/ballerina"
<br>

### Play with the scripts

If you want to run our method for spesific fidelity or simplicity levels, you can use the dedicated scripts under "scripts", spesifically:
* ```generate_fidelity_levels.py``` - generates a single sketch at a given fidelity layer. <br>
    For background, run with:
    ```bash
    python scripts/generate_fidelity_levels.py --im_name <im_name> --layer_opt <desired_layer> --object_or_background "background"
    ```
    For objects, run with:
    ```bash
    python scripts/generate_fidelity_levels.py --im_name <im_name> --layer_opt <desired_layer> --object_or_background "object" --resize_obj 1
    ```
* ```run_ratio.py``` - generates a single column of simplified sketches, for a given fidelity level. <br>
    For background, run with:
    ```bash
    python scripts/run_ratio.py --im_name <im_name> --layer_opt <desired_layer> --object_or_background "background" --min_div <step_size>
    ```
    For objects, run with:
    ```bash
    python scripts/run_ratio.py --im_name <im_name> --layer_opt <desired_layer> --object_or_background "object" --min_div <step_size> --resize 1
    ```
    Where <step_size> is the parameter to sample the function f_k (as described in the paper). You can find the spesific parameters under   "scripts/run_all.py" 



