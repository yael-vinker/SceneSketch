from torchvision import transforms as T
import torch
from CLIP import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = clip.load("ViT-B/32", device=device)[0]
model = model.eval().requires_grad_(False)
preprocess = T.Compose([
    T.Resize(224),  # CLIP was trained on 224x224, but we can resize to arbitrary resolution
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    )])


def calculate_structure_loss(model, outputs, inputs):
    with torch.no_grad():
        target_self_sim = model.calculate_self_sim(inputs)
    current_ss = model.calculate_self_sim(outputs)
    loss = torch.nn.MSELoss()(current_ss, target_self_sim)
    return loss