import timm
import torch


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create a lightweight vit for testing
    model = timm.create_model("vit_small_patch16_224").to(device)
    model.eval()