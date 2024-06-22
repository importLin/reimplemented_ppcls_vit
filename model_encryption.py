import timm
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np


def extracting_weight_from_model(timm_model):
    pe_weight = None
    pose_weight = None

    with torch.no_grad():
        for name, param in timm_model.named_parameters():
            # print(f"Name: {name}, Shape: {param.shape}")
            if name == "patch_embed.proj.weight":
                pe_weight = param.clone()

            if name == "pos_embed":
                pose_weight = param.clone()

    return pe_weight, pose_weight


def reloading_weight_to_model(timm_model, pe_weight, pose_weight):
    with torch.no_grad():
        for name, param in timm_model.named_parameters():
            # print(f"Name: {name}, Shape: {param.shape}")
            if name == "patch_embed.proj.weight":
                param.copy_(pe_weight)

            if name == "pos_embed":
                param.copy_(pose_weight)
    return timm_model


def pe_shuffling(pe_weight, shuffling_order):
    # (d, 3, 16, 16)
    original_shape = pe_weight.shape
    flatten_weight = torch.flatten(pe_weight, start_dim=1)
    shuffled_weight = flatten_weight[..., shuffling_order]
    shuffled_weight = shuffled_weight.reshape(original_shape)

    return shuffled_weight


def pose_shuffling(epos_weigh, shuffling_order):
    # (1, 197, 384) -> CLS TOKEN(1, 1, 384) + Other(1, 196, 384)

    # Encrypt "Other" 1~196
    processing_scope = epos_weigh[:, 1:, :]
    processing_scope = processing_scope[:, shuffling_order, :]
    epos_weigh[:, 1:, :] = processing_scope

    return epos_weigh


def main():
    # initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    key_dict = np.load("key_dicts/key-16-random.npy", allow_pickle=True).item()
    pre_process = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])
    ])

    # create a lightweight vit for testing
    model = timm.create_model("vit_small_patch16_224", pretrained=True).to(device)

    # model encryption
    pe_weight, pose_weight = extracting_weight_from_model(model)
    shuffled_pe_weight = pe_shuffling(pe_weight, key_dict['pe_key'])
    shuffled_pose_weight = pose_shuffling(pose_weight, key_dict['pos_key'])
    model = reloading_weight_to_model(model, shuffled_pe_weight, shuffled_pose_weight)

    # loading encrypted img
    testing_img = Image.open('sample_encrypted.png')
    img_tensor = pre_process(testing_img).to(device)
    img_tensor = img_tensor.unsqueeze(0)

    # classification
    model.eval()
    with torch.no_grad():
        out = model(img_tensor)
        print(f"inference result: class {out.argmax(axis=1).item()}")


if __name__ == '__main__':
    main()