import timm
import torch
import evaluate
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torch.utils.data import DataLoader



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data
    pre_process = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])
    ])

    testing_set = DataLoader(
        dataset=CIFAR10(root='./imgs', train=False, download=False),
        batch_size=1,
        shuffle=True,
        drop_last=False
    )

    metric = evaluate.load("accuracy")
    # create a lightweight vit for testing
    model = timm.create_model("vit_small_patch16_224").to(device)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(testing_set):
            img = batch[0].to(device)
            label = batch[1].to(device)

            logits = model(img)
            predictions = logits.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=label)

    metric.compute()