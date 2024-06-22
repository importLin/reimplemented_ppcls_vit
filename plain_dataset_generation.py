import torchvision
import os
from PIL import Image


def main():
    # 学習データを暗号化する場合，trainをTrueにする．
    cifar10 = torchvision.datasets.CIFAR10(root='./imgs', train=False, download=True)
    saving_root = "./imgs/testing_set_plain"
    if os.path.exists(saving_root) is False:
        os.makedirs(saving_root)

    for idx, img_data in enumerate(cifar10):
        img, label = img_data[0], img_data[1]
        # 　BICUBIC補間
        img = img.resize((224, 224), Image.BICUBIC)

        # 個人的は -> 一定の長さをもったid + ラベル + png のほうが処理上に便利
        # 重複したIdが発生しないようにIdの長さを8にしておく（CIFAR10には6万枚画像ある）
        img_name = f'{idx:08d}_{label}.png'
        img.save(os.path.join(saving_root, img_name))


if __name__ == '__main__':
    main()
