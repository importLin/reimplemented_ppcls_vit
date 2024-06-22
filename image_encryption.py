import cv2
import numpy as np


def loading_img_cv2format(img_path):
    img = cv2.imread(img_path)
    if img.shape[0] != 224:
        raise ValueError('Please check the input image size '
                         'whether or not the size is standard input size of vit')

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img.transpose(2, 0, 1)

    return rgb_img


def block_division(img, block_size):
    channels, height, width = img.shape
    if height != width:
        raise ValueError('Image must be square')

    block_num = height // block_size

    # (3, 224, 224) -> (14, 14, 3, 16, 16)
    block_group = img.reshape(channels, block_num, block_size, block_num, block_size)
    block_group = block_group.transpose(1, 3, 0, 2, 4)
    return block_group


def block_pos_shuffling(block_group, shuffling_order):
    # (14, 14, 3, 16, 16)
    b_idx_h, b_idx_w, channels, b_height, b_width = block_group.shape

    block_group = block_group.reshape(-1, channels, b_height, b_width)
    pos_shuffled_block_group = block_group[shuffling_order, ...]
    pos_shuffled_block_group = pos_shuffled_block_group.reshape(b_idx_h, b_idx_w,
                                                                channels, b_height, b_width)

    return pos_shuffled_block_group


def block_pix_shuffling(block_group, shuffling_order):
    # # (14, 14, 3, 16, 16)
    b_idx_h, b_idx_w, channels, b_height, b_width = block_group.shape

    block_group = block_group.reshape(b_idx_h, b_idx_w, -1)
    pix_shuffled_block_group = block_group[..., shuffling_order]
    pix_shuffled_block_group = pix_shuffled_block_group.reshape(b_idx_h, b_idx_w,
                                                                channels, b_height, b_width)
    return pix_shuffled_block_group


def block_integration(block_group):
    # (14, 14, 3, 16, 16) -> (3, 14, 16, 14, 16) -> (3, 224, 224)
    b_idx_h, b_idx_w, channels, b_height, b_width = block_group.shape

    block_group = block_group.transpose(2, 0, 3, 1, 4)
    block_group = block_group.reshape(-1, b_idx_h * b_height, b_idx_w * b_width)

    return block_group


def main():
    # if you want to test the function, just run the main step by step.

    # initialize key and img
    block_size = 16
    key_dict = np.load("key_dicts/key-16-random.npy", allow_pickle=True).item()
    img = loading_img_cv2format("sample.png")

    # encryption
    block_group = block_division(img, block_size)
    pos_shuffled_block_group = block_pos_shuffling(block_group, key_dict['pos_key'])
    mix_shuffled_block_group = block_pix_shuffling(pos_shuffled_block_group, key_dict['pe_key'])
    encrypted_img = block_integration(mix_shuffled_block_group)

    # visualization using cv2.imshow (optional)
    encrypted_img = encrypted_img.transpose(1, 2, 0)
    encrypted_img = cv2.cvtColor(encrypted_img, cv2.COLOR_RGB2BGR)

    cv2.imshow('win', encrypted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imwrite("sample_encrypted.png", encrypted_img)


if __name__ == '__main__':
    main()