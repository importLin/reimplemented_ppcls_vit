from image_encryption import *
import tqdm
import os


def main():
    block_size = 16
    key_dict = np.load(f"key_dicts/key-{block_size}-random.npy", allow_pickle=True).item()
    dataset_root = "./imgs/testing_set_plain"
    saving_root = "./imgs/testing_set_encrypted"

    if os.path.exists(saving_root) is False:
        os.makedirs(saving_root)

    img_name_list = os.listdir(dataset_root)

    # encryption
    for i in tqdm.tqdm(range(len(img_name_list))):
        img_path = os.path.join(dataset_root, img_name_list[i])
        destination_path = os.path.join(saving_root, img_name_list[i])
        img = loading_img_cv2format(img_path)

        block_group = block_division(img, block_size)
        pos_shuffled_block_group = block_pos_shuffling(block_group, key_dict['pos_key'])
        mix_shuffled_block_group = block_pix_shuffling(pos_shuffled_block_group, key_dict['pe_key'])
        encrypted_img = block_integration(mix_shuffled_block_group)

        # save the encrypted img
        encrypted_img = encrypted_img.transpose(1, 2, 0)
        encrypted_img = cv2.cvtColor(encrypted_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(destination_path, encrypted_img)


if __name__ == '__main__':
    main()