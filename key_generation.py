import numpy as np
import os


class KeyGenerator:
    def __init__(self,
                 patch_size=16,
                 saving_root='key_dicts',
                 key_seed=None):

        self.patch_size = patch_size
        self.input_size = 224
        self.saving_root = saving_root
        self.seed = key_seed

        if key_seed:
            np.random.seed(key_seed)
        else:
            print("Testing mode : Generating sequences without shuffling")

        # key used for patch embedding's weight E and img pix shuffling
        self.pe_key = self.generate_pe_key()

        # key used for position embedding's weight Epos and img pos shuffling
        self.pos_key = self.generate_pos_key()

    def generate_pe_key(self):
        key_length = 3 * self.patch_size ** 2
        key = np.arange(key_length)
        if self.seed:
            key = np.random.permutation(key_length)

        return key

    def generate_pos_key(self):
        key_length = (self.input_size // self.patch_size) ** 2
        key = np.arange(key_length)
        if self.seed:
            key = np.random.permutation(key_length)

        return key

    def key_save(self):
        key_dict = {
            'pe_key': self.pe_key,
            'pos_key': self.pos_key
        }

        if os.path.exists(self.saving_root) is False:
            print("Creating saving Folder..")
            os.makedirs(self.saving_root)
        if self.seed:
            filename = f"key-{self.patch_size}-random.npy"
        else:
            filename = f"key-{self.patch_size}-plain.npy"

        saving_path = os.path.join(self.saving_root, filename)
        np.save(saving_path, key_dict)


def show_keys(key_path):
    key_dict = np.load(key_path, allow_pickle=True).item()
    for k in key_dict.keys():
        print(f"{k}: {key_dict[k].shape}")


def main():
    seed = 100
    test_key = KeyGenerator(key_seed=seed)
    test_key.key_save()


if __name__ == '__main__':
    main()
