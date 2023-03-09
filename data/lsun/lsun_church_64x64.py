"""
first download the LSUN church 64*64 npy array from kaggle:
https://www.kaggle.com/datasets/ajaykgp12/lsunchurch
then right the script to generate train and validation set
"""

import numpy as np


def cut_into_tra_and_val(npz_path):
    npz_data = np.load(npz_path)
    print(npz_data.shape)

    tra_arr = npz_data[:120000]
    val_arr = npz_data[120000:]

    np.save('lsun_church_64x64_tra.npy', tra_arr)
    print(f"saved {tra_arr.shape} into 'lsun_church_64x64_tra.npy'")
    np.save('lsun_church_64x64_val.npy', val_arr)
    print(f"saved {val_arr.shape} into 'lsun_church_64x64_val.npy'")


def check_tra_and_val(path):
    npz_data = np.load(path)
    print(npz_data.shape)


if __name__ == '__main__':
    cut_into_tra_and_val('./church_outdoor_train_lmdb_color_64.npy')
    check_tra_and_val('lsun_church_64x64_tra.npy')