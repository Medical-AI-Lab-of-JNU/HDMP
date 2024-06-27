import random
import torch
import numpy as np


def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(arr):
    tensor = torch.from_numpy(arr).float()
    return tensor

def random_augment(s_T0_imgs, s_T1_imgs, s_T2_imgs, s_labels, q_T0_imgs, q_T1_imgs, q_T2_imgs, q_labels):
    """
    s_imgs.shape = [shot,slices,C,H,W]
    s_labels.shape = [shot,slices,C,H,W]
    q_imgs.shape = [slices,C,H,W]
    q_labels.shape = [slices,C,H,W]
    """
    ## do random rotation and flip
    k = random.sample([i for i in range(0, 4)], 1)[0]
    s_T0_imgs = np.rot90(s_T0_imgs, k, (3, 4)).copy()
    s_T1_imgs = np.rot90(s_T1_imgs, k, (3, 4)).copy()
    s_T2_imgs = np.rot90(s_T2_imgs, k, (3, 4)).copy()
    s_labels = np.rot90(s_labels, k, (3, 4)).copy()

    q_T0_imgs = np.rot90(q_T0_imgs, k, (2, 3)).copy()
    q_T1_imgs = np.rot90(q_T1_imgs, k, (2, 3)).copy()
    q_T2_imgs = np.rot90(q_T2_imgs, k, (2, 3)).copy()
    q_labels = np.rot90(q_labels, k, (2, 3)).copy()

    if random.random() < 0.5:
        s_T0_imgs = np.flip(s_T0_imgs, 3).copy()
        s_T1_imgs = np.flip(s_T1_imgs, 3).copy()
        s_T2_imgs = np.flip(s_T2_imgs, 3).copy()
        s_labels = np.flip(s_labels, 3).copy()

        q_T0_imgs = np.flip(q_T0_imgs, 2).copy()
        q_T1_imgs = np.flip(q_T1_imgs, 2).copy()
        q_T2_imgs = np.flip(q_T2_imgs, 2).copy()
        q_labels = np.flip(q_labels, 2).copy()

    if random.random() < 0.5:
        s_T0_imgs = np.flip(s_T0_imgs, 4).copy()
        s_T1_imgs = np.flip(s_T1_imgs, 4).copy()
        s_T2_imgs = np.flip(s_T2_imgs, 4).copy()
        s_labels = np.flip(s_labels, 4).copy()

        q_T0_imgs = np.flip(q_T0_imgs, 3).copy()
        q_T1_imgs = np.flip(q_T1_imgs, 3).copy()
        q_T2_imgs = np.flip(q_T2_imgs, 3).copy()
        q_labels = np.flip(q_labels, 3).copy()

    return s_T0_imgs, s_T1_imgs, s_T2_imgs, s_labels, q_T0_imgs, q_T1_imgs, q_T2_imgs, q_labels


def dice_score(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def dice_loss(pred, target):
    return 1.0 - dice_score(pred, target)
