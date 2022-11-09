import numpy as np
import torch


def torch_four_derangement(x):
    # Compute derangements
    d1 = torch.arange(x)
    d2 = torch.randperm(x)
    while (d1 == d2).sum().item() > 0:
        d2 = torch.randperm(x)
    d3 = torch.randperm(x)
    while (d1 == d3).sum().item() > 0 or (d2 == d3).sum().item() > 0:
        d3 = torch.randperm(x)
    d4 = torch.randperm(x)
    while (d1 == d4).sum().item() > 0 or (d2 == d4).sum().item() > 0 or (d3 == d4).sum().item() > 0:
        d4 = torch.randperm(x)
    return d1, d2, d3, d4


def random_crop(X, y, crop_size):
    # X.shape and y.shape: (C, H, W)
    _, H, W = X.size() 
    crop_h, crop_w = crop_size
    assert crop_h <= H and crop_w <= W
    x1 = np.random.randint(0, H-crop_h)
    y1 = np.random.randint(0, W-crop_w)
    return X[:, x1:x1+crop_h, y1:y1+crop_w], y[:, x1:x1+crop_h, y1:y1+crop_w]


def create_mosaic(X, y, center_coord):
    B, C, H, W = X.size()
    assert B == y.size(0) == 4
    # formatted as: x1, y1, x2, y2, imsize
    img1_coords = (0, 0, center_coord[0], center_coord[1], (center_coord[0], center_coord[1]))
    img2_coords = (center_coord[0], 0, H, center_coord[1], (H - center_coord[0], center_coord[1]))
    img3_coords = (0, center_coord[1], center_coord[0], W, (center_coord[0], W - center_coord[1]))
    img4_coords = (center_coord[0], center_coord[1], H, W, (H - center_coord[0], W - center_coord[1]))
    X_mosaic = X[0].clone()
    y_mosaic = y[0].clone()
    mosaic_list = []
    for idx, img_coord in enumerate([img1_coords, img2_coords, img3_coords, img4_coords]):
        x1, y1, x2, y2, imsize = img_coord
        X_mosaic[:, x1:x2, y1:y2], y_mosaic[:, x1:x2, y1:y2] = random_crop(X[idx], y[idx], imsize)
    return X_mosaic, y_mosaic


def apply_mosaic(X, y, center_ratio_range=(0.25, 0.75)):
    """
    Implementation of Mosaic augmentation for segmentation. 

    Args:
    - X (torch.Tensor): batch of images
    - y (torch.Tensor): batch of segmentation masks
    **Note: batch size PER PROCESS must be at least 4
    - center_ratio_range (tuple, list): tuple or list of 2 floats between 0 and 1,
        which indicate where the center (intersection) of the 4 images will be
    """
    B, C, H, W = X.size()
    assert H == y.size(2) and W == y.size(3)
    # Determine center coordinate
    a, b = center_ratio_range
    cx, cy = np.random.uniform(a, b, B), np.random.uniform(a, b, B)
    cx = (cx * H).astype("int")
    cy = (cy * W).astype("int")
    # 4 images will be combined into one 
    # Each image will be assigned to one of: top left, top right, bot left, bot right
    indices = torch.stack(torch_four_derangement(B)).swapaxes(0, 1)
    X_mosaic, y_mosaic = X.clone(), y.clone()
    for ii, inds in enumerate(indices):
        X_mosaic[ii], y_mosaic[ii] = create_mosaic(X[inds], y[inds], (cx[ii], cy[ii]))
    return X_mosaic, y_mosaic 


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img0 = np.zeros((1,3,512,512))
    img0[...] = 255
    img1 = np.zeros((1,3,512,512))
    img1[:,0] = 255
    img2 = np.zeros((1,3,512,512))
    img2[:,1] = 255
    img3 = np.zeros((1,3,512,512))
    img3[:,2] = 255
    X = np.concatenate([img0,img1,img2,img3], axis=0)
    y = X.copy()
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    Xm, ym = apply_mosaic(X, y)
    Xm = Xm.numpy().astype("uint8").transpose(0, 2, 3, 1)
    ym = ym.numpy().astype("uint8").transpose(0, 2, 3, 1)
    for i in range(len(Xm)):
        plt.subplot(1,2,1)
        plt.imshow(Xm[i])
        plt.subplot(1,2,2)
        plt.imshow(ym[i])
        plt.show()

