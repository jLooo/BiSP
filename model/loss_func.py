import cv2
import numpy as np
import torch

def calculate_ssim(image1, image2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    ssim_all = []
    one = []
    for i in range(image1.shape[0]):
        if not image1[i].shape == image2[i].shape:
            raise ValueError('Input images must have the same dimensions.')
        h, w = 256, 256 # img1.shape[2:4]
        img1 = np.transpose(image1[i, border:h-border, border:w-border], (1,2,0))
        img2 = np.transpose(image2[i, border:h-border, border:w-border], (1,2,0))

        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for j in range(3):
                    ssims.append(ssim(img1[:,:,j], img2[:,:,j]))

                ssim_all.append(np.array(ssims).mean())
    return torch.tensor(np.array(ssim_all).mean())


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

