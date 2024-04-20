import numpy as np
import torch
import torch.nn as nn
import cv2
import math
import copy
from sklearn.metrics import roc_auc_score
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):

    return 10 * math.log10(1 / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re

def point_score(outputs, imgs):
    
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score
    
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc

def score_sum(list1, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]))

    return list_result

def add_noise(noise_type, noise_scale, frame):
    if noise_type == 'gauss':
        gauss = np.random.normal(0, noise_scale, frame.size).reshape(frame.shape[0], frame.shape[1], frame.shape[2])
        noise_frames = frame + gauss

    return noise_frames.astype('float32')

def patch_max_mse(diff_map_appe, patches=3, size=16, step=4, is_multi=False):
    assert size % step == 0

    b_size = diff_map_appe.shape[0]
    max_mean = np.zeros([b_size, patches])

    # sliding window
    for i in range(0, diff_map_appe.shape[-2] - size, step):
        for j in range(0, diff_map_appe.shape[-1] - size, step):

            curr_mean = np.mean(diff_map_appe[..., i:i + size, j:j + size], axis=(1, 2, 3))
            for b in range(b_size):
                for n in range(patches):
                    if curr_mean[b] > max_mean[b, n]:
                        max_mean[b, n + 1:] = max_mean[b, n:-1]
                        max_mean[b, n] = curr_mean[b]
                        break
    return max_mean[:, 0]  #

def multi_patch_max_mse(diff_map_appe):
    mse_32 = patch_max_mse(diff_map_appe, patches=3, size=32, step=8, is_multi=False)
    mse_64 = patch_max_mse(diff_map_appe, patches=3, size=64, step=16, is_multi=False)
    mse_128 = patch_max_mse(diff_map_appe, patches=3, size=128, step=32, is_multi=False)
    return mse_32,mse_64,mse_128

def normalize_score_list_gel(score):           # normalize in each video and save in list form
    anomaly_score_list = list()
    for i in range(len(score)):
        anomaly_score_list.append(normalize_score_clip(score[i], np.max(score), np.min(score)))
    return anomaly_score_list

def normalize_score_clip(score, max_score, min_score):
    return ((score - min_score) / (max_score-min_score))

def multi_future_frames_to_scores(input):
    output = cv2.GaussianBlur(input, (5, 0), 10)
    return output