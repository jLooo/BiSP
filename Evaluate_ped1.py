import os

import torch.utils.data as data
from torch.autograd import Variable
from collections import OrderedDict
from model.utils import DataLoader
from utils import *
import random
import glob
import scipy.signal as signal
import argparse
from model.MGM_ped1 import *


def Eval(model=None):
    parser = argparse.ArgumentParser(description="BiSP")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
    parser.add_argument('--t_length', type=int, default=7, help='length of the frame sequences')
    parser.add_argument('--num_workers_test', type=int, default=2, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='ped1', help='type of dataset: ped1')
    parser.add_argument('--dataset_path', type=str, default='D:/Datasets', help='directory of data')
    parser.add_argument('--model_dir', type=str, default='./modelzoo/best_model_ped1.pth', help='directory of model')
    parser.add_argument('--seed', type=int, default=2023, help='directory of log')

    args = parser.parse_args()
    # set_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    else:
        gpus = ""
        for i in range(len(args.gpus)):
            gpus = gpus + args.gpus[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

    torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

    test_folder = args.dataset_path + "/" + args.dataset_type + "/testing/frames"

    # Loading dataset
    test_dataset = DataLoader(test_folder, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=args.h, resize_width=args.w, train=False, time_step=args.t_length)

    test_size = len(test_dataset)

    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    if model is None:  # if not training, we give a exist model and params path
        model = MGM_ped1(args.t_length // 2, args.c)
        try:
            model.load_state_dict(torch.load(args.model_dir).state_dict(), strict=False)
        except:
            model.load_state_dict(torch.load(args.model_dir), strict=False)
        model.cuda()
        
    # Loading the pretrained model
    # model = torch.load(args.model_dir)
    # model.cuda()
    
    labels = np.load('./data/frame_labels_' + args.dataset_type + '.npy')
    if labels.ndim == 1:
        labels = labels[np.newaxis, :]
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1].split('\\')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.tif'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    labels_list = []
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}

    print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1].split('\\')[-1]
        if args.method == 'pred':
            labels_list = np.append(labels_list,
                                    labels[0][3 + label_length:videos[video_name]['length'] + label_length - 3])
        else:
            labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length'] + label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1].split('\\')[-1]]['length']

    model.eval()
    with torch.no_grad():
        for k, (f_imgs, b_imgs, mid_img) in enumerate(test_batch):
            f_imgs = Variable(f_imgs).cuda()
            b_imgs = Variable(b_imgs).cuda()
            mid_img = Variable(mid_img).cuda()

            if k == label_length - 6 * (video_num + 1):
                video_num += 1
                label_length += videos[videos_list[video_num].split('/')[-1].split('\\')[-1]]['length']


            if args.method == 'pred':
                outputs = model.forward(f_imgs, b_imgs)
                output = (outputs['f2b'] + outputs['b2f']) / 2
                mse_imgs = (((output + 1) / 2) - ((mid_img + 1) / 2)) ** 2

            mse_32, mse_64, mse_128 = multi_patch_max_mse(mse_imgs.cpu().detach().numpy())
            mse_multi = mse_32 + mse_64 + mse_128
            psnr_list[videos_list[video_num].split('/')[-1].split('\\')[-1]].append(psnr(mse_multi))
        psnr_multi_list = []

        for video in sorted(videos_list):
            video_name = video.split('/')[-1].split('\\')[-1]
            psnr_multi_list.extend(
                multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_list[video_name]))))
        psnr_multi_list = np.asarray(psnr_multi_list)
        accuracy = AUC(psnr_multi_list, np.expand_dims(1 - labels_list, 0))

    print('The result of ', args.dataset_type)
    # np.save('./data/psnr/ped1/{}_{}.npy'.format(args.dataset_type, str(accuracy)), psnr_multi_list)
    print('AUC: ', accuracy * 100, '%')
    return accuracy


if __name__ == '__main__':
    Eval()
