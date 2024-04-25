import os
import sys
import time

import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from model.utils import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import argparse
from model.MGM_ped2 import *
from model.loss_func import *
import Evaluate as Evaluate

def main():
    parser = argparse.ArgumentParser(description="BiSP")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
    parser.add_argument('--t_length', type=int, default=6, help='length of the frame sequences')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
    parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
    parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
    parser.add_argument('--seed', type=int, default=1111, help='directory of log')
    parser.add_argument('--model_dir', type=str, default='', help='directory of model')
    parser.add_argument('--model_continue', type=bool, default=False, help='reload parameters')

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

    train_folder = args.dataset_path + "/" + args.dataset_type + "/training/frames"

    # Loading dataset
    train_dataset = DataLoader(train_folder, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length)

    train_size = len(train_dataset)

    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, drop_last=True)

    # Model setting
    assert args.method == 'pred' or args.method == 'recon', 'Wrong task name'
    if args.method == 'pred':
        model = MGM_ped2(args.t_length//2, args.c)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # , eta_min=1e-4


    model.cuda()
    # Report the training process
    log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    orig_stdout = sys.stdout
    f = open(os.path.join(log_dir, 'log.txt'), 'w')
    sys.stdout = f
    if args.model_continue:
        print(
            'batch_size: {}, lr: {}, dataset_type: {}, model_continue: {}, model_dir: {}'.format
            (args.batch_size, args.lr, args.dataset_type, args.model_continue, args.model_dir))
    else:
        print(
            'batch_size: {}, lr: {}, dataset_type: {}, model_continue: {}'.format(
                args.batch_size, args.lr, args.dataset_type, args.model_continue))
    loss_func_mse = nn.MSELoss(reduction='none')

    # Training

    if args.model_continue:
        model.load_state_dict(torch.load(args.model_dir).state_dict())
    early_stop = {'idx': 0, 'best_eval_auc': 0}
    for epoch in tqdm(range(args.epochs)):
        model.train()

        start = time.time()
        for j, (f_imgs, b_imgs) in enumerate(train_batch):
            f_imgs = Variable(f_imgs).cuda()
            b_imgs = Variable(b_imgs).cuda()

            outputs = model.forward(f_imgs, b_imgs)

            optimizer.zero_grad()

            loss_pixel_f2b = torch.mean(loss_func_mse(outputs['f2b'], b_imgs[:, 0:3]))
            loss_pixel_b2f = torch.mean(loss_func_mse(outputs['b2f'], f_imgs[:, 0:3]))
            loss_ssim = 1 - calculate_ssim(outputs['f2b'].cpu().detach().numpy(), outputs['b2f'].cpu().detach().numpy()).cuda()
            loss = loss_pixel_f2b + loss_pixel_b2f + loss_ssim
            loss.backward(retain_graph=True)
            optimizer.step()

        scheduler.step()

        print('----------------------------------------')
        print('Epoch:', epoch + 1)
        if args.method == 'pred':
            print('Loss:  {:.6f} / F2B: {:.6f} / B2F: {:.6f} /  SSIM: {:.6f}'.format(
                loss.item(), loss_pixel_f2b.item(), loss_pixel_b2f.item(), loss_ssim.item()), flush=True)

        print('----------------------------------------')

        score = Evaluate.Eval(model=model)
        if score > early_stop['best_eval_auc']:
            early_stop['best_eval_auc'] = score
            early_stop['idx'] = 0
            torch.save(model, os.path.join(log_dir, 'model_.pth'))
        else:
            early_stop['idx'] += 1
            print('Score drop! Model not saved')

        print('With {} epochs, auc score is: {}, best score is: {}, used time: {}'.format(epoch + 1, score,
                                                                                          early_stop['best_eval_auc'],
                                                                                          time.time() - start), flush=True)

    print('Training is finished')

    sys.stdout = orig_stdout
    f.close()


if __name__ == '__main__':
    main()
