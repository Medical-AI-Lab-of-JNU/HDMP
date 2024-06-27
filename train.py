import os

import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from initpack import *
import torch.backends.cudnn as cudnn
from model import pregruuet
from dataset import meta_data
from utils import *
import argparse
import torch


def train(args):
    set_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(f"snapshots/{args.k_shot}shot-{args.target_task}target"):
        os.makedirs(f"snapshots/{args.k_shot}shot-{args.target_task}target")
    for train_epoch in args.train_epoch:
        ###### Load Model ######
        model = pregruuet(args, device).to(device)
        model.train()
        print('###### Load data ######')
        make_data = meta_data
        train_dataset, test_dataset = make_data(args)
        trainloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,  # 4
            shuffle=True,
            num_workers=1,  # 1
            pin_memory=False,  # True load data while training gpu
            drop_last=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001, amsgrad=False)
        iter_n_train = len(trainloader)
        n_slices = args.slices
        for i_epoch in range(train_epoch):
            for i_iter, (sample, s_init_seed) in enumerate(trainloader):
                loss_per_volume = 0.0
                optimizer.zero_grad()
                s_T0_x = sample['s_T0_x'].to(device)
                s_T1_x = sample['s_T1_x'].to(device)
                s_T2_x = sample['s_T2_x'].to(device)
                s_y = sample['s_y'].to(device)

                q_T0_x = sample['q_T0_x'].to(device)
                q_T1_x = sample['q_T1_x'].to(device)
                q_T2_x = sample['q_T2_x'].to(device)
                q_y = sample['q_y'].to(device)
                s_init_seed = s_init_seed.cuda(non_blocking=True)
                preds = model(s_T0_x, s_T1_x, s_T2_x, s_y, q_T0_x, q_T1_x, q_T2_x, s_seed=s_init_seed)
                for frame_id in range(n_slices):
                    q_yi = q_y[:, frame_id, :, :, :]  # [B,1,256,256]
                    yhati = preds[frame_id]
                    loss = dice_loss(yhati, q_yi)
                    loss_per_volume += loss
                loss_per_volume.backward()
                optimizer.step()
                print(f"step[{i_epoch}]/[{train_epoch}][{i_iter}]/[{iter_n_train}],volume_loss:{loss_per_volume}")
                # loss_save.append(loss_epoch / iter_n_train)

            if i_epoch > 20:
                print('===> Saving models...')
                torch.save(model.state_dict(), f'snapshots/{args.k_shot}shot-{args.target_task}target/{i_epoch}.pth')

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--mode', type=str, help='train or test', default='train')
    # Dataset options
    argparse.add_argument('--data_dir', type=str, help='Dataset path used to train models',
                          default='./DCE_MRI_1')
    argparse.add_argument('--img_size', type=tuple, help='size of images', default=(256, 256))
    argparse.add_argument('--slices', type=int, help='number of slices', default=60)
    # Task options
    argparse.add_argument('--target_task', type=int, help='Number of target_task', default=0)
    argparse.add_argument('--k_shot', type=int, help='Number of images in support set', default=1)
    # Model
    argparse.add_argument('--n_layer', type=int, help='Number of PkGru layer', default=1)
    argparse.add_argument('--max_sp', type=int, help='Number of max superpixels', default=4)

    # Training options
    argparse.add_argument('--batch_size', type=int, help='Number of tasks in one batch', default=2)
    argparse.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparse.add_argument('--train_epoch', type=tuple, help='Total update steps for each epoch',
                          default=range(200, 220, 20))
    # Others
    argparse.add_argument('--seed', type=int, help='seed of random', default=1234)
    argparse.add_argument('--train_iter', type=int, help='The number of iterations', default=10)
    argparse.add_argument('--eval_iter', type=int, help='The number of iterations', default=5)
    argps = argparse.parse_args()
    for i in range(4):
        argps.target_task = i
        train(argps)
