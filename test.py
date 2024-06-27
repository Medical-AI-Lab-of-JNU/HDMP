import argparse
import os
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils import *
import torch.backends.cudnn as cudnn
from model import pregruuet
from dataset import meta_data
import argparse


def tes(args):
    set_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_slices = args.slices
    f1 = open(f'results/{args.k_shot}shot-{args.target_task}target/text.txt', 'w')
    max_dice_score = 0
    max_dice_epoch = 0
    for train_epoch in args.train_epoch:
        snapshot = f'{train_epoch}'
        if not os.path.exists(f'results/{args.k_shot}shot-{args.target_task}target/{snapshot}'):
            os.makedirs(f'results/{args.k_shot}shot-{args.target_task}target/{snapshot}')
        f = open(f'results/{args.k_shot}shot-{args.target_task}target/{snapshot}.txt', 'w')
        print('###### Load Model ######')
        model = pregruuet(args, device).to(device)
        model.eval()
        model_weight = torch.load(f'snapshots/{args.k_shot}shot-{args.target_task}target/{snapshot}.pth')
        model.load_state_dict(model_weight)
        print('###### Load data ######')
        make_data = meta_data
        _, test_dataset = make_data(args)
        testloader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False,  # True load data while training gpu
            drop_last=False
        )
        iter_n_test = len(testloader)
        with torch.no_grad():
            dice_scores = []
            for i_iter, (sample, s_init_seed) in enumerate(testloader):
                s_T0_x = sample['s_T0_x'].to(device)
                s_T1_x = sample['s_T1_x'].to(device)
                s_T2_x = sample['s_T2_x'].to(device)
                s_y = sample['s_y'].to(device)

                q_T0_x = sample['q_T0_x'].to(device)
                q_T1_x = sample['q_T1_x'].to(device)
                q_T2_x = sample['q_T2_x'].to(device)
                q_y = sample['q_y'].type(torch.LongTensor).to(device)
                s_init_seed = s_init_seed.cuda(non_blocking=True)
                preds = model(s_T0_x, s_T1_x, s_T2_x, s_y, q_T0_x, q_T1_x, q_T2_x, s_seed=s_init_seed)  # [slices_num,B,1,256,256]
                volume_dice = []
                for frame_id in range(n_slices):
                    q_xi = q_T2_x[:, frame_id, :, :, :]
                    q_yi = q_y[:, frame_id, :, :, :]  # [B,1,256,256]
                    yhati = preds[frame_id]
                    volume_dice.append(dice_score(yhati, q_yi).cpu().numpy())
                    pred = yhati[0].squeeze(0).cpu().numpy()
                    img = q_xi[0].squeeze(0).cpu().numpy()
                    label = q_yi[0].squeeze(0).cpu().numpy()
                    # cv2.imwrite(
                    #     f'results/{args.k_shot}shot-{args.target_task}target/{snapshot}/{i_iter}-{frame_id}-img.png',
                    #     img * 255)
                    # cv2.imwrite(
                    #     f'results/{args.k_shot}shot-{args.target_task}target/{snapshot}/{i_iter}-{frame_id}-pred.png',
                    #     pred * 255)
                    # cv2.imwrite(
                    #     f'results/{args.k_shot}shot-{args.target_task}target/{snapshot}/{i_iter}-{frame_id}-label.png',
                    #     label * 255)
                dice_scores.append(np.mean(volume_dice))
                print(f'Test-step{i_iter}/{iter_n_test} , dice_score = {np.mean(volume_dice)}')
                f.write(
                    f'Test-step{i_iter}/{iter_n_test} , dice_score = {np.mean(volume_dice)}\n')
            if np.mean(dice_scores) > max_dice_score:
                max_dice_score = np.mean(dice_scores)
                max_dice_epoch = train_epoch
            print(f'Final Result: train-epoch = {train_epoch},dice_score = {np.mean(dice_scores)}')
            f.write(f'Final Result: train-epoch = {train_epoch},dice_score = {np.mean(dice_scores)}\n')
            f1.write(f'train-epoch = {train_epoch}, dice_score = {np.mean(dice_scores)}\n')
        f.close()
    f1.close()
    print(f"max_dice_score:{max_dice_score}, max_dice_epoch:{max_dice_epoch}")


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--mode', type=str, help='train or test', default='train')
    # Dataset options
    argparse.add_argument('--data_dir', type=str, help='Dataset path used to train models',
                          default='./DCE_MRI_1')
    argparse.add_argument('--img_size', type=tuple, help='size of images', default=(256, 256))
    argparse.add_argument('--slices', type=int, help='number of slices', default=60)
    # Task options
    argparse.add_argument('--target_task', type=int, help='Number of target_task', default=1)
    argparse.add_argument('--k_shot', type=int, help='Number of images in support set', default=1)
    # Model
    argparse.add_argument('--n_layer', type=int, help='Number of PkGru layer', default=1)
    argparse.add_argument('--max_sp', type=int, help='Number of max superpixels', default=4)

    # Training options
    argparse.add_argument('--batch_size', type=int, help='Number of tasks in one batch', default=1)
    argparse.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparse.add_argument('--train_epoch', type=tuple, help='Total update steps for each epoch',
                          default=range(21, 200))
    # Others
    argparse.add_argument('--seed', type=int, help='seed of random', default=1234)
    argparse.add_argument('--train_iter', type=int, help='The number of iterations', default=10)
    argparse.add_argument('--eval_iter', type=int, help='The number of iterations', default=5)
    argps = argparse.parse_args()
    # for i in range(1, 4):
    #     argps.target_task = i
    #     tes(argps)
    tes(argps)