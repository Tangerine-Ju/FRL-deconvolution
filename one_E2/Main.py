'''
PSF模型模拟显微成像情况下的数字重聚焦（保存模型用于相似目标测试）
'''
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import time
import os
import Net as DN
import Function as FD
import multiprocessing
import argparse
import cv2
import pandas as pd
import skimage

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",  default=r'..\Fig11\data', help="path to folder containing train images")
parser.add_argument("--output_dir",  default=r'.\results', help="where to put output files")
parser.add_argument("--mode",  default="train", choices=["train", "test"])
parser.add_argument("--max_epochs", default=2000, type=int, help="number of training epochs")
parser.add_argument("--lr",  default=0.004, type=float, help="initial learning rate for adam")
parser.add_argument("--load_model", default='False', choices=['True', 'False'])
a = parser.parse_args()

# print('----------------- starting training --------------------')
if __name__ == '__main__':
    multiprocessing.freeze_support()

    for k, v in a._get_kwargs():
        print(k, "=", v)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if a.mode == "train":
        fname = "Object_A"  # 输入的清晰图
        f_name = a.input_dir + '/' + fname + '.png'
        img_a = skimage.io.imread(f_name)
        img_a2 = FD.Transform(img_a)
        img_ats = torch.tensor(img_a2, dtype=torch.float32).to(device)
        n_size = img_ats.size()[2]

        img_b = skimage.io.imread(a.input_dir + '/' + 'Object_B.png') # 参考的清晰图
        img_b2 = FD.Transform(img_b)
        img_bts = torch.tensor(img_b2, dtype=torch.float32).to(device)

        img_p = skimage.io.imread(a.input_dir + '/PSF_BW_01.png')  # 清晰图用的卷积核
        img_p2 = FD.Transform2(img_p)
        img_pts = torch.tensor(img_p2, dtype=torch.float32).to(device)

        x_net = DN.Net()
        if a.load_model == 'True':
            x_net.load_state_dict(torch.load(save_folder + r'/Net.pt'))
            print('-' * 10 + 'model load' + '-' * 10)
        x_net.to(device)
        optimizer2 = torch.optim.Adam(x_net.parameters(), lr=a.lr)
        exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer2, gamma=0.7)

        img_aats = nn.functional.pad(img_ats, (12, 12, 12, 12), mode='circular')
        img_pro1 = nn.functional.conv2d(img_aats, img_pts, padding=0, bias=None)
        img_bbts = nn.functional.pad(img_bts, (12, 12, 12, 12), mode='circular')
        img_pro2 = nn.functional.conv2d(img_bbts, img_pts, padding=0, bias=None)

        if not os.path.exists(a.output_dir):
            os.makedirs(a.output_dir)

        pro_1 = img_pro1[0].cpu().detach().numpy()
        pro_2 = img_pro2[0].cpu().detach().numpy()
        cv2.imwrite(a.output_dir + r'/Defocused_A' + '.png', pro_1[0] * 255)
        cv2.imwrite(a.output_dir + r'/Defocused_B' + '.png', pro_2[0] * 255)
        m_loss = []
        input = img_pro1
        start = time.time()
        for epoch in range(a.max_epochs):
            x_net.train()
            optimizer2.zero_grad()
            ref, input2 = FD.aug(img_bts, img_pro2, device)
            out_x = x_net(input)
            out_z = x_net(input2)

            loss_func = torch.nn.MSELoss(reduction='mean')
            loss1 = loss_func(out_z, ref)
            ssim = FD.SSIM(window_size=11)
            loss2 = -torch.log((1 + ssim(out_z, ref)) / 2) * 1e-2
            loss3 = DN.Gradient(device)(out_x) * 1e-4
            loss4 = DN.Gradient(device)(out_z) * 2e-4

            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()  # 反向传递
            optimizer2.step()  # 优化梯度

            ite_loss = loss.cpu().detach().numpy()
            m_loss.append(ite_loss)
            dataframe = pd.DataFrame({'loss_train': m_loss})
            dataframe.to_csv(a.output_dir + '/loss.csv', index=False)

            if (epoch + 1) % 400 == 0:
                exp_lr_scheduler.step()
            if (epoch + 1) % 100 == 0:
                waste_time = time.time() - start
                print('-' * 50)
                print('Epoch: {}/{}'.format(epoch + 1, a.max_epochs))
                print("Loss: {:.6f}, LR: {:.6f}, Waste: {:.0f}m {:.0f}s".format(ite_loss,
                optimizer2.state_dict()['param_groups'][0]['lr'], waste_time // 60, waste_time % 60))
                torch.save(x_net.state_dict(), a.output_dir + r'/model.pt')

        out = out_x[0].cpu().detach().numpy()
        cv2.imwrite(a.output_dir + r'/out.png', out[0] * 255)

    if a.mode == "test":
        test_dir = r'..\Fig11\GT'
        filenames = os.listdir(test_dir)
        img_p = skimage.io.imread(a.input_dir + '/PSF_BW_01.png')  # 清晰图用的卷积核
        img_p2 = FD.Transform2(img_p)
        img_pts = torch.tensor(img_p2, dtype=torch.float32).to(device)
        x_net = DN.Net()
        x_net.load_state_dict(torch.load(a.output_dir + r'/model.pt'))
        print('-' * 10 + 'model load' + '-' * 10)
        x_net.to(device)
        x_net.eval()
        start = time.time()
        for filename in filenames:
            name = os.path.join(test_dir, filename)
            img_a = skimage.io.imread(name)
            img_a2 = FD.Transform(img_a)
            img_ats = torch.tensor(img_a2, dtype=torch.float32).to(device)
            img_aats = nn.functional.pad(img_ats, (12, 12, 12, 12), mode='circular')
            img_pro1 = nn.functional.conv2d(img_aats, img_pts, padding=0, bias=None)
            input = img_pro1
            since = time.time()
            with torch.no_grad():
                out_x = x_net(input)
            out = out_x[0].cpu().detach().numpy()
            cv2.imwrite(a.output_dir + '/' + filename, out[0] * 255)
            print('{:.3f}s'.format(time.time() - since))

        waste_time = time.time() - start
        print('{:.3f}s'.format(waste_time))









