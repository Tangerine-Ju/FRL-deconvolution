'''
实验显微成像情况下的数字重聚焦
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
parser.add_argument("--input_dir",  default=r'..\Fig3\5um\data', help="path to folder containing train images")
parser.add_argument("--output_dir",  default=r'.\results', help="where to put output files")
parser.add_argument("--mode",  default="train", choices=["train", "test"])
parser.add_argument("--max_epochs", default=2000, type=int, help="number of training epochs")
parser.add_argument("--lr",  default=0.004, type=float, help="initial learning rate for adam")
a = parser.parse_args()

# print('----------------- starting training --------------------')
if __name__ == '__main__':
    multiprocessing.freeze_support()

    for k, v in a._get_kwargs():
        print(k, "=", v)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    img_b = skimage.io.imread(a.input_dir + '\Object_B.png') ## 参考的离焦图
    img_b1 = img_b[:, :, 0]# 非配对的清晰图
    img_b2 = FD.Transform(img_b1)
    img_bts = torch.tensor(img_b2, dtype=torch.float32).to(device)

    img_c = skimage.io.imread(a.input_dir + '\Defocused_A.png') ## 输入的离焦图
    img_c1 = img_c[:, :, 0]
    img_c2 = FD.Transform(img_c1)
    img_pro1 = torch.tensor(img_c2, dtype=torch.float32).to(device)

    img_d = skimage.io.imread(a.input_dir + '\Defocused_B.png')  ##参考图的标签
    img_d1 = img_d[:, :, 0]
    img_d2 = FD.Transform(img_d1)
    img_pro2 = torch.tensor(img_d2, dtype=torch.float32).to(device)

    x_net = DN.Net()
    x_net.to(device)
    optimizer2 = torch.optim.Adam(x_net.parameters(), lr=a.lr)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer2, gamma=0.7)

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

    out = out_x[0].cpu().detach().numpy()
    cv2.imwrite(a.output_dir + r'/out.png', out[0] * 255)









