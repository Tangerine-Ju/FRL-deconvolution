import torch
import torch.nn as nn
import torch.nn.functional as F
import Function as FD

# RLMNet
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=4,
                      kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4),
            nn.Softplus()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=4,
                      kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=4,
                      kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4),
            nn.Softplus()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=4,
                      kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4),
            nn.Softplus()
        )
        self.BN = nn.BatchNorm2d(1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=25, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=4,
                      kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4),
            nn.Softplus()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=4,
                      kernel_size=25, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=4,
                      kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4),
            nn.Softplus()
        )

    def forward(self, input):
        x1_p = nn.functional.pad(input, (2, 2, 2, 2), mode='circular')
        x1 = self.conv1(x1_p)
        x2 = torch.cat([input, x1], dim=1)
        x2_p = nn.functional.pad(x2, (2, 2, 2, 2), mode='circular')
        x3 = self.conv2(x2_p)
        back = self.conv3(x1_p)
        x4 = x3 + back
        FP = torch.mean(x4, dim=1, keepdim=True)
        DV = self.BN(torch.div(input, FP))
        DV_p = nn.functional.pad(DV, (13, 13, 13, 13), mode='circular')
        x5 = self.conv4(DV_p)
        x6 = torch.cat([DV, x5], dim=1)
        x6_p = nn.functional.pad(x6, (13, 13, 13, 13), mode='circular')
        x7 = self.conv5(x6_p)
        BP = torch.mean(x7, dim=1, keepdim=True)
        E1 = torch.mul(input, BP)
        return E1

class Gradient(nn.Module):
  def __init__(self, device):
    super(Gradient, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)

    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).to(device)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).to(device)

  def forward(self, x):
    grad_x = F.conv2d(x, self.weight_x, padding=1)
    grad_y = F.conv2d(x, self.weight_y, padding=1)
    # grad = torch.sqrt(torch.mean(torch.square(grad_x) + torch.square(grad_y)))
    grad = torch.mean(torch.abs(grad_x) + torch.abs(grad_y))
    return grad











