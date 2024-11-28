import torch
from torch import nn

din = (3, 48, 48) # rgb channels * 48x48 imgs
dconv1 = 7
dconv2 = 7
dconv3 = 7
dconv4 = 7
dconv5 = 7
dfc1 = 100
dfc2 = 50
dout = 7

class GiuNet(nn.Module):

    def __init__(self, n_actions):
        super(GiuNet, self).__init__()

        # arch (alexnet style)
        # 2*(conv -> response norm -> maxpool)
        # 3*conv -> maxpool
        # 2*(FC with dropout)
        # Linear -> softmax

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, dconv1),
            nn.MaxPool2d()
        )

        self.cnn = nn.Sequential(
            nn.Conv3d(d_e, d_f, k_d, padding='valid'),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(k_d3*d_f, d_o),
            nn.ELU(),
            nn.Linear(d_o, d_o),
            nn.ELU(),
        )
        self.output = nn.Linear(d_o, n_actions)

    def forward(self, x):
        # [B, 6, 5, 5, 5] permute-> [B, 5, 5, 5, 6] fc1-> [B, 5, 5, 5, d_e] permute-> [B, d_e, 5, 5, 5]
        # conv3d-> [B, d_f, k_d, k_d, k_d] flatten-> [B, k_d^3 *d_f] fc2-> [B, d_o] out-> [B, 12] -> out
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = self.fc1(x)
        x = torch.permute(x, (0, 4, 1, 2, 3))
        x = self.cnn(x)
        x = x.reshape(-1, k_d3 *d_f)
        x = self.fc2(x)
        x = self.output(x)
        return xs