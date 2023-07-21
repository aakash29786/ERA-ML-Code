import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.1
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # CONVOLUTION BLOCK 1 input 32/1/1
    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),

        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout(dropout_value)
    ) # output_size = 32/3/1


    self.convblock2 = nn.Sequential(
        nn.Conv2d(in_channels=32, groups=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
        nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1), padding=0, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)
    ) # output_size = 32/5/1

    # TRANSITION BLOCK 1
    self.convblock3 = nn.Sequential(
        nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(3,3),
            padding=2,
            stride=2,
            dilation=2,
            bias=False),
    ) # output_size = 16/7/2

    # CONVOLUTION BLOCK 2
    self.convblock4 = nn.Sequential(
        nn.Conv2d(in_channels=64, groups=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=0, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)
    ) # output_size = 16/11/2

    self.convblock5 = nn.Sequential(
        nn.Conv2d(in_channels=128, groups=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), padding=0, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout(dropout_value)
    ) # output_size = 16/15/2

    # TRANSITION BLOCK 2
    self.convblock6 = nn.Sequential(
        nn.Conv2d(in_channels=128,
                  out_channels=64,
                  kernel_size=(3,3),
                  padding=2,
                  dilation=2,
                  stride=2,
                  bias=False),
    ) # output_size = 8/19/4

    self.convblock7 = nn.Sequential(
        nn.Conv2d(in_channels=64, groups=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(dropout_value)
    ) # output_size = 8/24/4
    self.convblock8 = nn.Sequential(
        nn.Conv2d(in_channels=64, groups = 64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(dropout_value)
    ) # output_size = 8/32/4

    self.convblock9 = nn.Sequential(
        nn.Conv2d(in_channels=64, groups=64,  out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(dropout_value)
    ) # output_size = 6/40/4

    # OUTPUT BLOCK
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=6)
    ) # output_size = 1/64

    self.convblock10 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        # nn.BatchNorm2d(10),
        # nn.ReLU(),
        # nn.Dropout(dropout_value)
    )


    self.dropout = nn.Dropout(dropout_value)

  def forward(self, x):
      x = self.convblock1(x)
      x = self.convblock2(x)
      x = self.convblock3(x)
      x = self.convblock4(x)
      x = self.convblock5(x)
      x = self.convblock6(x)
      x = self.convblock7(x)
      x = self.convblock8(x)
      x = self.convblock9(x)
      x = self.gap(x)
      x = self.convblock10(x)

      x = x.view(-1, 10)
      return F.log_softmax(x, dim=-1)