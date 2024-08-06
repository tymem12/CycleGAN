import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, stride: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1, bias = True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        return self.conv(x)
    
class Discriminator(nn.Module):
    def __init__(self, in_channels : int = 3, features : list = [64,128,256,512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels= features[0], kernel_size=4,stride = 2, padding = 1, padding_mode = "reflect"),
            nn.LeakyReLU(0.2)

        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride = 1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride = 1, padding = 1, padding_mode="reflect"))  # kernel has dimensions (batch_size, 512, height, width) so we could obtain (batch_size, 1, output_height, output_width)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x)) # we want to get value between 0 and 1 for each segment (30 x 30 ) to know if it is real on not based on the discriminator
    

def test():
# the output of the test should be (5, 1 , 30, 30)

    x = torch.randn((5,3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()
    
