from __future__ import print_function
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class LinearClassifierResNet(nn.Module):
    def __init__(self, layer=6,n_classes=1000):
        super(LinearClassifierResNet, self).__init__()
        self.layer = layer
        nChannels = 512
        self.classifier = nn.Linear(nChannels, n_classes)
        self.initilize()
    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)
    def forward(self, x):
        if self.layer < 6:
            avg_pool = nn.AvgPool2d((x.shape[2], x.shape[3]))
            x = avg_pool(x).squeeze()
        return self.classifier(x)