import torch
import torch.nn as nn

class vgg16(nn.Module):
    def __init__(self, num_classes: int | tuple, dropout: float = 0.5):
        super(vgg16, self).__init__()

        self.classifier_num = num_classes
        self.features = nn.Sequential(self.convLayer(3,64),
                                      self.convLayer(64,64),
                                      nn.MaxPool2d((2,2),(2,2)),
                                      self.convLayer(64, 128),
                                      self.convLayer(128,128),
                                      nn.MaxPool2d((2,2),(2,2)),
                                      self.convLayer(128,256),
                                      self.convLayer(256,256),
                                      self.convLayer(256,256),
                                      nn.MaxPool2d((2,2),(2,2)),
                                      self.convLayer(256,512),
                                      self.convLayer(512,512),
                                      self.convLayer(512,512),
                                      nn.MaxPool2d((2,2),(2,2)),
                                      self.convLayer(512,512),
                                      self.convLayer(512,512),
                                      self.convLayer(512,512),
                                      nn.MaxPool2d((2,2),(2,2))
                                    )
        self.avgPool = nn.AdaptiveAvgPool2d((7,7))
        #print(isinstance(self.classifier_num, tuple))
        if isinstance(self.classifier_num, tuple):
            self.classifier1 = self.classifier(num_classes[0], dropout)
            self.classifier2 = self.classifier(num_classes[1], dropout)
        else:
            self.classifier1 = self.classifier(num_classes, dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.features(x)
        x = self.avgPool(x)
        x = torch.flatten(x,1)
        if isinstance(self.classifier_num, tuple):
            class_1 = self.classifier1(x)
            class_2 = self.classifier2(x)
            return class_1, class_2
        else:
            x = self.classifier1(x)
            return x

    def convLayer(self, layer_in: int, layer_out: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(layer_in,layer_out,3,1, padding="same"),
                      nn.BatchNorm2d(layer_out),
                      nn.ReLU())
    
    def classifier(self, num_classes: int, dropout:float) -> nn.Sequential:
        return nn.Sequential(self.linLayer(7*7*512, 4096, dropout),
                      self.linLayer(4096, 1024, dropout),
                      nn.Linear(1024, num_classes))

    def linLayer(self, layer_in: int, layer_out: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(nn.Linear(layer_in, layer_out),
                      nn.ReLU(),
                      nn.Dropout(dropout))
