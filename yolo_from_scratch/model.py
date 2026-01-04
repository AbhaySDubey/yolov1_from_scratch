import torch
import torch.nn as nn

architecture_config = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    (7,64,2,3),
    "M",
    (3,192,1,1),
    "M",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    "M",
    # List: tuples of conv. layer specifications, last int. represents num_repeats
    [(1,256,1,0), (3,512,1,1), 4],
    (1,512,1,0),
    (3,1024,1,1),
    "M",
    [(1,512,1,0), (3,1024,1,1), 2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyreul = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyreul(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        assert x.shape[1] == self.in_channels, (
            f"Input has {x.shape[1]} channels, but model expects {self.in_channels}"
        )
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    # creating the darknet (as proposed by Joseph Redmon)
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers.append(CNNBlock(
                    in_channels=in_channels,
                    kernel_size=x[0],
                    out_channels=x[1],
                    stride=x[2],
                    padding=x[3]
                ))
                in_channels = x[1]
            elif type(x) == str:
                layers.append(nn.MaxPool2d(
                    kernel_size=2, stride=2
                ))
            elif type(x) == list:
                num_repeats = x[-1]

                # i know there's just 2 conv. layers in the architecture definition above
                # but it's nice to have a more standard approach that handles variable number of conv layer stacks automatically
                # if i decide to experiment (FAFO ;)) later 
                for _ in range(num_repeats):
                    for i in range(len(x)-1):
                        conv = x[i]
                        layers.append(CNNBlock(
                            in_channels=in_channels,
                            kernel_size=conv[0],
                            out_channels=conv[1],
                            stride=conv[2],
                            padding=conv[3]
                        ))
                        # the out_channels for the final conv layer would act as the in_channels for the next time we repeat the loop
                        in_channels = conv[1]

        return nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),

            # welp, for now, i'm keeping this to 512 and i'll maybe change it later to 496, but as per the original paper this should be 4096
            
            # as per the original paper, this should be
            # nn.Linear(1024*S*S, 4096),
            # nn.Dropout(0.0),
            # nn.LeakyReLU(0.1),
            # nn.Linear(4096, S*S*(C+(B*5))), # (S,S,30) since, C+B*5 = 30 (C=20,B=2)
            
            nn.Linear(1024*S*S, 512),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(512, S*S*(C+(B*5))), # (S,S,30) since, C+B*5 = 30 (C=20,B=2)
        )
    
def test(S=7, B=2, C=20):
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2,3,448,448)) # shape => (batch_size,channels,height,width)

    # print(model(x).shape)

test()