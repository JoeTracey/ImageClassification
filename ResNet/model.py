import numpy as np
import torch
from torchsummary import summary




#implimenting ResNet50 through pytorch
class ResNet50(torch.nn.Module):
    def __init__(self, classes=1000):
        super().__init__()

        #Input Layers
        self.c0 = torch.nn.Conv2d(3,64, kernel_size=7, stride = 2, padding = 3)
        self.b0 = torch.nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.r0 = torch.nn.ReLU()
        self.m0 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        #Stage 1
        self.layers = []
        self.layers += [(self.generate_convblock(64,256,1,1), self.generate_skip(64,256,1,1))]
        self.layers += [(self.generate_convblock(256,256,1,2), self.generate_skip(256,256,1,2))]
        self.layers += [(self.generate_convblock(256,256,1,3), self.generate_skip(256,256,1,3))]
        #Stage 2
        self.layers += [(self.generate_convblock(256,512, 2,4), self.generate_skip(256,512, 2,4))]
        self.layers += [(self.generate_convblock(512,512,1,5), self.generate_skip(512,512,1,5))]
        self.layers += [(self.generate_convblock(512,512,1,6), self.generate_skip(512,512,1,6))]
        #Stage 3
        self.layers += [(self.generate_convblock(512,1024, 2,7), self.generate_skip(512,1024, 2,7))]
        self.layers += [(self.generate_convblock(1024,1024,1,8), self.generate_skip(1024,1024,1,8))]
        self.layers += [(self.generate_convblock(1024,1024,1,9), self.generate_skip(1024,1024,1,9))]
        #Stage 4
        self.layers += [(self.generate_convblock(1024,2048, 2,10) , self.generate_skip(1024,2048, 2,10))]
        self.layers += [(self.generate_convblock(2048,2048,1,11), self.generate_skip(2048,2048,1,11))]
        self.layers += [(self.generate_convblock(2048,2048,1,12), self.generate_skip(2048,2048,1,12))]

        # concat

        self.pool = torch.nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.dense = torch.nn.Linear(2048, classes)



    #determines flow of data through network
    def forward(self,x):

        #connect layers of input block
        x = self.c0(x)
        x = self.b0(x)
        x = self.r0(x)
        x = self.m0(x)

        #attach skip connections and convblocks to for body of network
        for layer in self.layers:
            x_skip = x.clone()
            for block_layer in layer[0]:
                x = block_layer(x)
            for skip_layer in layer[1]:
                x_skip = skip_layer(x_skip)
            x+=x_skip

        # Adding in output layers
        x = self.pool(x)
        x = torch.nn.Flatten()(x)
        x = self.dense(x)

        return(x)

    #Handles skip connections
    def generate_skip(self, input_channels, output_channels, stride = 1, step_number= 0):
        layers = []
        layers += [torch.nn.Conv2d(input_channels,output_channels, kernel_size=1, stride = stride)]
        layers += [torch.nn.BatchNorm2d(output_channels, eps=1e-5, momentum=0.1)]
        layers += [torch.nn.ReLU()]

        setattr(self, 'Skip'+str(step_number)+':conv', layers[-3])
        setattr(self, 'Skip'+str(step_number)+':Norm', layers[-2])
        setattr(self, 'Skip'+str(step_number)+':ReLU', layers[-1])
        return(layers)

    #Generates convblock layers
    def generate_convblock(self, input_channels, output_channels, stride = 1, step_number= 0):
        mid = output_channels//4
        layers = []
        #Build and configure layers
        layers += [torch.nn.Conv2d(input_channels,mid, kernel_size=1, stride = stride, padding = 0)]
        layers += [torch.nn.BatchNorm2d(mid, eps=1e-5, momentum=0.1)]
        layers += [torch.nn.ReLU()]
        #Name layers for clarity in model summary
        setattr(self, 'Block'+str(step_number)+':convA', layers[-3])
        setattr(self, 'Block'+str(step_number)+':NormA', layers[-2])
        setattr(self, 'Block'+str(step_number)+':ReLUA', layers[-1])

        layers += [torch.nn.Conv2d(mid,mid, kernel_size=3, stride = 1, padding = 1)]
        layers += [torch.nn.BatchNorm2d(mid, eps=1e-5, momentum=0.1)]
        layers += [torch.nn.ReLU()]
        setattr(self, 'Block'+str(step_number)+':convB', layers[-3])
        setattr(self, 'Block'+str(step_number)+':NormB', layers[-2])
        setattr(self, 'Block'+str(step_number)+':ReLUB', layers[-1])

        layers += [torch.nn.Conv2d(mid,output_channels, kernel_size=1, stride = 1, padding = 0)]
        layers += [torch.nn.BatchNorm2d(output_channels, eps=1e-5, momentum=0.1)]
        layers += [torch.nn.ReLU()]
        setattr(self, 'Block'+str(step_number)+':convC', layers[-3])
        setattr(self, 'Block'+str(step_number)+':NormC', layers[-2])
        setattr(self, 'Block'+str(step_number)+':ReLUC', layers[-1])

        return(layers)
