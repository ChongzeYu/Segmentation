import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.nn import BatchNorm2d

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1,padding=1,*args,**kwargs):
        super(ConvBNReLU,self).__init__()
        self.conv = nn.Conv2d(in_chan,out_chan,
                            kernel_size=ks,
                            stride=stride,
                            padding=padding,
                            bias=False)
        self.bn = BatchNorm2d(out_chan) #BN keep channel number
        self.relu = nn.ReLU(inplace=True) #it will modify the input directly, without allocating any additional output
        self.init_weight() # initialize weights

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def init_weight(self):
        for ly in self.children(): #review children
            if isinstance(ly,nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1) #the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
                if not ly.bias is None: nn.init.contant_(ly.bias,0)

class UpSample(nn.Modle):
    
    def __init__(self,n_chan, factor=2):
        super(UpSample,self).__init__()
        out_chan = n_chan*factor*factor
        self.proj = nn.Conv2d(n_chan, out_chan,1,1,0)
        self.up = nn.PixelShuffle(factor) #review
        self.init_weight()

    def forward(self,x):
        feat = self.proj(x)
        feat = self.up(x)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.) # review

class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args,**kwargs):
        super(BiSeNetOutput,self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan,mid_chan,ks=3,stride=1,padding=1) #review input size output size
        self.conv_out = nn.Conv2d(mid_chan, out_chan,kernel_size=1,bias=True)
        self.up = nn.Upsample(scale_factor=up_factor, mode="bilinear",align_corners=False)# review

        self.init_weight()

    def forward(self,x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x
    
    def init_weight(self):
        for ly in self.children(): #review children
            if isinstance(ly,nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1) #the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
                if not ly.bias is None: nn.init.contant_(ly.bias,0)

    def get_params(self): #review
        wd_params, nowd_params = [],[]
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args,**kwargs):
        super(AttentionRefinementModule,self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1,padding=1)
        self.conv_atten = nn.Conv2d(out_chan,out_chan,kernel_size=1,bias=False)
        self.bn_atten = BatchNorm2d(out_chan)

        self.init_weight()

    def forward(self,x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2,3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat,atten)
        return out

    def init_weight(self):
        for ly in self.children(): #review children
            if isinstance(ly,nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1) #the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
                if not ly.bias is None: nn.init.contant_(ly.bias,0)

# class ContextPath(nn.Module):
#     def 