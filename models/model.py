import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
from torch.autograd import Variable
import math
import functools
from .cbin import CBINorm2d
from .spectral import SpectralNorm


def get_norm_layer(layer_type='in', num_con=2):
    if layer_type == 'bn':
        norm_layer = None
        c_norm_layer = None
    elif layer_type == 'in':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(CBINorm2d, affine=True, num_con=num_con)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer, c_norm_layer


def get_act_layer(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'sigmoid':
        nl_layer = nn.Sigmoid
    elif layer_type == 'tanh':
        nl_layer = nn.Tanh
    else:
        raise NotImplementedError('nl_layer layer [%s] is not found' % layer_type)
    return nl_layer


def weights_init(init_type='xavier'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
    return init_fun


class Conv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1,
                 pad_type='reflect', bias=True, norm_layer=None, act_layer=None):
        super(Conv2dBlock, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=0, bias=bias)
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x

        if act_layer is not None:
            self.activation = act_layer()
        else:
            self.activation = lambda x: x

    def forward(self, x):
        return self.activation(self.norm(self.conv(self.pad(x))))


class TrConv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1,
                 bias=True, norm_layer=None, nl_layer=None):
        super(TrConv2dBlock, self).__init__()
        self.trConv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x

        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x

    def forward(self, x):
        return self.activation(self.norm(self.trConv(x)))


class Upsampling2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, type='Trp', norm_layer=None, nl_layer=None):
        super(Upsampling2dBlock, self).__init__()
        if type == 'Trp':
            self.upsample = TrConv2dBlock(in_planes, out_planes, kernel_size=4, stride=2,
                                          padding=1, bias=False, norm_layer=norm_layer, nl_layer=nl_layer)
        elif type == 'Ner':
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                Conv2dBlock(in_planes, out_planes, kernel_size=4, stride=1, padding=1, pad_type='reflect',
                            bias=False, norm_layer=norm_layer, act_layer=nl_layer))
        else:
            raise ('None Upsampling type {}'.format(type))

    def forward(self, x):
        return self.upsample(x)


def conv3x3(in_planes, out_planes, norm_layer=None, nl_layer=None):
    "3x3 convolution with padding"
    return Conv2dBlock(in_planes, out_planes, kernel_size=3, stride=1, padding=1, pad_type='reflect',
                       bias=False, norm_layer=norm_layer, act_layer=nl_layer)


                        ############ Generator ############
class CResidualBlock(nn.Module):
    def __init__(self, h_dim, c_norm_layer=None, act_layer=None):
        super(CResidualBlock, self).__init__()
        self.n1 = c_norm_layer(h_dim)
        self.a1 = act_layer()
        self.c1 = Conv2dBlock(h_dim, h_dim, kernel_size=3, stride=1, padding=1,
                              pad_type='reflect', bias=False)
        self.n2 = c_norm_layer(h_dim)
        self.a2 = act_layer()
        self.c2 = Conv2dBlock(h_dim, h_dim, kernel_size=3, stride=1, padding=1,
                              pad_type='reflect', bias=False)

    def forward(self, input):
        x, c = input[0], input[1]
        y = self.a1(self.n1(x, c))
        y = self.c1(y)
        y = self.a2(self.n2(y, c))
        y = self.c2(y)
        return [y+x, c]


class Generator(nn.Module):
    def __init__(self, ngf=64, nc=2, e_blocks=5, up_type='Trp'):
        super(Generator, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type='in', num_con=nc)
        act_layer = get_act_layer(layer_type='relu')
        self.c = Conv2dBlock(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1,
                             pad_type='reflect', bias=False)
        block = []
        for i in range(e_blocks):
            block.append(CResidualBlock(ngf * 4, c_norm_layer=c_norm_layer, act_layer=act_layer))
        self.resBlocks = nn.Sequential(*block)
        self.n3 = c_norm_layer(ngf * 4)
        self.a3 = act_layer()
        self.u1 = Upsampling2dBlock(ngf * 4, ngf * 2, type=up_type)
        self.u_n1 = c_norm_layer(ngf * 2)
        self.act1 = act_layer()
        self.u2 = Upsampling2dBlock(ngf * 2, ngf, type=up_type)
        self.u_n2 = c_norm_layer(ngf)
        self.act2 = act_layer()

        self.block = Conv2dBlock(ngf, 3, kernel_size=7, stride=1, padding=3, pad_type='reflect',
                                 bias=False, act_layer=nn.Tanh)

    def forward(self, x, c):
        x = self.c(x)
        x = self.resBlocks([x, c])[0]
        x = self.a3(self.n3(x, c))
        x = self.act1(self.u_n1(self.u1(x), c))
        x = self.act2(self.u_n2(self.u2(x), c))
        x = self.block(x)
        return x


                    ############# Discriminator #############
class SNConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0,
                 pad_type='reflect', bias=True, norm_layer=None, act_layer=None):
        super(SNConv2d, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        self.conv = SpectralNorm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                           stride=stride, padding=0, bias=bias))
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x

        if act_layer is not None:
            self.activation = act_layer()
        else:
            self.activation = lambda x: x

    def forward(self, x):
        return self.activation(self.norm(self.conv(self.pad(x))))


class D_Net(nn.Module):
    def __init__(self, ndf=32, block_num=3, num_class=2):
        super(D_Net, self).__init__()
        norm_layer = None
        act_layer = get_act_layer('lrelu')
        block = [SNConv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False,
                          norm_layer=norm_layer, act_layer=act_layer)]
        dim_in = ndf
        for n in range(1, block_num):
            dim_out = min(dim_in * 2, ndf * 8)
            block += [SNConv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1,
                               bias=False, norm_layer=norm_layer, act_layer=act_layer)]
            dim_in = dim_out
        self.main = nn.Sequential(*block)

                # Projection Discriminator
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dis = SpectralNorm(nn.Linear(dim_in, 1, bias=True))
        self.embed = nn.Embedding(num_class, dim_in)

    def forward(self, x, c):
        x = self.main(x)
        y = self.global_pool(x).squeeze()
        pred = self.dis(y)
        pred += torch.sum(y * self.embed(c), dim=1, keepdim=True)
        return pred


class Discriminator(nn.Module):  # 3-scale discriminator
    def __init__(self, input_nc=3, ndf=32, block_num=3, nd=2):
        super(Discriminator, self).__init__()
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.model_1 = D_Net(ndf=ndf, block_num=block_num, num_class=nd)
        self.model_2 = D_Net(ndf=ndf, block_num=block_num, num_class=nd)
        self.model_3 = D_Net(ndf=ndf, block_num=block_num, num_class=nd)

    def forward(self, x, c):
        pre1 = self.model_1(x, c)
        x = self.down(x)
        pre2 = self.model_2(x, c)
        x = self.down(x)
        pre3 = self.model_3(x, c)
        return [pre1, pre2, pre3]


                    ################ Style encoder #############
def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


class BasicBlock(nn.Module):
    def __init__(self, input_dim, out_dim, norm_layer=None, act_layer=None):
        super(BasicBlock, self).__init__()
        self.norm1 = norm_layer(input_dim)
        self.act1 = act_layer()
        self.conv1 = conv3x3(input_dim, input_dim)
        self.norm2 = norm_layer(input_dim)
        self.act2 = act_layer()
        self.cmp = convMeanpool(input_dim, out_dim)
        self.shortcut = meanpoolConv(input_dim, out_dim)

    def forward(self, input):
        x, c = input
        out = self.act1(self.norm1(x, c))
        out = self.conv1(out)
        out = self.act2(self.norm2(out, c))
        out = self.cmp(out)
        out += self.shortcut(x)
        return [out, c]


class Style(nn.Module):
    def __init__(self, output_nc=8, nef=64, nd=2, n_blocks=4):
        super(Style, self).__init__()
        _, norm_layer = get_norm_layer(layer_type='in', num_con=nd)
        max_ndf = 4
        act_layer = get_act_layer(layer_type='relu')
        self.entry = Conv2dBlock(3, nef, kernel_size=4, stride=2, padding=1, bias=True)
        conv_layers = []

        for n in range(1, n_blocks):
            input_ndf = nef * min(max_ndf, n)  # 2**(n-1)
            output_ndf = nef * min(max_ndf, n + 1)  # 2**n
            conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, act_layer)]

        self.middle = nn.Sequential(*conv_layers)
        self.norm = norm_layer(output_ndf)
        self.exit = nn.Sequential(*[act_layer(), nn.AdaptiveAvgPool2d(1)])

        self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x, c):
        x = self.entry(x)
        x = self.middle([x, c])[0]
        x = self.norm(x, c)
        x_conv = self.exit(x)
        b = x_conv.size(0)
        x_conv = x_conv.view(b, -1)
        mu = self.fc(x_conv)
        logvar = self.fcVar(x_conv)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


                    ################ Content encoder #############
class ResBlock(nn.Module):
    def __init__(self, h_dim, norm_layer=None, act_layer=None):
        super(ResBlock, self).__init__()
        self.c1 = Conv2dBlock(h_dim, h_dim, kernel_size=3, stride=1, padding=1,
                              pad_type='reflect', bias=False)
        self.n1 = norm_layer(h_dim)
        self.a1 = act_layer()

        self.c2 = Conv2dBlock(h_dim, h_dim, kernel_size=3, stride=1, padding=1,
                              pad_type='reflect', bias=False)
        self.n2 = norm_layer(h_dim)
        self.a2 = act_layer()

    def forward(self, x):
        y = self.c1(x)
        y = self.a1(self.n1(y))
        y = self.c2(y)
        y = self.a2(self.n2(y))
        return x + y


class Content(nn.Module):
    def __init__(self, input_dim, dim, nd=2):
        super(Content, self).__init__()
        norm_layer, _ = get_norm_layer(layer_type='in', num_con=nd)
        act_layer = get_act_layer(layer_type='relu')
        pad_type = 'reflect'
        self.c1 = Conv2dBlock(input_dim, dim, kernel_size=7, stride=1, padding=3, pad_type=pad_type)
        self.n1 = norm_layer(dim)
        self.a1 = act_layer()

        # downsampling blocks
        self.c2 = Conv2dBlock(dim, 2 * dim, kernel_size=4, stride=2, padding=1, pad_type=pad_type)
        self.n2 = norm_layer(dim*2)
        self.a2 = act_layer()
        dim *= 2
        self.c3 = Conv2dBlock(dim, 2 * dim, kernel_size=4, stride=2, padding=1, pad_type=pad_type)
        self.n3 = norm_layer(dim*2)
        self.a3 = act_layer()
        dim *= 2

        # residual blocks
        self.res1 = ResBlock(dim, act_layer=act_layer, norm_layer=norm_layer)
        self.res2 = ResBlock(dim, act_layer=act_layer, norm_layer=norm_layer)
        self.res3 = ResBlock(dim, act_layer=act_layer, norm_layer=norm_layer)
        self.res4 = ResBlock(dim, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x):
        x = self.c1(x)
        x = self.a1(self.n1(x))
        x = self.c2(x)
        x = self.a2(self.n2(x))
        x = self.c3(x)
        x = self.a3(self.n3(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return x