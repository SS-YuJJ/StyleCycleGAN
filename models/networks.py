from numpy import pad
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from loguru import logger
import clip
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import torchvision.transforms as transforms

from random import random

from math import floor, log2
from models.stylegan2.stylegan2_pytorch import Trainer
from models.stylegan2.stylegan2_pytorch import image_noise, mixed_list, noise_list, latent_to_w, styles_def_to_tensor

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, train_from_scratch, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    
    if train_from_scratch:
        init_weights(net, init_type, init_gain=init_gain)

    return net


def define_G(G_train_SG, train_from_scratch, input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'style':
        net = StyleGenerator(
            G_train_SG = G_train_SG,
            device = gpu_ids[0],
            # image_size = 256, 
            image_size = 128, 
            # network_capacity = 16,
            network_capacity = 8,
            load_from = 2225,
        )
        net.clip_encoder.build((16,3,224,224))
    elif netG == 'origin_cyc'
        net = ResnetGenerator(
            input_nc=3, 
            output_nc=3, 
            ngf=64, 
            n_blocks=9,
        )
        net.build((16,3,256,256))
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, train_from_scratch, init_type, init_gain, gpu_ids)


def define_D(D_train_enc, train_from_scratch, D_post_type, input_nc, ndf, netD, batch_size, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'clip':
        net = CLIPDiscriminator(
                                model_name='ViT-B/16',
                                num_post_processing_layers=3,
                                num_filters_post_processing_layers=128,
                                num_outputs_discriminator=1,
                                post_processing_type=D_post_type,
                                train_clip_embedding=D_train_enc,
                            )
        net.build((batch_size, 3, 224, 224))
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, train_from_scratch, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

# =================== Manipulated original resnet generator in cyclegan
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self, 
        input_nc=3, 
        output_nc=3, 
        ngf=64, 
        norm_layer=nn.BatchNorm2d, 
        use_dropout=False, 
        n_blocks=6, 
        padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm2d
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_dropout = use_dropout
        self.n_blocks = n_blocks
        self.padding_type = padding_type
        self.norm_layer = norm_layer
        self.normalization = tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.out_norm = tr.Compose(
                [
                    tr.Normalize((0,0,0),(2, 2, 2)),
                    tr.Normalize((-0.5, -0.5, -0.5),(1,1,1))

                ]
            )

    def build(self, input_shape: Tuple[int, int, int, int]):
        
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(self.input_nc, self.ngf, 
                            kernel_size=7, padding=0, bias=self.use_bias),
                 self.norm_layer(self.ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(self.ngf * mult, self.ngf * mult * 2, 
                                kernel_size=3, stride=2, padding=1, bias=self.use_bias),
                      self.norm_layer(self.ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(self.n_blocks):       # add ResNet blocks

            model += [ResnetBlock(self.ngf * mult, 
                                padding_type=self.padding_type, 
                                norm_layer=self.norm_layer, 
                                use_dropout=self.use_dropout, 
                                use_bias=self.use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(self.ngf * mult, int(self.ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=self.use_bias),
                      self.norm_layer(int(self.ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(self.ngf, self.output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def input_normalization(self, x):
        return self.normalization(x)

    def get_translation_module_parameters(self, include_name=False):
        yield from self.model.parameters()


    def forward(self, x):
        x = self.input_normalization(x)
        x = self.model(x)
        x = self.out_norm(x)
        return x


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


# =====================================


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        if num_downs == 8:
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, down_pad=1, up_pad=1)
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, down_pad=2, up_pad=2)
        else:
            for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, 
                 norm_layer=nn.BatchNorm2d, use_dropout=False, down_pad=1, up_pad=1):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=down_pad, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=up_pad)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=1, padding=2, bias=use_bias)

            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=1,
                                        padding=2, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=up_pad, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):

        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

        self.channel_mean = [0.5, 0.5, 0.5]
        self.channel_std = [0.5, 0.5, 0.5]
    
    def get_transforms(self, x):
        return normalize(x, mean=self.channel_mean, std=self.channel_std)

    def forward(self, x):
        """Standard forward."""
        x = self.get_transforms(x)
        return self.model(x)


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):

        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class TheDilation(nn.Module):
    def __init__(self, input_nc=3, ndf=64):
        super().__init__()
        self.dilation_layers = nn.Sequential(
            nn.Conv2d(8*ndf, 8*ndf, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.InstanceNorm2d(8*ndf),
            nn.ReLU(),
            nn.Conv2d(8*ndf, 8*ndf, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.InstanceNorm2d(8*ndf),
            nn.ReLU(),
            nn.Conv2d(8*ndf, 8*ndf, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.InstanceNorm2d(8*ndf),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.dilation_layers(x)


########################################################
#                  CLIPDiscriminator
########################################################

class CLIPDiscriminator(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_post_processing_layers: int,
        num_filters_post_processing_layers: int,
        num_outputs_discriminator: int,
        post_processing_type: str,      # linear | conv
        train_clip_embedding: bool,
    ):
        super().__init__()
        self.is_built = False
        self.model_name = model_name
        self.train_clip_embedding = train_clip_embedding
        self.post_processing_type = post_processing_type

        self.num_post_processing_layers = num_post_processing_layers
        self.num_filters_post_processing_layers = num_filters_post_processing_layers
        self.num_outputs_discriminator = num_outputs_discriminator
    
    def get_training_parameters(self, with_names=False):
        if self.train_clip_embedding:
            if with_names:
                yield from (
                list(self.post_processing_layers.named_parameters())
                + list(self.clip_embedding.clip_model_visual.named_parameters())
            )
            else:
                yield from (
                    list(self.post_processing_layers.parameters())
                    + list(self.clip_embedding.clip_model_visual.parameters())
                )
        
        else:
            if with_names:
                yield from list(self.post_processing_layers.named_parameters())
            else:
                yield from list(self.post_processing_layers.parameters())


    def build(self, input_shape):
        x = torch.zeros(input_shape)

        self.clip_embedding = CLIPSequentialOutput(self.model_name)

        out = self.clip_embedding.forward(x)        # [b, 197, 768]
        
        self.post_processing_layers = nn.ModuleDict() # shape -> [b, 197, 768]

        if self.post_processing_type == "linear":
            # out = out.mean(dim=1)
            out = out.view(out.shape[0],-1)

            for i in range(self.num_post_processing_layers):
                self.post_processing_layers[
                    f"post_processing_layer_{i}"
                ] = nn.Linear(
                    out.shape[1],
                    self.num_filters_post_processing_layers,
                    bias=True,
                )
                out = self.post_processing_layers[f"post_processing_layer_{i}"](out)
                out = F.leaky_relu(out)

            self.post_processing_layers["final_layer"] = nn.Linear(
                out.shape[1], self.num_outputs_discriminator, bias=True
            )
            out = self.post_processing_layers["final_layer"](out)
        
        elif self.post_processing_type == "conv":

            out = out.permute(0, 2, 1)              # [b, 768, 197]

            for i in range(self.num_post_processing_layers):
                self.post_processing_layers[
                    f"post_processing_layer_{i}"
                ] = nn.Conv1d(
                    in_channels=out.shape[1],
                    out_channels=int(out.shape[1]/2),
                    # out_channels=out.shape[1],
                    bias=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                out = self.post_processing_layers[f"post_processing_layer_{i}"](out)
                out = nn.InstanceNorm1d(out.shape[1])(out)
                out = F.leaky_relu(out)

            self.post_processing_layers[
                    f"output_layer"
                ] = nn.Conv1d(
                    in_channels=out.shape[1],
                    out_channels=1,
                    # out_channels=out.shape[1],
                    bias=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            out = self.post_processing_layers[
                    f"output_layer"
                ](out)


            logger.info(f"Shape of out after convolutional layers {out.shape}")

        else:
            raise Exception(f"Invalid 'post_processing_type' for CLIPDiscriminator: {self.post_processing_type}")

        logger.info(
            f"CLIP discriminator built, with "
            f"input shape {input_shape}, "
            f"and output shape {out.shape}, "
        )

        self.is_built = True

    def forward(self, x):

        if not self.is_built:
            self.build(input_shape=x.shape)

        out = self.clip_embedding.forward(x)

        if self.post_processing_type == "linear":
            # out = out.mean(dim=1)
            out = out.view(out.shape[0],-1)
            
            for i in range(self.num_post_processing_layers):
                out = self.post_processing_layers[f"post_processing_layer_{i}"](out)
                out = F.leaky_relu(out)

        elif self.post_processing_type == "conv":

            out = out.permute(0, 2, 1)              # [b, 768, 197]

            for i in range(self.num_post_processing_layers):
                out = self.post_processing_layers[f"post_processing_layer_{i}"](out)
                out = nn.InstanceNorm1d(out.shape[1])(out)
                out = F.leaky_relu(out)

            out = self.post_processing_layers["output_layer"].forward(out)

        return out

########################################################
#                   StyleGenerator
########################################################

class StyleGenerator(nn.Module):
    def __init__(self,
            G_train_SG:bool,
            device,
            image_size = 256, 
            network_capacity = 16,
            load_from = -1,
        ):
        super().__init__()
        
        self.G_train_SG = G_train_SG

        stylegan_model_args = dict(
        rank = device,
        name = 'styleGAN2_celebA',
        results_dir = './results',
        models_dir = './models',
        batch_size = 16,
        base_dir = './models/stylegan_models',
        gradient_accumulate_every = 1,

        image_size = image_size,
        network_capacity = network_capacity,

        fmap_max = 512,
        transparent = False,
        lr = 2e-4,
        lr_mlp = 0.1,
        ttur_mult = 1.5,
        rel_disc_loss = 1.5,
        num_workers = None,
        save_every = 1000,
        evaluate_every = 1000,
        num_image_tiles = 4,
        trunc_psi = 0.75,
        fp16 = False,
        no_pl_reg = False,
        cl_reg = False,
        fq_layers = [],
        fq_dict_size = 256,
        attn_layers = [],
        no_const = False,
        aug_prob = 0.,
        aug_types = ['translation', 'cutout'],
        top_k_training = False,
        generator_top_k_gamma = 0.99,
        generator_top_k_frac = 0.5,
        dual_contrast_loss = False,
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        mixed_prob = 0.9,
        log = False
    )   
        stylegan_model = Trainer(**stylegan_model_args)
        stylegan_model.load(load_from)

        self.stylegan_S = stylegan_model.GAN.S
        self.stylegan_G = stylegan_model.GAN.G

        self.mixed_prob = 0.9
        self.image_size = 256
        self.num_layers = int(log2(self.image_size) - 1)
        self.latent_dim = 512
        
        self.clip_encoder = CLIPSequentialOutput(model_name='ViT-B/16') # b, 197, 768
        self.weighted_average = nn.Conv1d(in_channels=197, out_channels=1,
                                          kernel_size=1, stride=1, padding=0,
                                          bias=True)  # b, 1, 768

        hid_dim = 512
        self.encoding_linears = nn.Sequential(
            nn.Linear(768, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_dim, 512),
            nn.LayerNorm(hid_dim),
        )
        

    def get_training_parameters(self, with_names=False):
        if with_names:
            if self.G_train_SG:
                yield from (
                    list(self.encoding_linears.named_parameters())
                    + list(self.weighted_average.named_parameters())
                    + list(self.clip_encoder.named_parameters())
                    + list(self.stylegan_S.named_parameters())
                    + list(self.stylegan_G.named_parameters())
                )
            else:
                yield from (
                    list(self.encoding_linears.named_parameters())
                    + list(self.weighted_average.named_parameters())
                    + list(self.clip_encoder.named_parameters())
                    + list(self.stylegan_G.blocks[5].named_parameters())
                )
        else:
            if self.G_train_SG:
                yield from (
                    list(self.encoding_linears.parameters())
                    + list(self.weighted_average.parameters())
                    + list(self.clip_encoder.parameters())
                    + list(self.stylegan_S.parameters())
                    + list(self.stylegan_G.parameters())
                )
            else:
                yield from (
                    list(self.encoding_linears.parameters())
                    + list(self.weighted_average.parameters())
                    + list(self.clip_encoder.parameters())
                    + list(self.stylegan_G.blocks[5].parameters())
                )

    def forward(self, x):
        batch_size = x.shape[0]
        # ============================
        out = self.clip_encoder(x) # [b, 197, 768]
        out = self.weighted_average(out).squeeze(1)  # [b, 768]
        out = self.encoding_linears(out)
        style = [(out, self.num_layers)]
        # =============================
        # returns a list of length 1(one noise) or 2(mixed noise), 
        # consisting of tuple(s): (noise list, layer num)
        # noise list in 'style' of shape [batch_size, latent_dim] = [b, 512]
        # if mixed, layer of each noise list sum up to 'num_layers' = 6

        noise = image_noise(batch_size, self.image_size, device=0)
        # noise shape = [b, 256, 256, 1]

        w_space = latent_to_w(self.stylegan_S, style)
        # list of tuples (output of stylegan_S, layer num), layer_num unchanged compared to 'style'
        # noise lists in 'style' are fed into stylegan_S
        # out shape = [b, latent_dim] = [b, 512]

        w_styles = styles_def_to_tensor(w_space)
        # w_styles shape = [b, layer_num, latent_dim] = [b, 6, 512]

        generated_images = self.stylegan_G(w_styles, noise)
        # generated_images shape = [b, 3, 256, 256]
    
        
        return generated_images

###################################################################################
#                               CLIP Embedding
###################################################################################

# Outputs the full vit embedding of CLIP -- [197, 768]
class CLIPSequentialOutput(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.is_built = False
        self.model_name = model_name
        self.channel_mean = [0.48145466, 0.4578275, 0.40821073]
        self.channel_std = [0.26862954, 0.26130258, 0.27577711]

    def model_transforms(self, x):
        return normalize(x, mean=self.channel_mean, std=self.channel_std)

    def forward_image(self, x: torch.Tensor):
        x = self.clip_model_visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(
            x.shape[0], x.shape[1], -1
        )  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.clip_model_visual.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0],
                    1,
                    x.shape[-1],
                    dtype=x.dtype,
                    device=x.device,
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model_visual.positional_embedding.to(x.dtype)
        x = self.clip_model_visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model_visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD    [b, 197, 768]
        return x

    def build(self, input_shape):
        self.clip_model, clip_input_transforms = clip.load(
            self.model_name, device="cpu"
        )
        self.clip_model_visual = self.clip_model.visual
        x = torch.zeros(input_shape)
        x = self.model_transforms(x)
        out = self.forward_image(x)

        logger.info(
            f"CLIP embedding built, with "
            f"input shape {input_shape}, "
            f"and output shape {out.shape}, "
            f"and transforms {clip_input_transforms}"
        )

        self.expected_input_size = clip_input_transforms.transforms[1].size
        self.is_built = True

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        x = self.model_transforms(x)
        if not self.is_built:
            self.build(input_shape=x.shape)
        out = self.forward_image(x)

        return out


# Outputs the final embedding of CLIP -- 512 tokens
class CLIPWithLinearHead(nn.Module):
    def __init__(self, model_name='ViT-B/16'):
        super().__init__()
        self.model_name = model_name
        self.channel_mean = [0.48145466, 0.4578275, 0.40821073]
        self.channel_std = [0.26862954, 0.26130258, 0.27577711]

        clip_model, _ = clip.load(self.model_name, device="cpu")
        self.clip_model_visual = clip_model.visual

    def model_transforms(self, x):
        return normalize(x, mean=self.channel_mean, std=self.channel_std)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear')

        x = self.model_transforms(x)

        out = self.clip_model_visual(x)

        return out


class CLIPInnerEncoder(nn.Module):
    def __init__(
        self, 
        layer_num,
        model_name='ViT-B/16', 
        ):

        super().__init__()  

        assert layer_num < 13, f"The number of CLIP inner layers used should be <= 12, get: {layer_num}"
        self.model_name = model_name
        self.layer_num = layer_num
        
        self.channel_mean = [0.48145466, 0.4578275, 0.40821073]
        self.channel_std = [0.26862954, 0.26130258, 0.27577711]

        clip_model, _ = clip.load(self.model_name, device="cpu")
        self.clip_model_visual = clip_model.visual

    def model_transforms(self, x):
        return normalize(x, mean=self.channel_mean, std=self.channel_std)
    
    def forward_images(self, x):
        
        x = self.clip_model_visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model_visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model_visual.positional_embedding.to(x.dtype)
        x = self.clip_model_visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        blocks = self.clip_model_visual.transformer.resblocks[:self.layer_num]
        x = blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD  [b, 197, 768]

        return x

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear')

        x = self.model_transforms(x)

        x = self.forward_images(x)      # [b, 768]

        

        return x


# class CLIPDiscriminator512(nn.Module):
#     def __init__(self, model_name='ViT-B/16'):
#         super().__init__()  
#         # ============================================
#         self.clip_embedding = CLIPEncoder(model_name)

#         # ============================================
#         self.discriminator_layers = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.LayerNorm(512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 512),
#         )

#     def get_training_parameters(self, with_names=False):
#         yield from (
#             list(self.discriminator_layers.parameters())
#             # + list(self.clip_embedding.parameters())
#         )


#     def forward(self, x):
#         # with torch.no_grad():
#         x = self.clip_embedding(x)        # [b, 512]
#         x = x.unsqueeze(1)              # [b, 1, 512]
        
#         x = self.discriminator_layers(x)
#         x = x.squeeze(1)                # [b, 1]

#         return x



