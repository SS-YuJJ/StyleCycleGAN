import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=5.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=5.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['D_A', 'D_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.loss_names = ['cycle_A', 'cycle_B', 'D_A', 'G_A', 'D_B', 'G_B']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A_to_B', 'G_B_to_A', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A_to_B', 'G_B_to_A']

        self.netG_A_to_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                             not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B_to_A = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                             not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.batch_size,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.batch_size,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss() # ideally should have a choice of L1, L2 and Huber Loss
            self.criterionIdt = torch.nn.L1Loss()
            
            if opt.netG == 'style':
                # for name, p in self.netG_A.module.get_training_parameters(with_names=True):
                #     if p.requires_grad == False:
                #         print(f"********* {name} **** NO GRAD *****")
                #     else:
                #         print(f"--- {name} ---")
                print("=="*50)
                print("Using style generator optim params.")
                self.optimizer_G = torch.optim.Adam(
                    itertools.chain(
                        self.netG_A_to_B.module.get_training_parameters(),
                        self.netG_B_to_A.module.get_training_parameters()
                    ), 
                    lr=opt.lr, 
                    betas=(opt.beta1, 0.999)
                )
            else:
                self.optimizer_G = torch.optim.Adam(
                    itertools.chain(
                        self.netG_A_to_B.parameters(),
                        self.netG_B_to_A.parameters()
                    ), 
                    lr=opt.lr, 
                    betas=(opt.beta1, 0.999))
            
            if opt.netD == 'clip':
                print("Using clip discriminators optim params.")
                self.optimizer_D = torch.optim.Adam(
                    itertools.chain(
                        self.netD_A.module.get_training_parameters(), 
                        self.netD_B.module.get_training_parameters()
                    ), 
                    lr=opt.lr, 
                    betas=(opt.beta1, 0.999)
                )
            else:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if opt.use_clip_inner:  # Can choose which inner layer to use, layer 12 is max
                self.clip_distance_embedding = networks.CLIPInnerEncoder(layer_num=opt.loss_clip_layernum).to(opt.gpu_ids[0])
                print(f"*** Build CLIPInnerEncoder for abstract loss, using [{opt.loss_clip_layernum}] inner layers ***")
            
            else:   # CLIPEncoder uses CLIP's final embedding of 512 tokens
                self.clip_distance_embedding = networks.CLIPWithLinearHead().to(opt.gpu_ids[0])
                print(f"*** Build CLIPEncoder for abstract loss, using final 512 tokens. ***")
            print("=="*50)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A_to_B(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B_to_A(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B_to_A(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A_to_B(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        # lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        # if lambda_idt > 0:
        #     # G_A should be identity if real_B is fed: ||G_A(B) - B||
        #     self.idt_A = self.netG_A(self.real_B)
        #     self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        #     # self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            
        #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
        #     self.idt_B = self.netG_B(self.real_A)
        #     self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        
        # else:
        #     self.loss_idt_A = 0
        #     self.loss_idt_B = 0

        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        # get cyc-clip-loss
        self.real_A_clip = self.clip_distance_embedding(self.real_A)
        self.real_B_clip = self.clip_distance_embedding(self.real_B)
        self.rec_A_clip = self.clip_distance_embedding(self.rec_A)
        self.rec_B_clip = self.clip_distance_embedding(self.rec_B)

        self.loss_cycle_A = self.criterionCycle(self.rec_A_clip, self.real_A_clip) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B_clip, self.real_B_clip) * lambda_B

        self.loss_G = (
            self.loss_G_A
            + self.loss_G_B
            + self.loss_cycle_A 
            + self.loss_cycle_B 
            # + self.loss_idt_A
            # + self.loss_idt_B
        )

        self.loss_G.backward()

    def optimize_parameters(self, update_D):
        self.forward()      # compute fake images and reconstruction imagesz

        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        if self.opt.netG =='style':
            self.netG_B_to_A.module.clip_encoder.clip_model_visual.zero_grad()
            self.netG_A_to_B.module.clip_encoder.clip_model_visual.zero_grad()
            self.netG_B_to_A.module.stylegan_S.zero_grad()
            self.netG_A_to_B.module.stylegan_S.zero_grad()
            self.netG_B_to_A.module.stylegan_G.zero_grad()
            self.netG_A_to_B.module.stylegan_G.zero_grad()
        
        self.clip_distance_embedding.zero_grad()

        self.optimizer_G.zero_grad() 

        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        # D_A and D_B
        if update_D:
            self.set_requires_grad([self.netD_A, self.netD_B], True)

            if 'clip' in self.opt.netD:
                self.netD_B.module.clip_embedding.clip_model_visual.zero_grad()
                self.netD_A.module.clip_embedding.clip_model_visual.zero_grad()

            self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate graidents for D_B
            self.optimizer_D.step()  # update D_A and D_B's weights

        
        else:
            self.loss_D_A = 0
            self.loss_D_B = 0
