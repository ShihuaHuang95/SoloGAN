from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from models.model import Discriminator, Generator, Style, Content, weights_init
from util.loss import GANLoss, KL_loss
from util.util import tensor2im, load_vgg16, vgg_preprocess
import numpy as np
import torch.nn.functional as F
import random

################## SoloGAN #############################
class SoloGAN():
    def name(self):
        return 'SoloGAN'

    def initialize(self, opt):
        torch.cuda.set_device(opt.gpu)
        cudnn.benchmark = True
        self.opt = opt
        self.build_models()

    def build_models(self):
        # style encoder and content encoder
        self.S = Style(output_nc=self.opt.c_num, nef=self.opt.nef, nd=self.opt.d_num, n_blocks=4)
        self.C = Content(input_dim=3, dim=64, nd=self.opt.d_num)

        # generator
        self.G = Generator(ngf=self.opt.ngf, nc=self.opt.c_num + self.opt.d_num, e_blocks=self.opt.e_blocks)

        if self.opt.isTrain:
            # multi-scale discriminators
            self.Ds = Discriminator(ndf=self.opt.ndf, block_num=4, nd=self.opt.d_num)

            # init_weights
            if self.opt.continue_train:  # resume training
                self.G.load_state_dict(torch.load('{}/G_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.S.load_state_dict(torch.load('{}/S_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.C.load_state_dict(torch.load('{}/C_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.Ds.load_state_dict(torch.load('{}/D_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
            else:
                self.G.apply(weights_init(self.opt.init_type))
                self.S.apply(weights_init(self.opt.init_type))
                self.C.apply(weights_init(self.opt.init_type))
                self.Ds.apply(weights_init(self.opt.init_type))
                    
            # use GPU
            self.G.cuda()
            self.S.cuda()
            self.C.cuda()
            self.Ds.cuda()
            # define optimizers
            self.G_opt = self.define_optimizer(self.G, self.opt.G_lr)
            self.S_opt = self.define_optimizer(self.S, self.opt.G_lr)
            self.C_opt = self.define_optimizer(self.C, self.opt.G_lr)
            self.Ds_opt = self.define_optimizer(self.Ds, self.opt.D_lr)
            # set criterion
            self.criterionGAN = GANLoss(mse_loss=True)
        else:  # test mode
            self.G.load_state_dict(torch.load('{}/G_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
            self.G.cuda()
            self.G.eval()
            if self.S is not None:
                self.S.load_state_dict(torch.load('{}/S_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.S.cuda()
                self.S.eval()
                self.C.load_state_dict(torch.load('{}/C_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.C.cuda()
                self.C.eval()

    def sample_latent_code(self, size):
        c = torch.cuda.FloatTensor(size).normal_()
        return Variable(c)

    def get_domain_code(self, domainLable):     # change the labels into one-hot vector
        domainCode = torch.zeros([len(domainLable), self.opt.d_num])
        for index in range(len(domainLable)):
            domainCode[index, domainLable[index]] = 1
        return Variable(domainCode).cuda()

    def define_optimizer(self, Net, lr):
        return optim.Adam(Net.parameters(), lr=lr, betas=(0.5, 0.999))

    def prepare_image(self, data):
        img, sourceD, name = data
        return Variable(torch.cat(img, 0)).cuda(), torch.cat(sourceD, 0), name

    def update_D(self, D, D_opt, real, fake):
        D.zero_grad()
        pred_fake = D(fake.detach())
        pred_real = D(real)
        errD = self.criterionGAN(pred_fake, False) + self.criterionGAN(pred_real, True)
        errD.backward()
        D_opt.step()
        return errD

    def calculate_G(self, D, fake):
        pred_fake = D(fake)
        errG = self.criterionGAN(pred_fake, True)
        return errG

    def update_model(self, data):
        self.real, sourceD, _ = self.prepare_image(data)    # prepare data
        sourceDC = self.get_domain_code(sourceD)  # one-hot vector

        # get the targetD by random selection
        index = []
        for i_ in range(self.opt.d_num):
            a = random.randint(0, self.opt.d_num-1)
            while i_ == a:
                a = random.randint(0, self.opt.d_num-1)
            
            index.append(a)
        targetD = sourceD[index]
        targetDC = self.get_domain_code(targetD)

        content = self.C(self.real)
        s_enc, mu, logvar = self.S(self.real, sourceDC)
        c_rand = self.sample_latent_code(s_enc.size())
        sourceC = torch.cat([sourceDC, s_enc], 1)  # insert one-hot class vector into style vector by concat
        targetC = torch.cat([targetDC, c_rand], 1)  # insert one-hot class vector into style vector by concat

        self.fake = self.G(content, targetC)
        content_rec = self.C(self.fake)

                    ### update D ###
        self.errDs = 0
        self.Ds.zero_grad()
        dis_real = self.Ds(self.real, sourceD.cuda())
        dis_fake = self.Ds(self.fake.detach(), targetD.cuda())
        self.errDs = self.criterionGAN(dis_fake, False) + self.criterionGAN(dis_real, True)
        errDs = self.errDs
        errDs.backward(retain_graph=True)
        self.Ds_opt.step()
        self.Ds.zero_grad()

                    ### update G ###
        self.errGs, self.errKl, self.errCode, errG_total = [], 0, 0, 0
        self.G.zero_grad()
        self.C.zero_grad()
        self.S.zero_grad()

        dis_fake = self.Ds(self.fake, targetD.cuda())
        errG = self.criterionGAN(dis_fake, True)

        self.errGs.append(errG)
        errG_total += errG
        
        # image reconstruction
        if self.opt.lambda_rec > 0:
            self.rec = self.G(content, sourceC)      # rec = real
            self.errRec = torch.mean(torch.abs(self.rec - self.real)) * self.opt.lambda_rec
            errG_total += self.errRec
        else:
            self.rec = None
            print("    ###   Training Without self reconstruction  ###   ")
            
        # Latent reconstruction: style and content
        if self.opt.lambda_c > 0:
            self.errRec_c = torch.mean(torch.abs(content_rec[0] - content[0])) * self.opt.lambda_c
            errG_total += self.errRec_c
            _, mu_rec, _ = self.S(self.fake, targetDC)    # mu_enc = c_rand
            self.errRec_s = torch.mean(torch.abs(mu_rec - c_rand)) * self.opt.lambda_c    # maybe is error
            errG_total += self.errRec_s
        else:
            print("    ###   Training Without latent reconstruction  ###   ")
        
        # cycle reconstruction, image 
        if self.opt.lambda_cyc > 0:
            self.cyc = self.G(content_rec, sourceC)  # cyc = real, maybe the cyc = rec is better
            self.errCyc = torch.mean(torch.abs(self.cyc - self.real)) * self.opt.lambda_cyc
            errG_total += self.errCyc
        else:
            self.cyc = None
            print("    ###   Training Without cycle reconstruction  ###   ")
            
        # KL divergence: KL loss
        self.errKL = KL_loss(mu, logvar) * self.opt.lambda_kl
        errG_total += self.errKL

        errG_total.backward(retain_graph=True)
        self.G_opt.step()
        self.S_opt.step()
        self.C_opt.step()
        self.G.zero_grad()
        self.S.zero_grad()
        self.C.zero_grad()

    def get_current_visuals(self):
        real = make_grid(self.real.data, nrow=self.real.size(0), padding=0)
        fake = make_grid(self.fake.data, nrow=self.real.size(0), padding=0)
        
        if self.opt.lambda_rec == 0:
            self.rec = self.real
        rec = make_grid(self.rec.data, nrow=self.real.size(0), padding=0)
        if self.opt.lambda_cyc == 0:
            self.cyc = self.real
        cyc = make_grid(self.cyc.data, nrow=self.real.size(0), padding=0)
        
        img = [real, rec, fake, cyc]
        name = 'rsal, rec, fake, cyc'
        img = torch.cat(img, 1)
        return OrderedDict([(name, tensor2im(img))])

    def translation(self, data, domain_names=None):
        input, sourceD, img_names = self.prepare_image(data)
        sourceDC = self.get_domain_code(sourceD)
        images, names = [], []
        for i in range(self.opt.d_num):
            images.append([])
            names.append([])
            
        c_enc, _, _ = self.S(input, sourceDC)
        content = self.C(input)

        for i in range(max(sourceD) + 1):
            images[i].append(tensor2im(input[i].data))
            names[i].append('D_{}'.format(i))

        # get the targetD by select given style ramdonly
        if self.opt.d_num == 2:
            indexs = [[1, 0]]
        elif self.opt.d_num ==3:
            indexs = [[1, 2, 0], [2, 0, 1]]
        else:  # self.opt.d_num = 4
            indexs = [[1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]

        for targetD in indexs:
            targetDC = self.get_domain_code(targetD)
            for i in range(self.opt.n_samples):         # random sample style codes from Gaussian distribution
                c_rand = self.sample_latent_code(torch.Size([input.size(0), self.opt.c_num]))
                targetC = torch.cat([targetDC, c_rand], 1)
                output = self.G(content, targetC)
                for j in range(output.size(0)):
                    images[sourceD[j]].append(tensor2im(output[j].data))
                    if domain_names == None:
                        names[sourceD[j]].append('{}_{}2{}'.format(i, sourceD[j], targetD[j]))
                    else:
                        names[sourceD[j]].append('{}_{}2{}'.format(i, domain_names[sourceD[j]], domain_names[targetD[j]]))

        return images, names

    def get_current_errors(self):
        dict = []
        for i in range(self.opt.d_num):
            dict += [('D_{}'.format(i), self.errDs[i].data[0])]
            dict += [('G_{}'.format(i), self.errGs[i].data[0])]
        dict += [('errCyc', self.errCyc.data[0])]
        dict += [('errKl', self.errKL.data[0])]
        dict += [('errCode', self.errCode.data[0])]
        return OrderedDict(dict)

    def update_lr(self, D_lr, G_lr):
        for param_group in self.G_opt.param_groups:
            param_group['lr'] = G_lr
        for param_group in self.S_opt.param_groups:
            param_group['lr'] = G_lr
        for param_group in self.C_opt.param_groups:
            param_group['lr'] = G_lr
            
        for param_group in self.Ds_opt.param_groups:
            param_group['lr'] = D_lr

    def save(self, name):
        torch.save(self.G.state_dict(), '{}/G_{}.pth'.format(self.opt.model_dir, name))
        torch.save(self.S.state_dict(), '{}/S_{}.pth'.format(self.opt.model_dir, name))
        torch.save(self.C.state_dict(), '{}/C_{}.pth'.format(self.opt.model_dir, name))
        torch.save(self.Ds.state_dict(), '{}/D_{}.pth'.format(self.opt.model_dir, name))
