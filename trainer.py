from __future__ import print_function

import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

import torch
from torch import nn
import torch.nn.parallel
import torchvision.utils as vutils
from torch.autograd import Variable

from models import *
from data_loader import get_loader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def next(loader):
    return loader.next()[0]

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

        self.num_gpu = config.num_gpu
        self.dataset = config.dataset

        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.build_model()

        if self.num_gpu > 0:
            self.G.cuda()
            self.D.cuda()

        if self.load_path:
            self.load_model()

        self.use_tensorboard = config.use_tensorboard
        if self.use_tensorboard:
            import tensorflow as tf
            self.summary_writer = tf.summary.FileWriter(self.model_dir)

            def inject_summary(summary_writer, tag, value, step):
                if hasattr(value, '__len__'):
                    for idx, img in enumerate(value):
                        summary = tf.Summary()
                        sio = StringIO.StringIO()
                        scipy.misc.toimage(img).save(sio, format="png")
                        image_summary = tf.Summary.Image(encoded_image_string=sio.getvalue())
                        summary.value.add(tag="{}/{}".format(tag, idx), image=image_summary)
                        summary_writer.add_summary(summary, global_step=step)
                else:
                    summary= tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                    summary_writer.add_summary(summary, global_step=step)

            self.inject_summary = inject_summary

    def build_model(self):
        channel, height, width = self.data_loader.shape
        assert height == width, "height and width should be same"

        repeat_num = int(np.log2(height)) - 2
        self.D = DiscriminatorCNN(
                channel, self.z_num, repeat_num, self.conv_hidden_num, self.num_gpu)
        self.G = GeneratorCNN(
                self.z_num, self.D.conv2_input_dim, channel, repeat_num, self.conv_hidden_num, self.num_gpu)

        self.G.apply(weights_init)
        self.D.apply(weights_init)

    def train(self):
        l1 = nn.L1Loss(size_average=True)

        #z_D = Variable(torch.FloatTensor(self.batch_size, self.z_num))
        z_G = Variable(torch.FloatTensor(self.batch_size, self.z_num))
        z_fixed = Variable(torch.FloatTensor(self.batch_size, self.z_num).normal_(0, 1), volatile=True)

        if self.num_gpu > 0:
            l1.cuda()

            #z_D = z_D.cuda()
            z_G = z_G.cuda()
            z_fixed = z_fixed.cuda()

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        def get_optimizer(g_lr, d_lr):
            return optimizer(self.G.parameters(), lr=g_lr, betas=(self.beta1, self.beta2)), \
                   optimizer(self.D.parameters(), lr=d_lr, betas=(self.beta1, self.beta2))

        g_optim, d_optim = get_optimizer(self.g_lr, self.d_lr)

        data_loader = iter(self.data_loader)
        x_fixed = self._get_variable(next(data_loader))
        vutils.save_image(x_fixed.data, '{}/x_fixed.png'.format(self.model_dir))

        k_t = 0
        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)

        for step in trange(self.start_step, self.max_step):
            try:
                x = next(data_loader)
            except StopIteration:
                data_loader = iter(self.data_loader)
                x = next(data_loader)

            x = self._get_variable(x)
            batch_size = x.size(0)

            self.D.zero_grad()
            self.G.zero_grad()

            z_G.data.normal_(0, 1)
            sample_z_G = self.G(z_G)

            #z_D.data.normal_(0, 1)
            #sample_z_D = self.G(z_G)

            AE_x = self.D(x)
            AE_G = self.D(sample_z_G.detach())

            d_loss_real = l1(AE_x, x)
            d_loss_fake = l1(AE_G, sample_z_G.detach())

            d_loss = d_loss_real - k_t * d_loss_fake

            d_loss.backward()
            d_optim.step()

            self.D.zero_grad()
            self.G.zero_grad()

            sample_z_G = self.G(z_G)
            AE_G = self.D(sample_z_G.detach())
            g_loss = l1(sample_z_G, AE_G.detach())

            #print(self.D.parameters().next()[0].sum().data[0], self.G.parameters().next()[0].sum().data[0])

            g_loss.backward()
            g_optim.step()

            #print(self.D.parameters().next()[0].sum().data[0], self.G.parameters().next()[0].sum().data[0])

            g_d_balance = (self.gamma * d_loss_real - d_loss_fake).data[0]
            k_t += self.lambda_k * g_d_balance
            k_t = max(min(1, k_t), 0)

            measure = d_loss_real.data[0] + abs(g_d_balance)
            measure_history.append(measure)

            if step % self.log_step == 0:
                print("[{}/{}] Loss_D: {:.3f} L_x: {:.3f} Loss_G: {:.3f} "
                      "measure: {:.3f}, k_t: {:.3f}, g_lr: {:.7f}, d_lr: {:.7f}". \
                      format(step, self.max_step, d_loss.data[0], d_loss_real.data[0],
                             g_loss.data[0], measure, k_t, self.g_lr, self.d_lr))
                x_fake = self.generate(z_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

                if self.use_tensorboard:
                    info = {
                        'loss/loss_D': d_loss.data[0],
                        'loss/L_x': d_loss_real.data[0],
                        'loss/Loss_G': g_loss.data[0],
                        'misc/measure': measure,
                        'misc/k_t': k_t,
                        'misc/d_lr': self.d_lr,
                        'misc/g_lr': self.g_lr,
                        'misc/balance': g_d_balance,
                    }
                    for tag, value in info.items():
                        self.inject_summary(self.summary_writer, tag, value, step)

                    self.inject_summary(
                            self.summary_writer, "AE_G", AE_G.data.cpu().numpy(), step)
                    self.inject_summary(
                            self.summary_writer, "AE_x", AE_x.data.cpu().numpy(), step)
                    self.inject_summary(
                            self.summary_writer, "z_G", sample_z_G.data.cpu().numpy(), step)

                    self.summary_writer.flush()

            if step % self.save_step == self.save_step - 1:
                self.save_model(step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.g_lr *= 0.5
                self.d_lr *= 0.5
                g_optim, d_optim = get_optimizer(self.g_lr, self.d_lr)

    def generate(self, inputs, path, idx=None):
        path = '{}/{}_G.png'.format(path, idx)
        x = self.G(inputs)
        vutils.save_image(x.data, path)
        print("[*] Samples saved: {}".format(path))
        return x

    def autoencode(self, inputs, path, idx=None, x_fake=None):
        x_path = '{}/{}_D.png'.format(path, idx)
        x = self.D(inputs)
        vutils.save_image(x.data, x_path)
        print("[*] Samples saved: {}".format(x_path))

        if x_fake is not None:
            x_fake_path = '{}/{}_D_fake.png'.format(path, idx)
            x = self.D(x_fake)
            vutils.save_image(x.data, x_fake_path)
            print("[*] Samples saved: {}".format(x_fake_path))

    def test(self):
        data_loader = iter(self.data_loader)
        x_fixed = self._get_variable(next(data_loader))
        vutils.save_image(x_fixed.data, '{}/x_fixed_test.png'.format(self.model_dir))
        self.autoencode(x_fixed, self.model_dir, idx="test", x_fake=None)

    def save_model(self, step):
        print("[*] Save models to {}...".format(self.model_dir))

        torch.save(self.G.state_dict(), '{}/G_{}.pth'.format(self.model_dir, step))
        torch.save(self.D.state_dict(), '{}/D_{}.pth'.format(self.model_dir, step))

    def load_model(self):
        print("[*] Load models from {}...".format(self.load_path))

        paths = glob(os.path.join(self.load_path, 'G_*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.load_path))
            return

        idxes = [int(os.path.basename(path.split('.')[1].split('_')[-1])) for path in paths]
        self.start_step = max(idxes)

        if self.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else: 
            map_location = None

        G_filename = '{}/G_{}.pth'.format(self.load_path, self.start_step)
        self.G.load_state_dict(
            torch.load(G_filename, map_location=map_location))
        print("[*] G network loaded: {}".format(G_filename))

        D_filename = '{}/D_{}.pth'.format(self.load_path, self.start_step)
        self.D.load_state_dict(
            torch.load(D_filename, map_location=map_location))
        print("[*] D network loaded: {}".format(D_filename))

    def _get_variable(self, inputs):
        if self.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out
