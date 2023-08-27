# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
# utils
import os
import datetime
import numpy as np
import joblib
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window,\
                  camel_to_snake
from skimage import feature
import cv2


def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault('device', torch.device('cuda'))
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']
    weights = torch.ones(n_classes)
    #weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(device)
    weights = kwargs.setdefault('weights', weights)

    if name == 'SSPN':
        patch_size = kwargs.setdefault('patch_size', 9)
        center_pixel = False
        model = SSPN(patch_size=patch_size, classes=n_classes, channels=n_bands)
        lr = kwargs.setdefault('learning_rate', 0.00008)
        # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005)
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 16)


    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault('epoch', 100)
    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=epoch//4, verbose=True))
    #kwargs.setdefault('scheduler', None)
    kwargs.setdefault('batch_size', 100)
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('flip_augmentation', False)
    kwargs.setdefault('radiation_augmentation', False)
    kwargs.setdefault('mixture_augmentation', False)
    kwargs['center_pixel'] = center_pixel
    return model, optimizer, criterion, kwargs

class semodule(nn.Module):

    def __init__(self, channel):
        super(semodule, self).__init__()
        self.channel = channel
        self.layer1 = nn.Conv3d(1, 1, kernel_size=(5, 1, 1), padding=(2, 0, 0))
        self.layer2 = nn.Conv3d(1, 1, kernel_size=(7, 1, 1), padding=(3, 0, 0))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel)
        self.fc2 = nn.Linear(channel, channel)
        self.re = nn.ReLU()
        self.sg = nn.Sigmoid()

    def forward(self, x):
        a, b, c, d, e = x.size()
        x_new = x.reshape((a, c, d, e))
        x1 = self.avgpool(x_new)
        x1 = x1.reshape((a, 1, c, 1, 1))
        x1_1 = self.layer1(x1)
        x1_2 = self.layer2(x1)
        x1 = x1.add(x1_1)
        x1 = x1.add(x1_2)
        x1 = x1.reshape(a, c, 1, 1)
        x2 = x1.view(-1, self.channel)
        x3 = self.fc1(x2)
        x4 = self.re(x3)
        x5 = self.fc2(x4)
        x6 = self.sg(x5)
        x6 = x6.reshape((a, c, 1, 1))
        x7 = x_new * x6
        x7 = x7.reshape(x.size())

        return x7

# 光谱金字塔
class bottom_up_333(nn.Module):

    def __init__(self, channels):
        super(bottom_up_333, self).__init__()
        self.channel = channels
        self.se = semodule(channel=self.channel)

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )

    def forward(self, x):
        x = self.se(x)
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)

        return [p1, p2, p3, p4, p5]

class bottom_up_335(nn.Module):

    def __init__(self, channels):
        super(bottom_up_335, self).__init__()
        self.channel = channels
        self.se = semodule(channel=self.channel)

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(5, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(5, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(5, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(5, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )

    def forward(self, x):
        x = self.se(x)
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)

        return [p1, p2, p3, p4, p5]

class bottom_up_337(nn.Module):

    def __init__(self, channels):
        super(bottom_up_337, self).__init__()
        self.channel = channels
        self.se = semodule(channel=self.channel)

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(7, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(7, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(7, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(7, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(7, 3, 3), padding=2),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1)
        )

    def forward(self, x):
        x = self.se(x)
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)

        return [p1, p2, p3, p4, p5]

class SSPN(nn.Module):

    def __init__(self, channels, patch_size, classes):
        super(SSPN, self).__init__()

        self.channel = channels
        self.patchsize = patch_size

        # 自底向上
        self.bottomup1 = bottom_up_333(self.channel)
        self.bottomup2 = bottom_up_335(self.channel)
        self.bottomup3 = bottom_up_337(self.channel)

        # 最上特征图改变通道数
        self.toplayer = nn.Conv3d(256, 32, kernel_size=1, stride=1, padding=0)

        # 平滑融合后的特征图
        self.smoothlayer = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)

        # 调整通道数
        self.changelayer1 = nn.Conv3d(128, 32, kernel_size=1, stride=1, padding=0)
        self.changelayer2 = nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0)
        self.changelayer3 = nn.Conv3d(32, 32, kernel_size=1, stride=1, padding=0)

        self.feature_size1 = self._feature_size_1()
        self.feature_size2 = self._feature_size_2()
        self.feature_size3 = self._feature_size_3()

        self.fc1_1 = nn.Linear(self.feature_size1[0] + self.feature_size1[1] + self.feature_size1[2], 256)
        self.fc1_2 = nn.Linear(self.feature_size2[0] + self.feature_size2[1] + self.feature_size2[2], 256)
        self.fc1_3 = nn.Linear(self.feature_size3[0] + self.feature_size3[1] + self.feature_size3[2], 256)

        self.dp = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, classes)

    # 上采样
    def _upsample_add(self, x, y):
        _, _, c, w, h = y.size()
        return F.interpolate(x, size=(c, h, w), mode='trilinear', align_corners=True)

    # 计算3*3*3与5*3*3的全连接的输出大小
    def _feature_size_1(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.channel, self.patchsize, self.patchsize))
            bottomup_feature1 = self.bottomup1(x)
            bottomup_feature2 = self.bottomup2(x)

            # 卷积核为3*3*3
            x1_1 = bottomup_feature1[0]
            x1_2 = bottomup_feature1[1]
            x1_3 = bottomup_feature1[2]
            x1_4 = bottomup_feature1[3]
            x1_5 = bottomup_feature1[4]

            # 卷积核为5*3*3
            x2_1 = bottomup_feature2[0]
            x2_2 = bottomup_feature2[1]
            x2_3 = bottomup_feature2[2]
            x2_4 = bottomup_feature2[3]
            x2_5 = bottomup_feature2[4]

            l1_1 = self.toplayer(x1_5)
            l1_ups_1 = self._upsample_add(l1_1, self.changelayer1(x2_4))
            l1_2 = l1_ups_1 + self.changelayer1(x2_4)
            l1_ups_2 = self._upsample_add(l1_ups_1, self.changelayer2(x2_3))
            l1_3 = l1_ups_2 + self.changelayer2(x2_3)
            l1_ups_3 = self._upsample_add(l1_ups_2, self.changelayer3(x2_2))
            l1_4 = l1_ups_3 + self.changelayer3(x2_2)

            l1_2 = self.smoothlayer(l1_2)
            l1_3 = self.smoothlayer(l1_3)
            l1_4 = self.smoothlayer(l1_4)

            _, t1, c1, w1, h1 = l1_2.size()
            _, t2, c2, w2, h2 = l1_3.size()
            _, t3, c3, w3, h3 = l1_4.size()

        return [t1 * c1 * w1 * h1, t2 * c2 * w2 * h2, t3 * c3 * w3 * h3]

    # 计算5*3*3与7*3*3的全连接的输出大小
    def _feature_size_2(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.channel, self.patchsize, self.patchsize))
            bottomup_feature2 = self.bottomup2(x)
            bottomup_feature3 = self.bottomup3(x)

            # 卷积核为5*3*3
            x2_1 = bottomup_feature2[0]
            x2_2 = bottomup_feature2[1]
            x2_3 = bottomup_feature2[2]
            x2_4 = bottomup_feature2[3]
            x2_5 = bottomup_feature2[4]

            # 卷积核为7*3*3
            x3_1 = bottomup_feature3[0]
            x3_2 = bottomup_feature3[1]
            x3_3 = bottomup_feature3[2]
            x3_4 = bottomup_feature3[3]
            x3_5 = bottomup_feature3[4]

            l2_1 = self.toplayer(x2_5)
            l2_ups_1 = self._upsample_add(l2_1, self.changelayer1(x3_4))
            l2_2 = l2_ups_1 + self.changelayer1(x3_4)
            l2_ups_2 = self._upsample_add(l2_ups_1, self.changelayer2(x3_3))
            l2_3 = l2_ups_2 + self.changelayer2(x3_3)
            l2_ups_3 = self._upsample_add(l2_ups_2, self.changelayer3(x3_2))
            l2_4 = l2_ups_3 + self.changelayer3(x3_2)

            l2_2 = self.smoothlayer(l2_2)
            l2_3 = self.smoothlayer(l2_3)
            l2_4 = self.smoothlayer(l2_4)

            _, t1, c1, w1, h1 = l2_2.size()
            _, t2, c2, w2, h2 = l2_3.size()
            _, t3, c3, w3, h3 = l2_4.size()

        return [t1 * c1 * w1 * h1, t2 * c2 * w2 * h2, t3 * c3 * w3 * h3]

    # 计算7*3*3与3*3*3的全连接的输出大小
    def _feature_size_3(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.channel, self.patchsize, self.patchsize))
            bottomup_feature1 = self.bottomup1(x)
            bottomup_feature3 = self.bottomup3(x)

            # 卷积核为3*3*3
            x1_1 = bottomup_feature1[0]
            x1_2 = bottomup_feature1[1]
            x1_3 = bottomup_feature1[2]
            x1_4 = bottomup_feature1[3]
            x1_5 = bottomup_feature1[4]

            # 卷积核为7*3*3
            x3_1 = bottomup_feature3[0]
            x3_2 = bottomup_feature3[1]
            x3_3 = bottomup_feature3[2]
            x3_4 = bottomup_feature3[3]
            x3_5 = bottomup_feature3[4]

            l3_1 = self.toplayer(x3_5)
            l3_ups_1 = self._upsample_add(l3_1, self.changelayer1(x1_4))
            l3_2 = l3_ups_1 + self.changelayer1(x1_4)
            l3_ups_2 = self._upsample_add(l3_ups_1, self.changelayer2(x1_3))
            l3_3 = l3_ups_2 + self.changelayer2(x1_3)
            l3_ups_3 = self._upsample_add(l3_ups_2, self.changelayer3(x1_2))
            l3_4 = l3_ups_3 + self.changelayer3(x1_2)

            l3_2 = self.smoothlayer(l3_2)
            l3_3 = self.smoothlayer(l3_3)
            l3_4 = self.smoothlayer(l3_4)

            _, t1, c1, w1, h1 = l3_2.size()
            _, t2, c2, w2, h2 = l3_3.size()
            _, t3, c3, w3, h3 = l3_4.size()

        return [t1 * c1 * w1 * h1, t2 * c2 * w2 * h2, t3 * c3 * w3 * h3]

    def forward(self, x):
        bottomup_feature1 = self.bottomup1(x)
        bottomup_feature2 = self.bottomup2(x)
        bottomup_feature3 = self.bottomup3(x)

        # 卷积核为3*3*3
        x1_1 = bottomup_feature1[0]
        x1_2 = bottomup_feature1[1]
        x1_3 = bottomup_feature1[2]
        x1_4 = bottomup_feature1[3]
        x1_5 = bottomup_feature1[4]

        # 卷积核为5*3*3
        x2_1 = bottomup_feature2[0]
        x2_2 = bottomup_feature2[1]
        x2_3 = bottomup_feature2[2]
        x2_4 = bottomup_feature2[3]
        x2_5 = bottomup_feature2[4]

        # 卷积核为7*3*3
        x3_1 = bottomup_feature3[0]
        x3_2 = bottomup_feature3[1]
        x3_3 = bottomup_feature3[2]
        x3_4 = bottomup_feature3[3]
        x3_5 = bottomup_feature3[4]

        # 通过上采样将卷积核为3*3*3与5*3*3下采样的结果进行融合
        l1_1 = self.toplayer(x1_5)
        l1_ups_1 = self._upsample_add(l1_1, self.changelayer1(x2_4))
        l1_2 = l1_ups_1 + self.changelayer1(x2_4)
        l1_ups_2 = self._upsample_add(l1_ups_1, self.changelayer2(x2_3))
        l1_3 = l1_ups_2 + self.changelayer2(x2_3)
        l1_ups_3 = self._upsample_add(l1_ups_2, self.changelayer3(x2_2))
        l1_4 = l1_ups_3 + self.changelayer3(x2_2)

        l1_2 = self.smoothlayer(l1_2)
        l1_3 = self.smoothlayer(l1_3)
        l1_4 = self.smoothlayer(l1_4)

        # 通过上采样将卷积核为5*3*3与7*3*3下采样的结果进行融合
        l2_1 = self.toplayer(x2_5)
        l2_ups_1 = self._upsample_add(l2_1, self.changelayer1(x3_4))
        l2_2 = l2_ups_1 + self.changelayer1(x3_4)
        l2_ups_2 = self._upsample_add(l2_ups_1, self.changelayer2(x3_3))
        l2_3 = l2_ups_2 + self.changelayer2(x3_3)
        l2_ups_3 = self._upsample_add(l2_ups_2, self.changelayer3(x3_2))
        l2_4 = l2_ups_3 + self.changelayer3(x3_2)

        l2_2 = self.smoothlayer(l2_2)
        l2_3 = self.smoothlayer(l2_3)
        l2_4 = self.smoothlayer(l2_4)

        # 通过上采样将卷积核为7*3*3与3*3*3下采样的结果进行融合
        l3_1 = self.toplayer(x3_5)
        l3_ups_1 = self._upsample_add(l3_1, self.changelayer1(x1_4))
        l3_2 = l3_ups_1 + self.changelayer1(x1_4)
        l3_ups_2 = self._upsample_add(l3_ups_1, self.changelayer2(x1_3))
        l3_3 = l3_ups_2 + self.changelayer2(x1_3)
        l3_ups_3 = self._upsample_add(l3_ups_2, self.changelayer3(x1_2))
        l3_4 = l3_ups_3 + self.changelayer3(x1_2)

        l3_2 = self.smoothlayer(l3_2)
        l3_3 = self.smoothlayer(l3_3)
        l3_4 = self.smoothlayer(l3_4)

        # 计算各个融合模块的展平大小
        final_size_1 = self.feature_size1
        final_size_2 = self.feature_size2
        final_size_3 = self.feature_size3

        # 3*3*3&5*3*3
        final_size1_1 = final_size_1[0]
        final_size1_2 = final_size_1[1]
        final_size1_3 = final_size_1[2]

        fc1_1_1 = l1_2.view(-1, final_size1_1)
        fc1_1_2 = l1_3.view(-1, final_size1_2)
        fc1_1_3 = l1_4.view(-1, final_size1_3)

        # 5*3*3&7*3*3
        final_size2_1 = final_size_2[0]
        final_size2_2 = final_size_2[1]
        final_size2_3 = final_size_2[2]

        fc1_2_1 = l2_2.view(-1, final_size2_1)
        fc1_2_2 = l2_3.view(-1, final_size2_2)
        fc1_2_3 = l2_4.view(-1, final_size2_3)

        # 3*3*3&7*3*3
        final_size3_1 = final_size_3[0]
        final_size3_2 = final_size_3[1]
        final_size3_3 = final_size_3[2]

        fc1_3_1 = l3_2.view(-1, final_size3_1)
        fc1_3_2 = l3_3.view(-1, final_size3_2)
        fc1_3_3 = l3_4.view(-1, final_size3_3)

        # 各个模块全连接
        fn1_1 = torch.cat((torch.cat((fc1_1_1, fc1_1_2), dim=1), fc1_1_3), dim=1)
        fn1_2 = torch.cat((torch.cat((fc1_2_1, fc1_2_2), dim=1), fc1_2_3), dim=1)
        fn1_3 = torch.cat((torch.cat((fc1_3_1, fc1_3_2), dim=1), fc1_3_3), dim=1)

        fn1_1 = self.fc1_1(fn1_1)
        fn1_2 = self.fc1_2(fn1_2)
        fn1_3 = self.fc1_3(fn1_3)
        fn1_4 = fn1_1 + fn1_2 + fn1_3

        fn1_1 = self.dp(fn1_1)
        fn1_2 = self.dp(fn1_2)
        fn1_3 = self.dp(fn1_3)
        fn1_4 = self.dp(fn1_4)

        fn1_1 = self.fc2(fn1_1)
        fn1_2 = self.fc2(fn1_2)
        fn1_3 = self.fc2(fn1_3)
        fn1_4 = self.fc2(fn1_4)

        return fn1_1, fn1_2, fn1_3, fn1_4

def train(net, optimizer, criterion, data_loader, epoch, model_name, scheduler=None,
          display_iter=100, device=torch.device('cpu'), display=None,
          val_loader=None, supervision='full'):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    save_epoch = epoch // 20 if epoch > 20 else 1


    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            a = target.size()
            if supervision == 'full':
                if model_name == 'SSPN':
                    output1_1, output1_2, output1_3, _ = net(data)
                    #target = target - 1
                    loss1 = criterion(output1_1, target)
                    loss2 = criterion(output1_2, target)
                    loss3 = criterion(output1_3, target)
                    # loss4 = criterion(output1_4, target)
                    loss = loss1 * (1/3) + loss2 * (1/3) + loss3 * (1/3)
                elif model_name:
                    output = net(data)
                    loss = criterion(output, target)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
                #target = target - 1
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[1](rec, data)
            else:
                raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                string = string.format(
                    e, epoch, batch_idx *
                    len(data), len(data) * len(data_loader),
                    100. * batch_idx / len(data_loader), mean_losses[iter_])
                update = None if loss_win is None else 'append'
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter:iter_],
                    win=loss_win,
                    update=update,
                    opts={'title': "Training loss",
                          'xlabel': "Iterations",
                          'ylabel': "Loss"
                         }
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(Y=np.array(val_accuracies),
                                           X=np.arange(len(val_accuracies)),
                                           win=val_win,
                                           opts={'title': "Validation accuracy",
                                                 'xlabel': "Epochs",
                                                 'ylabel': "Accuracy"
                                                })
            iter_ += 1
            if model_name == "SPNwithMaxp_havese_ps9_lr000008_cpfalse_norloss":
                del(data, target, loss, output1_1, output1_2, output1_3)
            elif model_name:
                del(data, target, loss, output)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, model_name=model_name, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if e % save_epoch == 0:
            save_model(net, camel_to_snake(str(net.__class__.__name__)), data_loader.dataset.name, epoch=e, metric=abs(metric))

def save_model(model, model_name, dataset_name, **kwargs):
     model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
     if not os.path.isdir(model_dir):
         os.makedirs(model_dir, exist_ok=True)
     if isinstance(model, torch.nn.Module):
         filename = str('run') + "_epoch{epoch}_{metric:.2f}".format(**kwargs)
         tqdm.write("Saving neural network weights in {}".format(filename))
         torch.save(model.state_dict(), model_dir + filename + '.pth')
     else:
         filename = str('run')
         tqdm.write("Saving model params in {}".format(filename))
         joblib.dump(model, model_dir + filename + '.pkl')


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    model_name = hyperparams['model']
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))
    pred = []

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            # print("indices:", len(indices))
            data = data.to(device)
            if model_name == "SSPN":
                _, _, _,output = net(data)
            elif model_name:
                output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')

            output = output.numpy()
            # print("output shape:", output.shape)
            pred.extend(output.argmax(axis=1))
            
            # print("output", output)

            # if patch_size == 1 or center_pixel:
            #     output = output.numpy()
            # else:
            #     print("output shape is ", output.size())
            #     output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x:x + w, y:y + h] += out
                    # print("out", out)
    return probs

def val(net, data_loader, model_name, device='cpu', supervision='full'):
# TODO : fix me using metrics()
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == 'full':
                if model_name == "SSPN":
                    _, _, _, output = net(data)
                elif model_name:
                    output = net(data)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            #target = target - 1
            for pred, out in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total
