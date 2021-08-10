import time
import numpy as np
import pandas as pd
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
# 1. dataset
data_dir = {
    'train_path': 'train.csv',
    # 'test_path': 'test.csv',
    'image_path': 'images/',
    # 'submit_path': 'sample_submission.csv'
}
train_data = pd.read_csv(data_dir['train_path'], header=None)
# header=None是去掉表头部分.   索引需要减一
# 索引: [第n列][第n行]  ,[:][0] = image,label
# test_data = pd.read_csv(data_dir['test_path'], header=None)

label_sort = sorted(list(set(train_data[1][1:])))  # 除去标签'label', 避免t >= 0 && t < n_classes failed
n_classes = len(label_sort)
class_to_num = dict(zip(label_sort, range(n_classes)))
# number to str
num_to_class = {v: k for k, v in class_to_num.items()}

##   切分数据集
# train :0.8 , val :0.2
index = [i for i in range(len(train_data[1:]))]
# random.shuffle(index)   # 打乱
Train_Set = np.asarray(train_data[1:].iloc[index[:int(len(index) * 0.8)]])
Train_Set_img = Train_Set[:, 0]
Train_Set_label = Train_Set[:, 1]
Val_Set = np.asarray(train_data[1:].iloc[index[int(len(index) * 0.8):]])
Val_Set_img = Val_Set[:, 0]
Val_Set_label = Val_Set[:, 1]


class ClassDataset(Dataset):

    def __init__(self, data_path, data_label, transform):

        self.img_path = data_path
        self.img_label = data_label
        self.transform = transform


    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')  # RGB 彩图，L 灰度图
        img = self.transform(img)
        # get img_str_label
        label_idx = self.img_label[index]
        # get img_str_label -> num_label
        label = class_to_num[label_idx]

        return img, label

    def __len__(self):
        return len(self.img_path)


def data_loader(resize=(224,224), batch_size=32):

    train_loader = torch.utils.data.DataLoader(
        ClassDataset(Train_Set_img, Train_Set_label,
                     transforms.Compose([
                         transforms.Resize(resize),
                         transforms.ToTensor()
                     ])),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        ClassDataset(Val_Set_img, Val_Set_label,
                     transforms.Compose([
                         transforms.Resize(resize),
                         transforms.ToTensor(),
                     ])),
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, val_loader

# 2. models

#########################################################################################
#                                         LeNet()
#########################################################################################


class LeNet(nn.Module):
    def __init__(self, n_classes):
        super(LeNet, self).__init__()

        self.net = torch.nn.Sequential(nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                                       nn.AvgPool2d(kernel_size=2, stride=2),
                                       nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                                       nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                                       nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                                       nn.Linear(120, 84), nn.Sigmoid(),
                                       nn.Linear(84, n_classes))

    def forward(self, img):
        img = img.view(-1, 3, 28, 28)
        feat = self.net(img)

        return feat

        # X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)   #  检测
        # net = LeNet().net
        # for layer in net:
        #     X = layer(X)
        #     print(layer.__class__.__name__, 'output shape: \t', X.shape)


#########################################################################################
#                                         AlexNet()
#########################################################################################


class AlexNet(nn.Module):
    def __init__(self, n_classes):
        super(AlexNet, self).__init__()

        self.net = torch.nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes))

    def forward(self, img):
        # img = img.view(-1, 1, 224, 224)         #shape '[-1, 1, 224, 224]' is invalid for input of size 90000
        feat = self.net(img)

        return feat


#########################################################################################
#                                         VGG()
#########################################################################################


class VGG(nn.Module):

    def vgg_block(self, num_conv, in_channels, out_channels):
        layers = []
        for _ in range(num_conv):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def vgg_main(self, conv_arch=None):
        conv_blks = []
        in_channels = 3
        for (num_conv, out_channels) in conv_arch:
            conv_blks.append(
                self.vgg_block(num_conv, in_channels, out_channels))
            in_channels = out_channels

        return in_channels, conv_blks

    def __init__(self, n_classes):
        super(VGG, self).__init__()
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        out_channels, conv_blks = self.vgg_main(conv_arch=conv_arch)
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(4096, n_classes))

    def forward(self, img):
        img = img.view(-1, 3, 224, 224)
        feat = self.net(img)

        return feat


#########################################################################################
#                                         NiN()
#########################################################################################


class NiN(nn.Module):  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def nin_block(self, in_channel, out_channel, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, strides, padding),
            nn.ReLU(), nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.ReLU(), nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.ReLU())

    def __init__(self, n_classes):
        super(NiN, self).__init__()
        self.net = nn.Sequential(
            self.nin_block(3, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            self.nin_block(256, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            self.nin_block(384, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(384, n_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())

    def forward(self, img):
        img = img.view(-1, 3, 224, 224)
        feat = self.net(img)
        return feat


#########################################################################################
#                                         NiN()
#########################################################################################


class NinNet(nn.Module):
    def __init__(self, n_classes):
        super(NinNet, self).__init__()

        self.net = nn.Sequential(
            self.make_layers(3, 96, 11, 4, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.make_layers(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.make_layers(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.make_layers(384, n_classes, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=6, stride=1)
        )

    def make_layers(self, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),  # 1x1卷积,整合多个feature map的特征
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),  # 1x1卷积,整合多个feature map的特征
            nn.ReLU(inplace=True)
        )

        return conv

    def forward(self, img):
        output = self.net(img)
        output = output.view(img.shape[0], -1)  # [batch,10,1,1]-->[batch,10]
        return output


#########################################################################################
#                                         GoogLeNet()
#########################################################################################
class Inception(nn.Module):
    # `c1`--`c4` 是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, n_classes):
        super(GoogLeNet, self).__init__()
        b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                           nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2,
                                                   padding=1))

        b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                           nn.Conv2d(64, 192, kernel_size=3, padding=1),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                           Inception(256, 128, (128, 192), (32, 96), 64),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                           Inception(512, 160, (112, 224), (24, 64), 64),
                           Inception(512, 128, (128, 256), (24, 64), 64),
                           Inception(512, 112, (144, 288), (32, 64), 64),
                           Inception(528, 256, (160, 320), (32, 128), 128),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                           Inception(832, 384, (192, 384), (48, 128), 128),
                           nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, n_classes))

    def forward(self, img):
        feat = self.net(img)
        return feat


#########################################################################################
#                                         ResNet()
#########################################################################################


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)  # inplace == 改写

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
        # #输入==输出
        # blk = Residual(3, 3)
        # X = torch.rand(4, 3, 6, 6)
        # Y = blk(X)
        # Y.shape
        # #输出减半，通道加倍
        # blk = Residual(3, 6, use_1x1conv=True, strides=2)
        # blk(X).shape


class ResNet(nn.Module):
    def __init__(self, n_classes):
        super(ResNet, self).__init__()
        b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        def resnet_block(input_channels, num_channels, num_residuals,
                         first_block=False):
            blk = []
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    blk.append(
                        Residual(input_channels, num_channels, use_1x1conv=True,
                                 strides=2))
                else:
                    blk.append(Residual(num_channels, num_channels))
            return blk

        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(), nn.Linear(512, n_classes))

    def forward(self, img):
        feat = self.net(img)
        return feat


#########################################################################################
#                                         MobileNet()
#########################################################################################


class MobileNetV1(nn.Module):
    def __init__(self, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


# 3. defind train and test


def train(net, train_iter, optimizer, criterion):
    if use_GPU:
        net.cuda()

    train_loss = []
    train_accs = []

    for i, (img, label) in enumerate(tqdm.tqdm(train_iter)):
        # if use_GPU:
        img = img.cuda()
        label = label.cuda()
        optimizer.zero_grad()

        prediction = net(img)
        loss = criterion(prediction, label)

        loss.backward()
        optimizer.step()

        acc = (prediction.argmax(dim=-1) == label).float().mean()
        train_accs.append(acc)
        train_loss.append(loss.item())

    train_acc = sum(train_accs) / len(train_accs)
    train_loss = np.mean(train_loss)
    return train_acc, train_loss


def test(net, test_iter, criterion):
    net.eval()
    test_loss = []
    test_accs = []

    with torch.no_grad():
        for i, (img, label) in enumerate(tqdm.tqdm(test_iter)):  # enumerate()
            # if use_GPU:
            img = img.cuda()
            label = label.cuda()

            prediction = net(img)
            loss = criterion(prediction, label)

            acc = (prediction.argmax(dim=-1) == label).float().mean()
            test_accs.append(acc)
            test_loss.append(loss.item())
    test_acc = sum(test_accs) / len(test_accs)
    test_loss = np.mean(test_loss)

    return test_acc, test_loss


def train_main(net, train_iter, test_iter, optimizer, criterion, epochs):
    best_loss = 1000.0
    best_acc = 0.0
    plot_train_acc = []
    plot_test_acc = []
    plot_train_loss = []
    plot_test_loss = []

    for epoch in range(epochs):
        print(f"epoch:{epoch + 1}")
        time_start = time.time()
        (train_acc, train_loss) = train(net, train_iter, optimizer, criterion)
        (test_acc, test_loss) = test(net, test_iter, criterion)
        time_cost = time.time() - time_start
        print(f"time_spend:{time_cost :.2f} epoch/s")

        plot_train_acc.append(train_acc)
        plot_test_acc.append(test_acc)
        plot_train_loss.append(train_loss)
        plot_test_loss.append(test_loss)
        print(f'train_loss:{train_loss}, train_acc:{train_acc}')
        print(f'test_loss:{test_loss}, test_acc:{test_acc}')

        if test_loss < best_loss:
            best_loss = test_loss

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 10 == 0:
            # plot acc and loss
            plt.plot(plot_train_acc, ":", label="train_acc")
            plt.plot(plot_test_acc, ":", label="val_acc")
            plt.plot(plot_train_loss, label="train_loss")
            plt.plot(plot_test_loss, label="val_loss")
            plt.title(net_name + '-' + str(epoch + 1))
            plt.legend()
            plt.show()

    print(f"best_loss: {best_loss}, best_acc: {best_acc}")


### config ###
lr = 1e-2
momentum = 0.9  # 动量(momentum)的引入就是为了加快SGD学习过程
weights_dacay = 1e-4
net = ResNet(n_classes)
resize = (224, 224)  # LeNet:28, AlexNet,VGG,NiN:224
batch_size = 64
net_name = 'ResNet'
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)       # torch.optim.Adam(net.parameters(), lr=lr)  # torch.optim.SGD(net.parameters(), lr=lr)
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weights_dacay)     # torch.optim.SGD(net.parameters(), lr=lr)
lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
epochs = 50
use_GPU = True
train_iter, test_iter = data_loader(resize, batch_size)
train_main(net, train_iter, test_iter, lr_scheduler, criterion, epochs)

#NiN
# time_spend:40.24 epoch/s
# train_loss:3.0441212389780126, train_acc:0.43087372183799744
# test_loss:3.8731267863306504, test_acc:0.28817233443260193

# Alex
# time_spend:40.41 epoch/s
# train_loss:0.09581793055505208, train_acc:0.9669522643089294
# test_loss:2.0168238220543695, test_acc:0.6488474607467651
# best_loss: 1.543514143804024, best_acc: 0.6710550785064697

# VGG
# time_spend:218.56 epoch/s
# train_loss:0.06710535570862283, train_acc:0.9779411554336548
# test_loss:2.194955616897863, test_acc:0.6454192399978638
# best_loss: 1.4821232471777046, best_acc: 0.6799300909042358

#GoogLeNet
# time_spend:84.03 epoch/s
# train_loss:0.22739675336557885, train_acc:0.9228625893592834
# test_loss:1.398411529845205, test_acc:0.7100589871406555
# best_loss: 1.1775950650716651, best_acc: 0.7169813513755798

#ResNet
# time_spend:74.33 epoch/s
# train_loss:0.08730341279069366, train_acc:0.9705528020858765
# test_loss:1.3457964360713959, test_acc:0.7362256646156311
# best_loss: 1.1546454511839768, best_acc: 0.7362490892410278
# time_spend:74.05 epoch/s
# train_loss:0.05315887179914052, train_acc:0.9842391014099121
# test_loss:1.0596235862066006, test_acc:0.7639148831367493
# best_loss: 1.0596235862066006, best_acc: 0.7639148831367493
