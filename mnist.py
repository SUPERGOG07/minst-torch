import torch as torch
import torchvision
from torch import nn as nn
from torch.utils.data import DataLoader

# 超参数
batch_size = 128
epochs = 8
learning_rate = 0.001

# 导入测试集和训练集
train_dataset = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ]))
test_dataset = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ]))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 构建网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(nn.Linear(128 * 4 * 4, 1024),
                                nn.ReLU(inplace=True),
                                nn.Linear(1024, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        fc_out = self.fc(x)
        return fc_out


net = Net()
if torch.cuda.is_available():
    net = net.cuda()

# 定义损失函数和迭代器
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


def train(epoch):
    net.train()
    for i, (img, label) in enumerate(train_loader, 1):
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        # 前向传播
        optimizer.zero_grad()
        output = net(img)
        loss = loss_fn(output, label)
        # 反向传播
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            # 计算损失率与准确率
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, i * len(img),
                                                                           len(train_loader.dataset),
                                                                           100 * i / len(train_loader),
                                                                           loss.item()))
            torch.save(net.state_dict(), './net.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


def test():
    net.eval()
    loss_rate = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for (img, label) in test_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            output = net(img)
            loss = loss_fn(output, label)
            loss_rate += loss.item() * label.size(0)
            prediction = output.data.max(1, keepdim=True)[1]
            accuracy += prediction.eq(label.data.view_as(prediction)).sum()
    loss_rate /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss_rate, accuracy, len(test_loader.dataset),
        100. * accuracy / len(test_loader.dataset)))


for epoch in range(1,epochs+1):
    train(epoch)
    test()
