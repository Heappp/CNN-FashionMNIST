import torch
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseBlock(nn.Module):
    def __init__(self, num, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num):
            self.layers.append(self.DenseLayer(in_channels + out_channels * i, out_channels))

    @staticmethod
    def DenseLayer(in_channels, out_channels):
        layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        return layer

    def forward(self, x):
        for layer in self.layers:
            x1 = layer(x)
            x = torch.concat([x, x1], dim=1)
        return x


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
        )
        self.DenseBlock1 = nn.Sequential(
            DenseBlock(6, 64, 32),
            self.TransitionBlock(256, 128)
        )
        self.DenseBlock2 = nn.Sequential(
            DenseBlock(12, 128, 32),
            self.TransitionBlock(512, 256)
        )
        self.DenseBlock3 = nn.Sequential(
            DenseBlock(24, 256, 32),
            self.TransitionBlock(1024, 512)
        )
        self.DenseBlock4 = nn.Sequential(
            DenseBlock(16, 512, 32),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 10),
        )

    @staticmethod
    def TransitionBlock(in_channels, out_channels):
        layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        )
        return layer

    def forward(self, x):
        x = self.conv(x)
        x = self.DenseBlock1(x)
        x = self.DenseBlock2(x)
        x = self.DenseBlock3(x)
        x = self.DenseBlock4(x)
        x = self.fc(x)
        return x


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
mnist_train = torchvision.datasets.FashionMNIST(root="./", train=True, transform=transform, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./", train=False, transform=transform, download=True)

net = DenseNet().to(device)
# net.load_state_dict(torch.load('save\DenseNet.pt', map_location='cpu'))
loss_function = nn.CrossEntropyLoss(label_smoothing=0.2).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
my_writer = SummaryWriter("tf-logs")

epoch = 20
batch_size = 64

train_dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=False)
test_dataloader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

for step in range(epoch):
    # 训练
    net.train()
    train_loss, train_acc = 0, 0
    for img, label in train_dataloader:
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        y = net.forward(img)
        loss = loss_function(y, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()
    scheduler.step()

    # 测试
    net.eval()
    test_loss, test_acc = 0, 0
    for img, label in test_dataloader:
        img = img.to(device)
        label = label.to(device)
        y = net.forward(img)
        loss = loss_function(y, label)

        test_loss += loss.item()
        test_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()

    # 统计
    my_writer.add_scalars("Loss", {"train": train_loss / len(train_dataloader), "test": test_loss / len(test_dataloader)}, step)
    my_writer.add_scalars("Acc", {"train": train_acc / len(mnist_train), "test": test_acc / len(mnist_test)}, step)

# 保存模型
torch.save(net.state_dict(), "save/DenseNet.pt")
