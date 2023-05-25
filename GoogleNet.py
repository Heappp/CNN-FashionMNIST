import torch
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[2], kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.Conv2d(out_channels[2], out_channels[2], kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels, out_channels[3], kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        )
        self.inception1 = nn.Sequential(
            Inception(192, (64, 128, 32, 32)),
            Inception(256, (128, 192, 96, 64)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            Inception(480, (192, 208, 48, 64)),
        )
        self.inception2 = nn.Sequential(
            Inception(512, (160, 224, 64, 64)),
            Inception(512, (128, 256, 64, 64)),
            Inception(512, (112, 288, 64, 64)),
        )
        self.inception3 = nn.Sequential(
            Inception(528, (256, 320, 128, 128)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            Inception(832, (256, 320, 128, 128)),
            Inception(832, (384, 384, 128, 128)),
        )
        self.fc1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3), padding=0),
            nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )
        self.fc2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3), padding=0),
            nn.Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )
        self.fc3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.inception1(x)
        x2 = self.inception2(x1)
        x3 = self.inception3(x2)
        return self.fc1(x1), self.fc2(x2), self.fc3(x3)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
mnist_train = torchvision.datasets.FashionMNIST(root="./", train=True, transform=transform, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./", train=False, transform=transform, download=True)

net = GoogleNet().to(device)
# net.load_state_dict(torch.load('save\GoogleNet.pt', map_location='cpu'))
loss_function = nn.CrossEntropyLoss(label_smoothing=0.2).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
my_writer = SummaryWriter("tf-logs")

epoch = 20
batch_size = 128


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
        y1, y2, y3 = net.forward(img)
        loss = 0.3 * loss_function(y1, label) + 0.3 * loss_function(y2, label) + loss_function(y3, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += torch.sum(torch.eq(torch.max(y1, dim=1)[1], label)).item()
    scheduler.step()

    # 测试
    net.eval()
    test_loss, test_acc = 0, 0
    for img, label in test_dataloader:
        img = img.to(device)
        label = label.to(device)
        y, _, _ = net.forward(img)
        loss = loss_function(y, label)

        test_loss += loss.item()
        test_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()

    # 统计
    my_writer.add_scalars("Loss", {"train": train_loss / len(train_dataloader), "test": test_loss / len(test_dataloader)}, step)
    my_writer.add_scalars("Acc", {"train": train_acc / len(mnist_train), "test": test_acc / len(mnist_test)}, step)

# 保存模型
torch.save(net.state_dict(), "save/GoogleNet.pt")
