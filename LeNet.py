import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, 10),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


mnist_train = torchvision.datasets.FashionMNIST(root="./", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./", train=False, transform=transforms.ToTensor(), download=True)

net = LeNet().to(device)
# net.load_state_dict(torch.load('save\LeNet.pt', map_location='cpu'))
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
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
        train_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()  # 计算准确率

    # 测试T
    net.eval()
    test_loss, test_acc = 0, 0
    for img, label in test_dataloader:
        img = img.to(device)
        label = label.to(device)
        y = net.forward(img)
        loss = loss_function(y, label)

        test_loss += loss.item()
        test_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()  # 计算准确率

    # 统计
    my_writer.add_scalars("Loss", {"train": train_loss / len(train_dataloader), "test": test_loss / len(test_dataloader)}, step)
    my_writer.add_scalars("Acc", {"train": train_acc / len(mnist_train), "test": test_acc / len(mnist_test)}, step)

# 保存模型
torch.save(net.state_dict(), "save/LeNet.pt")
