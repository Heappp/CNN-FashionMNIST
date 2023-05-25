import torch
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.layers1 = self.VGG_Module(1, 1, 64)
        self.layers2 = self.VGG_Module(1, 64, 128)
        self.layers3 = self.VGG_Module(1, 128, 256)
        self.layers4 = self.VGG_Module(1, 256, 512)
        self.layers5 = self.VGG_Module(1, 512, 512)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 10),
        )

    @staticmethod
    def VGG_Module(num, in_channels, out_channels):
        layers = nn.ModuleList()
        for _ in range(num):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU()
            ))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.layers5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
mnist_train = torchvision.datasets.FashionMNIST(root="./", train=True, transform=transform, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./", train=False, transform=transform, download=True)

net = VGG().to(device)
# net.load_state_dict(torch.load('save\VGG.pt', map_location='cpu'))
loss_function = nn.CrossEntropyLoss().to(device)
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
torch.save(net.state_dict(), "save/VGG.pt")
