## 前言

​	这篇笔记是我在完成深度学习与神经网络课程设计时所记录下的，其中不乏有许多错误之处，如有问题的地方或者不理解的地方，可以及时联系我。另外我会非常乐意与大家交流分享一些技术上的知识，此篇博客完整代码我将会放在我的Github上，有需要请自取哈。
另外我的博客地址echoself.com

## 包括内容

​	卷积神经网络（Convolutional Neural Network，CNN）是一种前馈神经网络，主要应用于图像、语音等领域的特征提取和分类任务。卷积神经网络的核心思想是通过卷积操作从输入数据中提取出关键的特征，并使用这些特征来识别和分类输入数据。

​	卷积神经网络由多个层组成，包括卷积层、池化层和全连接层。其中，卷积层用来提取输入数据中的特征，池化层则可以减小特征图的大小并降低计算量，全连接层则用于对特征进行分类或回归等操作。

​	为此，基于几种基本的卷积神经网络模型，利用pytorch框架自带的基本FashionMNIST数据集，实现多分类问题，以下是本文需要解决的问题如下：

- Fashion MNIST数据集介绍；
- LeNet卷积神经网络的实现，并尝试对比修改其结构所产生的不同结果；
- AlexNet、VGG、GoogleNet、ResNet、DenseNet等深度卷积神经网络的实现；

## FashionMNIST

​	Fashion MNIST数据集是一个用于衣服图像分类的数据集，与手写数字的MNIST数据集类似，是机器学习领域常用的实验数据集之一。FashionMNIST数据集包含70000张28×28像素的灰度图像，共分为10个类别，每个类别有6000张训练图像和1000张测试图像。这10个类别包括：T-shirt/top （T恤）、Trouser （裤子）、Pullover （套衫）、Dress （连衣裙）、Coat （外套）、Sandal （凉鞋）、Shirt （衬衫）、Sneaker （运动鞋）、Bag （包）、Ankle boot （短靴）。本次一些基本的卷积神经网络均采用该数据集进行训练并进行对比。

​	在torchvision.datasets下可以直接获取该数据集，代码如下：

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
mnist_train = torchvision.datasets.FashionMNIST(root="./", train=True, transform=transform, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./", train=False, transform=transform, download=True)
```

​	其中参数transform可以传入一些预处理操作，如改变图像大小、数据增强、数据归一化等。在其中我们可以利用以下代码查看该数据集的一些基本信息。需要注意的是进行Resize((224, 224))之后图像大小似乎并没有立刻发生改变，那是因为好像其中有一种惰性机制的存在，此时并没有进行实际的变换操作，只是记录了一个变换序列。这种惰性计算的机制可以有效地降低内存占用，提高代码效率，特别是当处理大量数据时。

```python
print(mnist_train.data.shape, mnist_test.data.shape)
# torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])
print(mnist_train.data.dtype)
# ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(mnist_train.classes)
# torch.uint8
```

## LeNet

#### 网络介绍

​	LeNet是一种经典的卷积神经网络模型，是深度学习领域的开山之作。它由Yann LeCun教授等人在1998年提出，用于手写数字识别任务，成为了当时机器学习研究中非常重要的模型，LeNet的结构图如下图所示。

<img src="save\LeNet.png" style="zoom:60%;" />

​	由以上结构图可以看出输入图像大小为$1\times28\times28$即为单通道的图片，之后分别经过卷积层、池化层、全连接层后得到分类的输出。其图片从输入到输出依次经过以下卷积层和全连接层。以下为图像卷积前后大小计算公式，其中p为padding_size，k为kennel_size，s为stride，除法为整除计算。
$$
x'=\frac{x+2\times p-k}{s}+1
$$

- 首先通过大小为6个$k=5,s=1,p=2$的卷积可以得到$6\times28\times28$的特征图，再经过大小为$k=2,s=2,p=0$的平均池化层可以得到$6\times14\times14$的特征图；

- 同样地通过$6\times16$个$k=5,s=1,p=0$的卷积核可以得到$16\times10\times10$的特征图，再经过大小为$k=2,s=2,p=0$的平均池化层得到的$16\times5\times5$特征图；

- 将特征图拉平后得到神经元数量为的全连接层，之后经过最后结果输出10个神经元；

​	LeNet的激活函数采用sigmoid函数、损失函数采用交叉熵损失函数，池化层采用平均池化。总体来说LeNet是结构还是非常简单的，但又是非常的经典。

#### 代码实现

​	首先我们实现LeNet的网络结构，它继承自torch.nn.Module，它由2个conv和3个fc组成，代码如下：

```python
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
            nn.Flatten(),  # 此操作可以将拉平，默认从第一维开始到最后一维，即batch_size不受影响
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, 10),  # 不用添加nn.Softmax(), 因为torch的交叉熵损失函数中带有softmax
        )

    def forward(self, x):  # [batch_size, 1, 28, 28]
        x = self.conv1(x)  # [batch_size, 6, 14, 14]
        x = self.conv2(x)  # [batch_size, 16, 5, 5]
        x = self.fc1(x)    # [batch_size, 120]
        x = self.fc2(x)    # [batch_size, 84]
        x = self.fc3(x)    # [batch_size, 10]
        return x
```

​	其次是网络的构建和数据的准备，代码如下：

```python
# 是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取FashionMNIST
mnist_train = torchvision.datasets.FashionMNIST(root="./", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./", train=False, transform=transforms.ToTensor(), download=True)

net = LeNet().to(device)  # 实例化LeNet
# net.load_state_dict(torch.load('save\LeNet.pt', map_location='cpu'))  # 加载训练过程中保存的网络
loss_function = nn.CrossEntropyLoss().to(device)  # 交叉熵损失
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)  # 定义Adam优化器并设置学习率大小
my_writer = SummaryWriter("tf-logs")  # 使用tensorboard进行训练过程可视化

epoch = 20  # 训练次数
batch_size = 256  # 批大小

# 封装FashMNIST为DataLoader类型方便训练
train_dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=False)
test_dataloader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)
```

​	最后就是训练过程，在反向传播的过程当中会计算梯度，梯度更新的过程当中会根据学习率更新权值，两者都不会将先前的梯度清除，所以在每个batch中一定要先进行梯度清除，在进行反向传播、最后进行权值更新。代码如下：

```python
for step in range(epoch): 
    # 训练
    net.train()  # 修改为训练模式
    train_loss, train_acc = 0, 0
    for img, label in train_dataloader:
        img = img.to(device)  # [64, 1, 28, 28]
        label = label.to(device)  # [64, 10]

        optimizer.zero_grad()  # 梯度清除
        y = net.forward(img)  # 前向传播
        loss = loss_function(y, label)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 权值更新

        train_loss += loss.item()  # 统计损失
        train_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()  # 统计准确率

    # 测试
    net.eval() # 修改为测试模式
    test_loss, test_acc = 0, 0
    for img, label in test_dataloader:
        img = img.to(device)
        label = label.to(device)

        y = net.forward(img)  # 前向传播
        loss = loss_function(y, label)  # 计算损失

        test_loss += loss.item()  # 统计损失
        test_acc += torch.sum(torch.eq(torch.max(y, dim=1)[1], label)).item()  # 统计准确率

    # 统计
    my_writer.add_scalars("Loss",
                          {"train": train_loss / len(train_dataloader), "test": test_loss / len(test_dataloader)}, step)
    my_writer.add_scalars("Acc", {"train": train_acc / len(mnist_train), "test": test_acc / len(mnist_test)}, step)

# 保存模型
torch.save(net.state_dict(), "save/LeNet.pt")
```

#### 实验结果

​	启动运行后在Terminal输入 tensorboard --logdir="tf-logs" 就可以实时观察训练过程，输入ctrl+z可以退出。对于原始的LeNet在FashionMNIST数据集上面已经有很好的效果，在选取学习率为1e-3和batch_size为64，迭代20次的情况下测试集的准确率能达到0.87，训练结果如下图所示。
<center class="LeNet">
<img src="D:\Document\YFY\markdown\深度学习\image\Acc_LeNet.png" style="zoom:60%;" >
<img src="D:\Document\YFY\markdown\深度学习\image\Loss_LeNet.png" style="zoom:60%;" >
</center>
​	另外我尝试修改其参数，包括学习率、batch_size、激活函数和迭代更多的次数，从以下的对比中好像可以看出合理调整超参数可以获得更好的效果。

<table align="center" style="zoom:60%; height:50%; width:50%; text-align: center;">
    <tr>
        <td></td>
        <td>批大小</td>
        <td>学习率</td>
        <td>激活函数</td>
        <td>迭代次数</td>
        <td>训练集准损失</td>
        <td>测试集损失</td>
        <td>训练集准确率</td>
        <td>测试集准确率 </td>
        <td></td>
    </tr>
    <tr>
        <td>0</td>
        <td>256</td>
        <td>0.001</td>
        <td>sigmoid</td>
        <td>20</td>
        <td>0.379</td>
        <td>0.426</td>
        <td>0.86</td>
        <td>0.846 </td>
        <td></td>
    </tr>
    <tr>
        <td>1</td>
        <td>64</td>
        <td>0.001</td>
        <td>sigmoid</td>
        <td>20</td>
        <td >0.297</td>
        <td>0.336</td>
        <td>0.8875</td>
        <td>0.8769 </td>
        <td></td>
    </tr>
    <tr>
        <td>2</td>
        <td>256</td>
        <td>0.005</td>
        <td>sigmoid</td>
        <td>20</td>
        <td>0.251</td>
        <td>0.423</td>
        <td>0.905</td>
        <td>0.88 </td>
        <td></td>
    </tr>
    <tr>
        <td>3</td>
        <td>256</td>
        <td>0.001</td>
        <td>ReLU</td>
        <td>20</td>
        <td>0.268</td>
        <td>0.337</td>
        <td>0.898</td>
        <td>0.8767 </td>
        <td></td>
    </tr>
    <tr>
        <td>4</td>
        <td>256</td>
        <td>0.001</td>
        <td>sigmoid</td>
        <td>30</td>
        <td>0.343</td>
        <td>0.387</td>
        <td>0.871</td>
        <td>0.857 </td>
        <td></td>
    </tr>
</table>  
## AlexNet

#### 网络介绍

​	AlexNet是一个经典的卷积神经网络模型，其网络结构比较深，并且使用了一些现代神经网络中常用的技巧，如ReLU激活函数、Dropout正则化和数据增强等，相较于之前的神经网络更加有效。AlexNet包含8个层次：5个卷积和3个全连接层，AlexNet的结构图如下图所示。

<img src="D:\Document\YFY\markdown\深度学习\image\AlexNet.png" style="zoom:40%;" />

​	由以上结构图可以看出输入图像大小为$3\times224\times224$即为3通道的图片，之后同样分别经过卷积层、池化层和全连接层后得到分类的输出。其中Alex的结构更宽、更深，并采取了最大池化替换平均池化，ReLU替换Sigmoid激活函数、添加Dropout正则化等方法，使得网络具有更强的拟合能力和泛化能力。

​	其中Dropout在训练过程当中以概率p随机让一些神经元不工作即在反向传播的过程当中与其相连的权值不会更新，这样能起到模型混合的效果，是一种有效的正则化方法，而在评估过程当中则所有的神经元都会工作；最大池化可以更好的提取纹理信息，一般使用在前面的卷积，平均池化则特征图当中都有所贡献，一般使用在最后的卷积，但具体使用场景在具体任务上是不一样的；$ReLU(x)=max(0, x)$，这样的计算非常简单只需要进行一个max操作，另外在$x>0$的部分为恒等映射，其导数一直为1，相比于Sigmoid函数可以有效的避免由于层数增加经过链式法则梯度回传时梯度消失的现象。

#### 代码实现

​	AlexNet代码的实现总体和LeNet差不多，由于FashionMNIST为单通道图片，网络结构输入通道变为1，在获取数据集时和FashionMNIST介绍代码一样Resize为224。另外其训练过程代码都大致一样，这里不再赘述，AlexNet代码实现如下：

```python
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 4096),
            nn.Dropout(0.5),  # 当p=0.5时神经网络一半都会失活，此时组合数最多
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, 10)
        )

    def forward(self, x):   # [batch_size, 1, 224, 224]
        x = self.conv1(x)   # [batch_size, 96, 26, 26]
        x = self.conv2(x)   # [batch_size, 256, 12, 12]
        x = self.conv3(x)   # [batch_size, 256, 5, 5]
        x = self.fc1(x)		# [batch_size, 4096]
        x = self.fc2(x)     # [batch_size, 4096]
        x = self.fc3(x)     # [batch_size, 10]
        return x
```

#### 实验结果

​	同样地启动运行后在Terminal输入 tensorboard --logdir="tf-logs" 就可以实时观察训练过程，输入ctrl+z可以退出。对比于原始的LeNet已经有了很大的提升，在选取学习率为1e-3和batch_size为128，每迭代一次学习率变为原来的0.9倍，迭代20次的情况下测试集的准确率能达到0.92，训练结果如下图所示。

<center class="LeNet">
<img src="D:\Document\YFY\markdown\深度学习\image\Acc_AlexNet.png" style="zoom:60%;" >
<img src="D:\Document\YFY\markdown\深度学习\image\Loss_AlexNet.png" style="zoom:60%;" >
</center>

## VGG

#### 网络介绍

​	VGG的主要特点是使用了非常小的卷积核（尺寸为$3\times3$），并且极其简洁、规整的网络结构，以达到更好的性能。VGG模型在2014年ImageNet比赛中也表现极为优秀，由此成为了深度学习领域又一个重要的里程碑。

​	VGG相比于AlexNet引入VGG块的设计，每个VGG块都使用多个大小为$3\times3$的卷积核、$2\times2$的池化层组成。其中卷积操作改变通道数而不改变大小，池化操作改变大小而不改变通道数。尽管这样的设计使得VGG的参数数量非常庞大，但它的网络结构非常规整，有利于管理和调试。同时，它也通过使用大量卷积层和少量的池化层来增强网络对图片特征的提取能力。VGG结构图如下图所示。

<img src="D:\Document\YFY\markdown\深度学习\image\VGG.png" style="zoom:50%;" />

#### 代码实现

​	VGG代码中实现了VGG_Module模块，可以返回n个卷积加一个最大池化操作，池化过程大小减半。在这里num全部取1，代码如下：

```python
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

    @staticmethod  # 静态方法，返回一个Sequential对象
    def VGG_Module(num, in_channels, out_channels):
        layers = nn.ModuleList()
        for _ in range(num):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU()
            ))
            in_channels = out_channels   # 能够让后面的通道数对的上
        layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0))
        return nn.Sequential(*layers)

    def forward(self, x):     # [batch_size, 1, 224, 224]
        x = self.layers1(x)   # [batch_size, 64, 112, 112]
        x = self.layers2(x)   # [batch_size, 128, 56, 56]
        x = self.layers3(x)   # [batch_size, 256, 28, 28]
        x = self.layers4(x)   # [batch_size, 512, 14, 14]
        x = self.layers5(x)   # [batch_size, 512, 7, 7]
        x = self.fc1(x)       # [batch_size, 4096]
        x = self.fc2(x)       # [batch_size, 10]
        return x
```

#### 实验结果

​	同样在选取学习率为1e-3和batch_size为128，每迭代一次学习率变为原来的0.9倍，迭代20次的情况下测试集的准确率能达到0.93。相比于AlexNet，VGG明显的收敛速度更快，在第6、7次迭代就已经收敛，后面便是验证集loss上升的现象了。训练结果如下图所示。

<center class="LeNet">
<img src="D:\Document\YFY\markdown\深度学习\image\Acc_VGG.png" style="zoom:58%;" >
<img src="D:\Document\YFY\markdown\深度学习\image\Loss_VGG.png" style="zoom:60%;" >
</center>

## GoogleNet

#### 网络介绍

​	GoogLeNet是由谷歌团队提出的深度卷积神经网络模型，该模型曾于2014年在ImageNet比赛中夺冠，并在计算机视觉领域得到广泛应用。与传统卷积神经网络不同，GoogLeNet并不是一个简单的序列网络，而是一个由多个模块构成的深度网络。

​	首先GoogleNet引入了Inception模块，采用大小为1$\times1$、$3\times3$、$5\times5$的卷积核以及$3\times3$的最大池化共4个分别对图像进行特征提取，之后进行concat（拼接）操作作为Inception模块的输出。另外为了减少参数量和运算，其中添加了大小为$1\times1$的卷积进行通道过度以防止concat后通道数目爆炸，这样的设计能够结合不同卷积核大小的感受野，其结构图如下图所示。

<img src="D:\Document\YFY\markdown\深度学习\image\Inception.png" style="zoom:60%;" />

​	其次GoogleNet最后使用了全局平均池化。其实卷积和池化的区别就是：卷积拥有训练的参数，梯度更新时自动学习特征，池化的参数是给定的，以至于池化并不能改变通道数目是因为池化只给定一套卷积参数。另外在浅层引入了辅助分类器，使得整个网络更快的收敛。整个GoogleNet结构图如下所示。

![](D:\Document\YFY\markdown\深度学习\image\GoogleNet.png)

#### 代码实现

​	首先定义Inception模块，其中由四个部分组成，除了已有大小为$1\times1$的卷积块，另外三个添加$1\times1$的卷积核来首先通道过度减少参数。最后在输出时经过三个进行concat（拼接)操作之后再输出，期间不改变图像大小。其中Inception传入的out_channels是一个元组，分别为4个卷积后的输出通道即最后Inception模块的输出通道为out_channels的和，Inception代码如下所示。

```python
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

    def forward(self, x):    # [batch_size, in_channels, w, w]
        x1 = self.conv1(x)   # [batch_size, out_channels[0], w, w]
        x2 = self.conv2(x)   # [batch_size, out_channels[1], w, w]
        x3 = self.conv3(x)   # [batch_size, out_channels[2], w, w]
        x4 = self.conv4(x)   # [batch_size, out_channels[3], w, w]
        return torch.cat([x1, x2, x3, x4], dim=1)  # [batch_size, sum(out_channels), w, w] 在维度为1（通道维度）上拼接
```

​	之后定义GoogleNet，本次实现并没有考虑LRN的实现，其作用也影响不大。但需要注意的是由于辅助训练器的原因，在前向传播的过程当中需要返回三个结果。如上GoogleNet结构图的结构划分所示，分别实现各个结构划分，代码如下所示。

```python
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

    def forward(self, x):          # [batch_size, 1, 224, 224]
        x = self.conv1(x)          # [batch_size, 64, 56, 56]
        x = self.conv2(x)          # [batch_size, 192, 56, 56]
        x1 = self.inception1(x)    # [batch_size, 512, 28, 28]
        x2 = self.inception2(x1)   # [batch_size, 528, 14, 14]
        x3 = self.inception3(x2)   # [batch_size, 1024, 7, 7]
        return self.fc1(x1), self.fc2(x2), self.fc3(x3)   # 3 个 [batch_size, 10]
```

​	Google的训练过程的辅助训练器仅仅来帮助加速收敛，训练过程当中由三者按比例共同决定损失，评估过程则只由最后一层决定最后输出结果。

```python
# 训练过程
y1, y2, y3 = net.forward(img)
loss = 0.3 * loss_function(y1, label) + 0.3 * loss_function(y2, label) + loss_function(y3, label)

# 评估过程
y, _, _ = net.forward(img)
loss = loss_function(y, label)
```

#### 实验结果

​	同样在选取学习率为1e-3和batch_size为128，每迭代一次学习率变为原来的0.9倍和0.2的平滑标签（平滑标签就是在计算交叉熵损失时从正确标签那里分配一点给其它错误标签，比如不平滑时用[0, 1, 0]在计算交叉熵，而平滑标签后用[0.1, 0.8, 0.1]计算交叉熵，这样可以增强泛化能力），迭代20次的情况下测试集的准确率能达到0.9412。由于训练过程的损失为三个损失的和，所以训练结果长这样..,(懒的再训练一次)，训练结果如下图所示。

<center class="LeNet">
<img src="D:\Document\YFY\markdown\深度学习\image\Acc_GoogleNet.png" style="zoom:60%;" >
<img src="D:\Document\YFY\markdown\深度学习\image\Loss_GoogleNet.png" style="zoom:60%;" >
</center>

## ResNet

#### 网络介绍

​	ResNet最显著的特点是通过添加残差模块，即Residual Block，实现了跨层连接（skip connection），这样的设计可以使得在梯度反向传递过程中梯度得以更快地流向浅层，从而避免了深层网络中网络退化的问题。

​	考虑这样一个问题：网络的深度很大程度上决定了这个网络的学习能力，但是网络深度太大时多余的深度就需要学习恒等变换即$f(x)=x$保证结果和最理想深度结果一样，这样的累计导致深层的网络不如浅层的网络即网络退化现象。而添加残差块将学习$f(x)=x$变为$f(x)=x + g(x)$即网络部分只用学习$g(x)=0$即可，而学习$g(x)=0$比学习$f(x)=x$对于网络来说简单许多。残差块如下图所示。

<img src="D:\Document\YFY\markdown\深度学习\image\Residual.png" style="zoom:50%;" />

​	另外，ResNet还采用了批处理标准化（Batch Normalization）技术，大大加速网络的训练过程。这种技术可以使输入数据在经过卷积操作后保持零均值和单位方差，从而加速了网络的收敛过程。此外，ResNet也对网络的层数进行了深度拓展，并通过使用平均池化层代替全连接层来减少了网络中的参数数量，从而进一步提高了模型的性能。

为什么批归一化有效果？在神经网络的训练过程当中，虽然输入的数据取至同一分布，但是经过一层神经网络之后分布会发生改变，导致网络难以训练。但是我们若进行批归一化保证每一批的特征来自同一分布有助于网络的训练。一般我们采取如下公式进行归一化：
$$
x_{norm}=\frac{x-x_{mean}}{x_{std}}
$$
​	其中我们为了提高BN的表达能力，添加了两个学习参数γ和β，这样可以在分布变化不大的前提下提高网络的表达能力。所以最后公式为：
$$
x'=\gamma x_{norm}+\beta
$$
​	另外归一化当中的批次N，对于每一个C、H、W都进行一次归一化，这样我们会得到$2\times C\times H\times W$个学习参数，参数太多了。所以我们将N、H、W放一起看待,对于每一个通道C再进行归一化，这样所需要学习的参数量减少为$2\times C$（权值共享)。

#### 代码实现

​	首先定义残差块，如下图，其中op为当通道数改变时，x项要进行通道变换的选项，stride用于减少特征图大小。这样的设计可以使得残差块可以自由的改变通道数和特征图大小（相加时要保证通道数和特征图大小保持一致），残差块代码如下所示。

<img src="D:\Document\YFY\markdown\深度学习\image\Residual_op.png" style="zoom: 40%;" />

```python
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), op=True):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, padding=0),
            nn.BatchNorm2d(out_channels)
        ) if op else None
        self.ReLU = nn.ReLU()

    def forward(self, x):           # [batch_size, in_channels, w, w]
        x1 = self.conv1(x)			# [batch_size, out_channels, w / stride, w / stride]
        x1 = self.conv2(x1)			# [batch_size, out_channels, w / stride, w / stride]
        if self.conv0:
            x = self.conv0(x)		# [batch_size, out_channels, w / stride, w / stride]
        return self.ReLU(x + x1)
```

​	之后定义ResNet，其结构图如下，和Google一样通过$7\times7$的卷积核最大池化先缩小特征图大小，之后经过8个残差块再接全连接层之后得到输出，其中分别在3、5、7个分别传入op=True和stride=(2, 2)使得通道数倍增和特征图大小倍减，代码如下。

<img src="D:\Document\YFY\markdown\深度学习\image\ResNet.png" style="zoom:60%;" />

```python
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
        )
        self.ResNetBlock1 = nn.Sequential(
            ResNetBlock(64, 64, stride=(1, 1), op=False),
            ResNetBlock(64, 64, stride=(1, 1), op=False)
        )
        self.ResNetBlock2 = nn.Sequential(
            ResNetBlock(64, 128, stride=(2, 2), op=True),
            ResNetBlock(128, 128, stride=(1, 1), op=False),
            ResNetBlock(128, 256, stride=(2, 2), op=True),
            ResNetBlock(256, 256, stride=(1, 1), op=False),
            ResNetBlock(256, 512, stride=(2, 2), op=True),
            ResNetBlock(512, 512, stride=(1, 1), op=False),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def forward(self, x):				# [batch_size, 1, 224, 224]
        x = self.conv(x)             	# [batch_size, 1, 56, 56]
        x = self.ResNetBlock1(x)		# [batch_size, 1, 56, 56]
        x = self.ResNetBlock2(x)		# [batch_size, 1, 7, 7]
        x = self.fc(x)					# [batch_size, 10]
        return x
```

#### 实验结果

​	同样在选取学习率为1e-3和batch_size为128，每迭代一次学习率变为原来的0.9倍和0.2的平滑标签，迭代20次的情况下测试集的准确率能达到0.9434,与前面网络有些许提升（差不多）。训练结果如下图所示。

<center class="LeNet">
<img src="D:\Document\YFY\markdown\深度学习\image\Acc_ResNet.png" style="zoom:60%;" >
<img src="D:\Document\YFY\markdown\深度学习\image\Loss_ResNet.png" style="zoom:60%;" >
</center>

## DenseNet

#### 网络介绍

​	DenseNet，全名为Densely Connected Convolutional Network，是由李沐等人在2017年提出的深度卷积神经网络模型。DenseNet主要通过密集连接和特征复用来缓解神经网络中的梯度消失和参数稀疏性等问题，让网络更加高效并具有更好的泛化能力。相对于ResNet的残差块的直接相加，DenseNet采用concat以学习更复杂的映射关系，其结构图如下图所示。

![](D:\Document\YFY\markdown\深度学习\image\DenseNetBlock.png)

#### 代码实现

<img src="D:\Document\YFY\markdown\深度学习\image\DenseNet.png" style="zoom:50%;" />

​	同样地，我们首先定义稠密层DenseBlock。如上图，它通过不断的卷积为通道数为32大小的特征图，同时不断concat（拼接）到原来输入的特征图上面。通过控制输入通道，输出通道和累加次数即可确定最终输出的通道数，这个过程并不改变特征图大小。在代码中通过实现DenseLayer来卷积出一个out_channels的特征图，其中有一个特点就是采用BN+ReLU+Conv，如果我们采用Conv+ReLU+BN的顺序，在concat（拼接）过程当中会导致两者批归一化不一致，相反交换顺序则不会有这样的问题。

```python
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
        for layer in self.layers:				# 循环num次即最后输出通道数为in_channels + out_channels * num
            x1 = layer(x)						# [batch_size, out_channels, w, w]
            x = torch.concat([x, x1], dim=1)   	# [batch_size, in_channels + out_channels, w, w]
        return x
```

​	之后定义DenseNet，其中主要结构划分如上图所示，其中实现了TransitionBlock。这个主要用于减少特征图的大小，也可以减少特征图的通道数，其代码如下。

```python
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

    def forward(self, x):			# [batch_size, 1, 224, 224]
        x = self.conv(x)			# [batch_size, 64, 56, 56]
        x = self.DenseBlock1(x)		# [batch_size, 128, 28, 28]
        x = self.DenseBlock2(x)		# [batch_size, 256, 14, 14]
        x = self.DenseBlock3(x)		# [batch_size, 512, 7, 7]
        x = self.DenseBlock4(x)		# [batch_size, 1024, 7, 7]
        x = self.fc(x)				# [batch_size, 10]
        return x
```

#### 实验结果

​	同样在选取学习率为1e-3和batch_size为64，每迭代一次学习率变为原来的0.9倍和0.2的平滑标签，迭代20次的情况下测试集的准确率能达到0.9431，和ResNet的准确率几乎一样。训练结果如下图所示。

<center class="LeNet">
<img src="D:\Document\YFY\markdown\深度学习\image\Acc_DenseNet.png" style="zoom:57%;" >
<img src="D:\Document\YFY\markdown\深度学习\image\Loss_DenseNet.png" style="zoom:60%;" >
</center>

## 总结

​	经典卷积神经网络是深度学习发展过程中的重要里程碑，包括LeNet、AlexNet、VGG、GoogLeNet、ResNet和DenseNet等模型。这些模型各具特色，但都以卷积层、池化层、激活函数和全连接层等模块为基础构建。它们使用大量图像数据进行训练，并在许多计算机视觉任务中取得了出色表现。

​	LeNet和AlexNet是最早的深度学习模型之一，经典卷积神经网络是深度学习发展过程中的重要里程碑。VGG通过使用小尺寸卷积核来提高模型性能，为后面的网络模型提供了可参考的设计思路。GoogLeNet引入了Inception模块，使得模型结构具备了很好的可扩展性。ResNet则通过残差连接解决了深度神经网络训练中梯度消失和退化问题，在各种计算机视觉任务中表现优异。DenseNet相比于其他模型所具有的特点是其网络中的特征图可以访问到过去所有层的信息，从而极大地促进信息的传递与利用，增强了模型的表达能力和稳定性。

​	总的来说，这些经典卷积神经网络可以帮助我们在设计新的卷积神经网络时可以借鉴他们的经验，尝试改进模型结构、激活函数、正则化等方法以提高性能和泛化能力。No free lunch，这些卷积神经网络都有自己的特点，说不定在某些数据集上VGG的效果可能高于后面的网络。神经网络有太多的参数可以调整，每次调整效果可能大不相同，这也是被称为“炼丹”的原因。在设计自己的卷积神经网络的时候，完全可以借鉴这些神经网络优秀的地方，根据自己特定的数据集，以达到最好的效果，并不说一定与这些网络完全一样。

​	
