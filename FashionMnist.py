import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import time

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
)
# 加载MNIST数据集的测试集
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# batch大小
batch_size = 64

# 将文件打包成一个个batch的样子
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 遍历dataloader
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)  # 每个batch数据的形状
    print("Shape of y: ", y.shape)  # 每个batch标签的形状  标签为【64*1】 即每一个图片有一个标签
    break


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # 加入flat层 将图像转化为列向量
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(1)
        )

    def forward(self, x):
        '''

        :param x: 扁平化之后的输入
        :return: 前向传播的输出
        '''
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(train_dataLoader, model, loss_fn, optimizer):
    '''
    进行神经网络的训练
    :param train_dataLoader: 训练集的数据
    :param model:训练的模型结构
    :param loss_fn:损失函数
    :param optimizer:优化方式
    :return:
    '''
    model.train()

    for images, labels in train_dataLoader:
        # images = images.to(device)
        # labels = labels.to(device)
        pred = model(images)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        # 步进优化器
        optimizer.step()


def Test(test_dataloader, model, loss_fn):
    '''

    :param test_dataloader: 测试集的dataloader
    model:              网络模型
    loss_fn:            损失函数
    '''
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for image, label in test_dataloader:
            pred = model(image)
            test_loss += loss_fn(pred, label)
            correct += (pred.argmax(1) == label).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 打印使用的是gpu/cpu
    print("Using {} device".format(device))

    # 实例化模型
    model = NeuralNetwork().to(device)
    # 打印模型结构
    print(model)

    # 遍历dataloader
    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)  # 每个batch数据的形状
        print("Shape of y: ", y.shape)  # 每个batch标签的形状  标签为【64*1】 即每一个图片有一个标签
        break
    # ==============================================模型训练=============================================================
    # ----------模型参数设置--------------
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    epochs = 5

    for t in range(epochs):  # 迭代epochs次
        print(f"Epoch {t + 1}\n-------------------------------")
        # 训练
        train(train_dataloader, model, loss_fn, optimizer)

        # 当次训练完后测试目前模型在测试集上的表现
        Test(test_dataloader, model, loss_fn)
    print("Done!")
    path = './model'
    if not os.path.exists(path):
        os.mkdir(path)
    st = time.strftime("%Y-%m-%d,%H-%M", time.localtime())
    torch.save(model.state_dict(), path + '/model{}.pth'.format(st))
#----------------------------------载入模型并预测--------------------------------------------------------#
    # model.load_state_dict(torch.load("./model/model1.pth"))
    # print(model)
    # for name, parameters in model.named_parameters():
    #     print(name, ';', parameters.size())
    # test(test_dataloader, model, loss_fn=nn.CrossEntropyLoss())
