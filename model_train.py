import copy
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from tiaoshi import LightweightResNet18,LightweightResidual
# from model import ResNet18,Residual
# from resnet50 import ResNet50
from googlenet import GoogLeNet
# from EfficientNet import EfficientNet
# from convnext import convnext_tiny
from mobilenet import MobileNetV4
import torch.nn as nn
import pandas as pd
# from PP_LCNet import PP_LCNet
# from GhostNetV3 import GhostNetV3
# from mfresnet50 import ImprovedMSCKE_CFMSS_ResNet50

def train_val_data_process():
    # 定义数据集的路径
    ROOT_TRAIN = 'D:/daima/resnet/puimage/train_images'

    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # 加载数据集
    train_data = ImageFolder(ROOT_TRAIN, transform=train_transform)

    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=2)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=64,
                                     shuffle=True,
                                     num_workers=2)

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs, model_save_path):
    device = torch.device("cuda:0")
    print(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        # 训练阶段
        model.train()
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            output = model(b_x)
            loss = criterion(output, b_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(torch.argmax(output, dim=1) == b_y.data)
            train_num += b_x.size(0)

        # 验证阶段
        model.eval()
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            with torch.no_grad():
                output = model(b_x)
                loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(torch.argmax(output, dim=1) == b_y.data)
            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} 训练损失: {:.4f} 训练准确率: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} 验证损失: {:.4f} 验证准确率: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 更新最佳模型权重
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # 更新调度器
        scheduler.step()

        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 保存最佳模型的权重
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_save_path)  # 保存最佳模型的状态字典

    return pd.DataFrame(data={"epoch": range(num_epochs),
                              "train_loss_all": train_loss_all,
                              "val_loss_all": val_loss_all,
                              "train_acc_all": train_acc_all,
                              "val_acc_all": val_acc_all})



def matplot_acc_loss(train_process, save_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(save_path)
    plt.close()  # Close the figure window


if __name__ == '__main__':
    # Removed the loop for multiple trials
    num_epochs = 50

    # 加载模型
    # model = ResNet18(Residual)
    model = LightweightResNet18(LightweightResidual)
    # model = ResNet50()
    # model = GoogLeNet()
    # model = convnext_tiny(10)
    # model = EfficientNet(num_classes=10, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2)
    # model = MobileNetV4(10)
    # model = PP_LCNet(num_classes=10, width_mult=1.0)
    # model = GhostNetV3(num_classes=10, width_mult=1.3)
    # model = ImprovedMSCKE_CFMSS_ResNet50(num_classes=10)




    # 加载数据集
    train_data, val_data = train_val_data_process()

    # 训练模型
    model_save_path = "D:/daima/resnet/puGMA.pth"
    train_process = train_model_process(model, train_data, val_data, num_epochs=num_epochs,
                                        model_save_path=model_save_path)

    # 保存图表
    plot_save_path = "D:/daima/resnet/puGMA.png"
    matplot_acc_loss(train_process, plot_save_path)

    print("Training completed.")


