import copy
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
from torch.utils.data import Dataset  # 这里添加导入 Dataset
import numpy as np
import matplotlib.pyplot as plt
from duomotaigru import LightweightResNet18, LightweightResidual
import torch.nn as nn
import pandas as pd
from scipy.io import loadmat
# from jiangzao import LightweightResNet18,LightweightResidual


class MatImageDataset(Dataset):  # 使用 Dataset 作为基类
    def __init__(self, image_folder, mat_folder, transform=None):
        self.image_dataset = ImageFolder(image_folder, transform=transform)
        self.mat_folder = mat_folder

        # 收集所有 .mat 文件的路径，假设图像文件与 .mat 文件顺序对齐
        self.mat_files = []
        for class_folder in sorted(os.listdir(mat_folder)):
            class_path = os.path.join(mat_folder, class_folder)
            if os.path.isdir(class_path):
                self.mat_files.extend(
                    [os.path.join(class_path, f) for f in sorted(os.listdir(class_path)) if f.endswith('.mat')]
                )

        assert len(self.image_dataset) == len(self.mat_files), (
            f"图像样本数量 ({len(self.image_dataset)}) 与 .mat 文件数量 ({len(self.mat_files)}) 不匹配！"
        )

        self.transform = transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # 获取图像和对应的标签
        image, label = self.image_dataset[idx]

        # 加载对应的 .mat 文件中的一维特征
        mat_file = self.mat_files[idx]
        mat_data = loadmat(mat_file)
        one_d_feature = mat_data['X']  # 假设特征存储在 mat 文件的某个键中


        # 将一维特征转换为张量
        one_d_tensor = torch.tensor(one_d_feature, dtype=torch.float32)

        # 如果特征是一维的（[512]），则将其转换为 [1, 512]
        if one_d_tensor.dim() == 1:  # 如果是 [512]
            one_d_tensor = one_d_tensor.unsqueeze(0)  # 将其变为 [1, 512]
        elif one_d_tensor.dim() == 2 and one_d_tensor.shape[0] == 1:  # 如果是 [1, 512]
            pass  # 如果已经是 [1, 512]，不需要处理
        else:
            raise ValueError(f"Expected shape [1, 512], got {one_d_tensor.shape}")

        # 确保形状是 [1, 512]，即 1 个通道，512 长度
        assert one_d_tensor.dim() == 2 and one_d_tensor.shape[
            0] == 1, f"Expected shape [1, 512], got {one_d_tensor.shape}"

        return image, one_d_tensor, label

def train_val_data_process(image_folder, one_d_data_folder, batch_size=64):
    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = MatImageDataset(image_folder, one_d_data_folder, transform=train_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # 正确地分割数据集
    datasets = Data.random_split(dataset, [train_size, val_size])

    train_dataset = datasets[0]
    val_dataset = datasets[1]

    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_dataloader, val_dataloader



def train_model_process(model, train_dataloader, val_dataloader, num_epochs, model_save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        for step, (b_x, b_feature, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_feature = b_feature.to(device)  # 一维特征
            b_y = b_y.to(device)

            optimizer.zero_grad()
            output = model(b_x, b_feature)  # 将图像和特征一起传入模型
            loss = criterion(output, b_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(torch.argmax(output, dim=1) == b_y.data)
            train_num += b_x.size(0)

        # 验证阶段
        model.eval()
        for step, (b_x, b_feature, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_feature = b_feature.to(device)
            b_y = b_y.to(device)

            with torch.no_grad():
                output = model(b_x, b_feature)
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

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_save_path)

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


if __name__ == "__main__":
    num_epochs = 50
    batch_size = 64
    model_save_path = "D:/daima/resnet/pu.pth"
    plot_save_path = "D:/daima/resnet/pu.png"

    # 数据集路径
    image_folder = "D:/daima/resnet/puimage/train_images"
    one_d_data_folder = "D:/daima/resnet/pudata/train_data"  # 假设这里存储 .mat 文件

    # 数据加载
    train_loader, val_loader = train_val_data_process(image_folder, one_d_data_folder, batch_size)

    # 定义模型
    feature_extractor = LightweightResNet18(LightweightResidual)  # 假设已有的模型
    model = LightweightResNet18(LightweightResidual)

    # 训练模型
    train_process = train_model_process(model, train_loader, val_loader, num_epochs, model_save_path)

    # 保存训练过程图
    matplot_acc_loss(train_process, plot_save_path)

    print("训练完成，模型已保存！")
