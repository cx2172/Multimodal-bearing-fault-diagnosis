import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tiaoshi import LightweightResNet18, LightweightResidual
from model import ResNet18,Residual
from resnet50 import ResNet50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from EfficientNet import EfficientNet
# from googlenet import GoogLeNet
import os
# from GhostNetV3 import GhostNetV3
from mobilenet import MobileNetV4
from mfresnet50 import ImprovedMSCKE_CFMSS_ResNet50
from PP_LCNet import PP_LCNet


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_data_process():
    # 定义数据集的路径
    ROOT_TRAIN = r'D:/daima/resnet/puimage/test_images'

    # 定义数据集处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # 加载数据集
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0)
    return test_dataloader


def test_model_process(model, test_dataloader):
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = 'cuda'

    # 将模型放入到训练设备中
    model = model.to(device)

    # 用于计算评估指标的列表
    all_preds = []
    all_labels = []

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output = model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)

            # 保存预测值和真实标签
            all_preds.extend(pre_lab.cpu().numpy())
            all_labels.extend(test_data_y.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    cm = confusion_matrix(all_labels, all_preds)

    # 输出结果
    print("准确率: {:.4f}".format(accuracy))
    print("加权精确率: {:.4f}".format(precision))
    print("加权召回率: {:.4f}".format(recall))
    print("加权F1值: {:.4f}".format(f1))
    print("混淆矩阵:\n", cm)


if __name__ == "__main__":
    # 加载模型
    # model = ResNet18(Residual)
    model = LightweightResNet18(LightweightResidual)
    # model = ResNet50()
    # model = EfficientNet(num_classes=10, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2)
    # model = GoogLeNet()
    # model = GhostNetV3(num_classes=10, width_mult=1.3)
    # model = MobileNetV4(10)
    # model = PP_LCNet(num_classes=10, width_mult=1.0)
    # model = ImprovedMSCKE_CFMSS_ResNet50(num_classes=10)
    model.load_state_dict(torch.load('puGMA.pth'))

    # 获取测试数据
    test_dataloader = test_data_process()

    # 进行模型测试并计算指标
    test_model_process(model, test_dataloader)
