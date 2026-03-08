import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入你的EMA模型类
from train_resnet18_ema_ablation import ResNet18_EMA_Ablation


def test_data_process():
    # 定义数据集的路径
    ROOT_TRAIN = r'D:/daima/resnet/duomotai/test_images'

    # 定义数据集处理方法变量
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 加载数据集
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)

    test_dataloader = Data.DataLoader(
        dataset=test_data,
        batch_size=32,  # 可以适当调整batch_size
        shuffle=False,  # 测试时通常不需要shuffle
        num_workers=2
    )
    return test_dataloader


def test_model_process(model, test_dataloader, device):
    # 将模型放入到设备中
    model = model.to(device)

    # 用于计算评估指标的列表
    all_preds = []
    all_labels = []

    # 只进行前向传播计算，不计算梯度
    model.eval()
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将数据放到设备上
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 前向传播
            output = model(test_data_x)

            # 获取预测结果
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
    print("=" * 50)
    print("测试结果:")
    print("准确率: {:.4f}".format(accuracy))
    print("加权精确率: {:.4f}".format(precision))
    print("加权召回率: {:.4f}".format(recall))
    print("加权F1值: {:.4f}".format(f1))
    print("混淆矩阵:")
    print(cm)
    print("=" * 50)

    return accuracy, precision, recall, f1, cm


def test_single_model(G_value, model_path, num_classes=10):
    """测试单个G值的模型"""
    print(f"\n{'=' * 60}")
    print(f"正在测试 G={G_value} 的模型")
    print(f"{'=' * 60}")

    # 1. 实例化模型（注意：这里的G值需要和训练时保持一致）
    model = ResNet18_EMA_Ablation(G=G_value, num_classes=num_classes)

    # 2. 加载训练好的权重
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"成功加载模型权重: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

    # 3. 获取测试数据
    test_dataloader = test_data_process()

    # 4. 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 5. 测试模型
    results = test_model_process(model, test_dataloader, device)

    return results


def test_all_models():
    """测试所有G值的模型"""
    G_values = [2, 4, 8, 16, 32]
    results_dict = {}

    for G in G_values:
        # 构建模型路径（根据你的训练代码）
        model_path = f"D:/daima/resnet/xichu_resnet18_ema_G{G}.pth"

        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在 - {model_path}")
            continue

        # 测试该模型
        results = test_single_model(G, model_path)
        if results is not None:
            results_dict[G] = results

    # 汇总比较结果
    print("\n" + "=" * 60)
    print("所有模型测试结果汇总:")
    print("=" * 60)

    print(f"{'G值':<8} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1值':<10}")
    print("-" * 50)

    best_accuracy = 0
    best_G = None

    for G, (acc, pre, rec, f1, cm) in results_dict.items():
        print(f"{G:<8} {acc:.4f}     {pre:.4f}     {rec:.4f}     {f1:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_G = G

    print("-" * 50)
    print(f"最佳模型: G={best_G}, 准确率={best_accuracy:.4f}")

    return results_dict


if __name__ == "__main__":
    # 方式1: 测试单个G值的模型
    # G_to_test = 8  # 指定要测试的G值
    # model_path = f"D:/daima/resnet/xichu_resnet18_ema_G{G_to_test}.pth"
    # test_single_model(G_to_test, model_path)

    # 方式2: 测试所有G值的模型并进行比较
    results = test_all_models()