import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix as calculate_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.io import loadmat
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
from torchvision.datasets import ImageFolder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义类别
classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']


# 自定义数据集类（与您的训练代码一致）
class MatImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, mat_folder, transform=None):
        self.image_dataset = ImageFolder(image_folder, transform=transform)
        self.mat_folder = mat_folder

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
        image, label = self.image_dataset[idx]
        mat_file = self.mat_files[idx]
        mat_data = loadmat(mat_file)
        one_d_feature = mat_data['X']

        one_d_tensor = torch.tensor(one_d_feature, dtype=torch.float32)
        if one_d_tensor.dim() == 1:
            one_d_tensor = one_d_tensor.unsqueeze(0)

        return image, one_d_tensor, label


# 测试数据处理
def load_test_data(image_folder, one_d_folder, batch_size=1):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = MatImageDataset(image_folder, one_d_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader, dataset.image_dataset.classes


# 修复的多模态 Grad-CAM 实现
class MultiModalGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # 注册钩子到图像分支的最后一个残差块
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 注册钩子到图像分支的最后一个残差块
        target_layer = self.model.b5[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_image, input_signal, target_class=None):
        # 确保输入张量需要梯度
        input_image = input_image.clone().detach().requires_grad_(True)

        self.model.zero_grad()

        # 前向传播
        output = self.model(input_image, input_signal)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 创建one-hot编码
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        # 反向传播 - 使用 retain_graph=True
        output.backward(gradient=one_hot, retain_graph=True)

        # 生成图像热力图
        if self.gradients is not None and self.activations is not None:
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = torch.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            image_cam = cam.squeeze().cpu().numpy()
        else:
            image_cam = None

        return {
            'image_cam': image_cam,
            'target_class': target_class,
            'prediction_confidence': torch.softmax(output, dim=1)[0, target_class].item()
        }


# 修复的可视化函数 - 使用Matplotlib替代OpenCV
def visualize_multimodal_heatmap(original_image, original_signal, heatmaps, true_label, pred_label, save_path):
    image_cam = heatmaps['image_cam']

    # 准备图像
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 第一行：图像相关
    img = original_image.permute(1, 2, 0).numpy()

    # 原始图像
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f'Original Image\nTrue: {classes[true_label]}')
    axes[0, 0].axis('off')

    if image_cam is not None:
        # 调整热力图大小以匹配图像
        from scipy.ndimage import zoom
        zoom_factors = (img.shape[0] / image_cam.shape[0], img.shape[1] / image_cam.shape[1])
        heatmap_resized = zoom(image_cam, zoom_factors)

        # 纯热力图
        im1 = axes[0, 1].imshow(heatmap_resized, cmap='jet')
        axes[0, 1].set_title('Image Attention Heatmap')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])

        # 图像叠加
        axes[0, 2].imshow(img, alpha=0.7)
        im2 = axes[0, 2].imshow(heatmap_resized, cmap='jet', alpha=0.5)
        axes[0, 2].set_title(f'Image Overlay\nPred: {classes[pred_label]}')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2])
    else:
        for i in range(1, 3):
            axes[0, i].text(0.5, 0.5, 'Heatmap\nNot Available',
                            ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].set_title('Image Attention')
            axes[0, i].axis('off')

    # 第二行：信号相关
    signal_data = original_signal.squeeze().numpy()

    # 原始信号
    axes[1, 0].plot(signal_data)
    axes[1, 0].set_title('Original 1D Signal')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True)

    # 信号频谱
    fft_signal = np.fft.fft(signal_data)
    freq = np.fft.fftfreq(len(signal_data))
    axes[1, 1].plot(freq[:len(freq) // 2], np.abs(fft_signal)[:len(fft_signal) // 2])
    axes[1, 1].set_title('Signal Frequency Spectrum')
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].grid(True)

    # 置信度信息
    conf = heatmaps.get('prediction_confidence', 0)
    axes[1, 2].text(0.5, 0.5,
                    f'Prediction Info\nTrue: {classes[true_label]}\nPred: {classes[pred_label]}\nConfidence: {conf:.3f}',
                    ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12,
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
    axes[1, 2].set_title('Classification Result')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# 修复的测试模型函数
def test_model_with_multimodal_heatmaps(model, test_dataloader, output_dir="multimodal_heatmaps", num_heatmaps=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 确保模型处于训练模式以计算梯度
    model.train()

    os.makedirs(output_dir, exist_ok=True)

    # 初始化多模态Grad-CAM
    cam = MultiModalGradCAM(model)

    y_true = []
    y_pred = []
    heatmap_count = 0

    # 不使用 torch.no_grad()，因为我们需要计算梯度
    for batch_idx, (images, features, labels) in enumerate(test_dataloader):
        images = images.to(device)
        features = features.to(device)
        labels = labels.to(device)

        # 计算输出用于混淆矩阵（不需要梯度）
        with torch.no_grad():
            outputs = model(images, features)
            preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        # 为前几个样本生成热力图
        if heatmap_count < num_heatmaps:
            for i in range(len(images)):
                if heatmap_count >= num_heatmaps:
                    break

                try:
                    # 生成多模态热力图
                    heatmaps = cam.generate_cam(
                        images[i:i + 1], features[i:i + 1], preds[i].item()
                    )

                    # 可视化并保存
                    true_label = labels[i].item()
                    pred_label = preds[i].item()

                    save_path = os.path.join(
                        output_dir,
                        f"heatmap_{heatmap_count}_true_{classes[true_label]}_pred_{classes[pred_label]}.png"
                    )

                    visualize_multimodal_heatmap(
                        images[i].cpu(),
                        features[i].cpu(),
                        heatmaps,
                        true_label,
                        pred_label,
                        save_path
                    )

                    print(f"Generated multimodal heatmap {heatmap_count + 1}/{num_heatmaps}: {save_path}")
                    heatmap_count += 1

                except Exception as e:
                    print(f"Error generating heatmap {heatmap_count}: {e}")
                    continue

    # 构建混淆矩阵
    cm = calculate_confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    return cm, y_true, y_pred


# 修复的可视化混淆矩阵函数
def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))

    # 确保混淆矩阵是numpy数组格式
    cm_array = np.array(cm, dtype=int)

    plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm_array.max() / 2.

    # 添加数值标注
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            plt.text(j, i, format(cm_array[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_array[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # 保存图像而不是显示
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存至: {save_path}")
    plt.close()


# 主程序
if __name__ == "__main__":
    # 导入您的模型
    from duomotaigru import LightweightResNet18, LightweightResidual

    try:
        # 加载模型和权重
        model = LightweightResNet18(LightweightResidual)
        model.load_state_dict(torch.load('D:/daima/resnet/duomotaiGRU.pth', map_location=torch.device('cpu')))
        print("模型和权重加载成功！")

        # 数据路径
        test_image_folder = "D:/daima/resnet/duomotai/test_images"
        test_one_d_folder = "D:/daima/resnet/duomotaidata/test_data"

        # 加载测试数据
        test_dataloader, class_names = load_test_data(test_image_folder, test_one_d_folder)
        print(f"测试数据加载成功！共 {len(test_dataloader)} 个样本")

        # 使用多模态热力图测试
        print("开始生成多模态注意力热力图...")
        confusion_matrix, y_true, y_pred = test_model_with_multimodal_heatmaps(
            model, test_dataloader, output_dir="D:/daima/resnet/multimodal_heatmaps", num_heatmaps=10
        )

        # 输出性能指标
        print("\n性能指标：")
        print("混淆矩阵：")
        print(confusion_matrix)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

        print(f"准确率: {accuracy:.4f}")
        print(f"加权精确率: {precision:.4f}")
        print(f"加权召回率: {recall:.4f}")
        print(f"加权F1值: {f1:.4f}")

        # 绘制混淆矩阵 - 保存为文件
        plot_confusion_matrix(confusion_matrix, classes, save_path='D:/daima/resnet/confusion_matrix.png')

        print(f"\n多模态热力图已保存到: D:/daima/resnet/multimodal_heatmaps")
        print(f"混淆矩阵已保存到: D:/daima/resnet/confusion_matrix.png")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback

        traceback.print_exc()