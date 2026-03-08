
import torch
from sklearn.metrics import confusion_matrix as calculate_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.datasets import ImageFolder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================
# 类别
# =========================
classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

# 论文风格字体
plt.rcParams.update({
    "font.size": 12,
    "font.family": "Times New Roman",
})


# =========================
# 数据集
# =========================
class MatImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, mat_folder, transform=None):
        self.image_dataset = ImageFolder(image_folder, transform=transform)

        self.mat_files = []
        for class_folder in sorted(os.listdir(mat_folder)):
            class_path = os.path.join(mat_folder, class_folder)
            if os.path.isdir(class_path):
                self.mat_files.extend(
                    [os.path.join(class_path, f)
                     for f in sorted(os.listdir(class_path))
                     if f.endswith('.mat')]
                )

        assert len(self.image_dataset) == len(self.mat_files), \
            f"图像数量 ({len(self.image_dataset)}) 与 mat 数量 ({len(self.mat_files)}) 不匹配！"

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]

        mat_file = self.mat_files[idx]
        mat_data = loadmat(mat_file)

        # 自动获取真实变量名
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if len(keys) == 0:
            raise ValueError(f"{mat_file} 中没有有效数据变量")

        one_d_feature = mat_data[keys[0]]

        one_d_tensor = torch.tensor(one_d_feature, dtype=torch.float32)

        # 保证形状为 [1, N]
        if one_d_tensor.dim() == 1:
            one_d_tensor = one_d_tensor.unsqueeze(0)
        elif one_d_tensor.dim() == 2 and one_d_tensor.shape[0] != 1:
            one_d_tensor = one_d_tensor.view(1, -1)

        return image, one_d_tensor, label


# =========================
# 加载测试数据
# =========================
def load_test_data(image_folder, one_d_folder, batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = MatImageDataset(image_folder, one_d_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    return dataloader, dataset.image_dataset.classes


# =========================
# 测试模型
# =========================
def test_model(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, features, labels in test_dataloader:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(images, features)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = calculate_confusion_matrix(
        y_true, y_pred,
        labels=list(range(len(classes)))
    )

    return cm, y_true, y_pred


# =========================
# 绘制混淆矩阵
# =========================
def plot_confusion_matrix(cm, classes, save_path):

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    from duomotaigruagfn import LightweightResNet18, LightweightResidual

    model = LightweightResNet18(LightweightResidual)
    model.load_state_dict(
        torch.load('D:/daima/resnet/duomotaiagfnpu.pth',
                   map_location=torch.device('cpu'))
    )

    test_image_folder = "D:/daima/resnet/puimage/test_images"
    test_one_d_folder = "D:/daima/resnet/pudata/test_data"

    test_dataloader, class_names = load_test_data(
        test_image_folder,
        test_one_d_folder
    )

    cm, y_true, y_pred = test_model(model, test_dataloader)

    print("混淆矩阵：")
    print(cm)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")

    plot_confusion_matrix(cm, classes,
                          "D:/daima/resnet/confusion_matrix.png")

    print("\n混淆矩阵已保存（600 dpi PNG）")

