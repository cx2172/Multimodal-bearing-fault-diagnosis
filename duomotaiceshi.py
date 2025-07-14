import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.io import loadmat
from duomotaigru import LightweightResNet18, LightweightResidual
from torchvision.datasets import ImageFolder
import os


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
        return image, one_d_tensor, label

def load_test_data(image_folder, one_d_folder, batch_size=1):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = MatImageDataset(image_folder, one_d_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader

def test_model(model, test_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, features, labels in test_dataloader:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(images, features)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print("准确率: {:.4f}".format(accuracy))
    print("加权精确率: {:.4f}".format(precision))
    print("加权召回率: {:.4f}".format(recall))
    print("加权F1值: {:.4f}".format(f1))
    print("混淆矩阵:\n", conf_matrix)

if __name__ == "__main__":
    # 模型加载
    model_path = "D:/daima/resnet/zhongbei.pth"
    model = LightweightResNet18(LightweightResidual)
    model.load_state_dict(torch.load(model_path))

    # 数据路径
    test_image_folder = "D:/daima/resnet/zhongbei/test_images"
    test_one_d_folder = "D:/daima/resnet/zhongbeidata/test_data"

    # 加载测试数据
    test_dataloader = load_test_data(test_image_folder, test_one_d_folder)

    # 测试模型并输出结果
    test_model(model, test_dataloader)
