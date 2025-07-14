import torch
from torch import nn
from torchsummary import summary
from ema import EMA  # 确保 EMA 类在实际代码中已实现
# from CBAM import Cbam


class LightweightResidual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(LightweightResidual, self).__init__()
        self.ReLU = nn.LeakyReLU(0.1, inplace=True)
        self.out_channels = num_channels // 4

        self.conv_3x3 = nn.Conv2d(in_channels=input_channels, out_channels=self.out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv_5x5 = nn.Conv2d(in_channels=input_channels, out_channels=self.out_channels, kernel_size=5, padding=2, stride=strides)
        self.conv_7x7 = nn.Conv2d(in_channels=input_channels, out_channels=self.out_channels, kernel_size=7, padding=3, stride=strides)
        self.conv_11x11 = nn.Conv2d(in_channels=input_channels, out_channels=self.out_channels, kernel_size=11, padding=5, stride=strides)

        self.bn = nn.BatchNorm2d(num_channels)
        self.add = EMA(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        for conv in [self.conv_3x3, self.conv_5x5, self.conv_7x7, self.conv_11x11]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        y1 = self.conv_3x3(x)
        y2 = self.conv_5x5(x)
        y3 = self.conv_7x7(x)
        y4 = self.conv_11x11(x)

        y = torch.cat((y1, y2, y3, y4), dim=1)
        y = self.bn(y)
        y = self.ReLU(y)
        y = self.add(y)
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(y + x)
        return y


class LightweightResNet18(nn.Module):
    def __init__(self, Residual):
        super(LightweightResNet18, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b2 = nn.Sequential(Residual(64, 64, use_1conv=False, strides=1),
                                Residual(64, 64, use_1conv=False, strides=1))

        self.b3 = nn.Sequential(Residual(64, 128, use_1conv=True, strides=2),
                                Residual(128, 128, use_1conv=False, strides=1))

        self.b4 = nn.Sequential(Residual(128, 256, use_1conv=True, strides=2),
                                Residual(256, 256, use_1conv=False, strides=1))

        self.b5 = nn.Sequential(Residual(256, 512, use_1conv=True, strides=2),
                                Residual(512, 512, use_1conv=False, strides=1))

        self.image_features = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256)  # 图像特征输出为 256 维
        )

        # 一维信号处理 - 使用卷积和 GRU
        self.signal_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.signal_features = nn.Sequential(
            nn.Linear(256 * 2, 256),  # GRU 输出为 256 * 2 (双向)
            nn.ReLU()
        )


        # 融合后的分类器
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 128),  # 输入为图像+信号特征
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10)  # 假设输出为 10 类
        )

    def forward(self, image, signal):
        # 图像特征提取
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        image_features = self.image_features(x)

        # 一维信号特征提取 (卷积 + GRU)
        signal = self.signal_conv1d(signal)
        signal = signal.permute(0, 2, 1)  # 调整维度以适配 GRU 输入 (batch_size, seq_len, input_size)
        gru_out, _ = self.gru(signal)
        gru_out = gru_out[:, -1, :]  # 取 GRU 的最后一个时间步输出
        signal_features = self.signal_features(gru_out)

        # 特征融合
        features = torch.cat((image_features, signal_features), dim=1)

        # 分类
        output = self.classifier(features)
        return output


# 测试模型
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightResNet18(LightweightResidual).to(device)

    # 示例输入
    image_input = torch.randn(8, 3, 224, 224).to(device)  # 图像输入
    signal_input = torch.randn(8, 1, 512).to(device)  # 一维信号输入

    # 测试前向传播
    output = model(image_input, signal_input)
    print("Model output shape:", output.shape)
