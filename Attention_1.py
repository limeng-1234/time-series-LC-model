import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_path_data = r"E:\project\PythonCode\ExplainGroup\paper_review\LSTM_pre\decision_lstm\01_tracks.csv"
file_path_tracksMeta = r"E:\project\PythonCode\ExplainGroup\paper_review\LSTM_pre\decision_lstm\01_tracksMeta.csv"

file_path_data = "E:\BaiduNetdiskDownload\highd-dataset-v1.0\data\\"  # 第一个斜杠转义
file_path_tracksMeta = "E:\BaiduNetdiskDownload\highd-dataset-v1.0\data\\"  # 第一个斜杠转义

window_length = 10
output_length = 0




#%%
def process_data(file_path, file_path_Meta, window_length=5, output_length=0):
    time_series_data = []
    target_data = []
    for i in range(1, 51):
        if i < 10:
            file_path_data = file_path + "0%d_tracks.csv" % i
            file_path_tracksMeta = file_path_Meta + "0%d_tracksMeta.csv" % i
        else:
            file_path_data = file_path + "%d_tracks.csv" % i
            file_path_tracksMeta = file_path_Meta + "%d_tracksMeta.csv" % i


        df = pd.read_csv(file_path_data)
        df_trackMeta = pd.read_csv(file_path_tracksMeta)
        # 一个dataframe，如果列drivingDirection为1，将对应的id列的数据保存到列表中
        id_list = []
        for index, row in df_trackMeta.iterrows():
            # 如果drivingDirection列的值为1，将对应的id列的值添加到列表中
            if row['drivingDirection'] == 1:
                id_list.append(row['id'])
        # 唯一的车辆id列表
        vehicle_ids = df['id'].unique()
        # 存储每个车辆的时序数据的列表


        # 记录一次换道、同向、max_id-min_id>0的换道车辆id
        LC_id = []
        # 记录同向、车道保持、最大侧向速度绝对值小于某个阈值的LH车辆id
        LH_id = []
        # 对于每个车辆
        for vehicle_id in vehicle_ids:
            # 确定同一驾驶方向内的车辆，标志都为1
            if vehicle_id in id_list:
            # if True:
            # 提取轨迹数据
                vehicle_data_id = df[df['id'] == vehicle_id][['laneId']].reset_index(drop=True)
                max_laneId = vehicle_data_id['laneId'].max()
                min_laneId = vehicle_data_id['laneId'].min()
                # 第一个判断：换一次道；第二个判断：同向的换道
                if df_trackMeta['numLaneChanges'][vehicle_id - 1] == 1 and (max_laneId-min_laneId) != 0:
                    max_index = vehicle_data_id['laneId'].idxmax()
                    vehicle_data = df[df['id'] == vehicle_id][['x', 'y', 'xVelocity', 'yVelocity']].reset_index(drop=True)
                    vehicle_data = vehicle_data.iloc[:max_index, :]

                    n_rows, n_cols = vehicle_data.shape
                    if n_rows < window_length * 4:
                        continue
                    # 有的数据长度不够，删除
                    vehicle_data = vehicle_data.iloc[::4, :]
                    label_tar = 1

                    LC_id.append(vehicle_id)
                    # 转换为numpy数组
                    vehicle_data = vehicle_data.values
                    # 计算滑动窗口的个数
                    num_windows = vehicle_data.shape[0] - (window_length + output_length) + 1
                    for i in range(num_windows):
                        # 滑动窗口的数据,减去最后一个时刻的值(类似于归一化)
                        window_data = vehicle_data[i:i + window_length] - vehicle_data[window_length - 1:window_length, :]
                        target = label_tar
                        # 添加到时序数据列表中
                        time_series_data.append(window_data)
                        target_data.append(target)

                elif df_trackMeta['numLaneChanges'][vehicle_id - 1] == 0:
                    vehicle_data = df[df['id'] == vehicle_id][['x', 'y', 'xVelocity', 'yVelocity']].reset_index(drop=True)
                    max_yVelocity = max(vehicle_data['yVelocity'].abs())

                    n_rows, n_cols = vehicle_data.shape
                    if n_rows < window_length * 4:
                        continue
                    vehicle_data = vehicle_data.iloc[::4, :]
                    if max_yVelocity > 0.3:
                        continue
                    label_tar = 0
                    LH_id.append(vehicle_id)

                    # 转换为numpy数组
                    vehicle_data = vehicle_data.values
                    num_windows = vehicle_data.shape[0] - (window_length + output_length) + 1
                    for i in range(num_windows):
                        # 滑动窗口的数据,减去最后一个时刻的值(类似于归一化)
                        window_data = vehicle_data[i:i + window_length] - vehicle_data[window_length - 1:window_length, :]
                        target = label_tar
                        # 添加到时序数据列表中
                        time_series_data.append(window_data)
                        target_data.append(target)
                else:
                    break
                a = 1
    return time_series_data, target_data
#%%
datapre = True
if datapre == True:
    time_series_data, target_data = process_data(file_path_data, file_path_tracksMeta, window_length=window_length, output_length=output_length)
# 可以使用下列代码将标签为0的数据随机采样，使其数量与标签为1的数据一样多：
#%% 训练模型
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
#加载数据
X = np.array(time_series_data)
y = np.array(target_data)
# 分割输入和输出

# 将数据转换为张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
torch.save(X, 'X_tensor.pt')
torch.save(y, 'y_tensor.pt')
#%%
# 将数据随机分割为训练数据和测试数据
X = torch.load('X_tensor.pt')
y = torch.load('y_tensor.pt')
train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 将数据分割为训练数据和测试数据
train_size = int(len(train_dataset) * 0.8)
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#定义LSTM网络
#定义LSTM网络
import torch.nn as nn
import torch.nn.functional as F

#%%
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, output_size):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        q = q.view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads).permute(0, 2, 1, 3)

        attention = torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.hidden_size // self.num_heads)
        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_size)
        out = self.fc(out)
        return out


class MultiHeadAttentionNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, output_size):
        super().__init__()
        self.attention = MultiHeadAttention(input_size, hidden_size, num_heads, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.attention(x, x, x)
        x = x[:, -1:, :]
        x = self.sigmoid(x)
        return x
#%%
input_size = X.shape[2]
hidden_size = 128
output_size = y.shape[2]
# Replace the LSTM network with the MultiHeadAttentionNet
model = MultiHeadAttentionNet(input_size, hidden_size, 8, output_size)

# 初始化模型

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
losses = []
Train = True
if Train == True:
    for epoch in range(num_epochs):
        for step, (X_batch, y_batch) in enumerate(train_loader):
            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch.squeeze())
            losses.append(loss.item())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 打印训练信息
            if (step + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], step:{step} Loss: {loss.item()}")
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'model_3.pt')
torch.save(model.state_dict(), 'model_3.pt')


#%% 绘图
# 绘制训练过程中的 loss 曲线
import time
plt.plot(losses)
plt.xlabel("Steps", fontsize=15)
plt.ylabel("Loss", fontsize=15)
# 设置坐标轴刻度标记的字体大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# 设置坐标轴线条粗细
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
plt.title("Training Loss")

current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
name = "train"
plt.savefig(f"{current_time+name}.pdf")
plt.show()

import time
import matplotlib.pyplot as plt
#%%
input_size = X.shape[2]
hidden_size = 128
output_size = y.shape[2]

model = MultiHeadAttentionNet(input_size, hidden_size, 8, output_size)
model.load_state_dict(torch.load('model_3.pt'))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = (outputs.squeeze().data > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

accuracy = correct / total
print('Accuracy: {:.2f}%'.format(accuracy * 100))