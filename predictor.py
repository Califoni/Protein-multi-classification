import torch
import torch.nn as nn
import torch.nn.init as init
from DataProcess import load_data,  make_ylabel
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, hamming_loss
import numpy as np
import os
from torchviz import make_dot
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk



class MultiLabelCNN(nn.Module):
    def __init__(self, input_dim=46, num_classes=4):
        super(MultiLabelCNN, self).__init__()

        # 定义卷积层和池化层
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_layer4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 将卷积层输出展平后接入全连接层
        # 计算经过三次卷积和池化后的特征维度
        self.flatten = nn.Flatten()
        self.fc_layer = nn.Sequential(
            nn.Linear(128, 64),  # 根据卷积和池化后的维度计算全连接层输入维度
            nn.ReLU(),
            nn.Linear(64, num_classes)  # 输出层，对应四个标签
        )
        # 初始化模型参数
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear:
            # 使用均匀分布初始化卷积层和全连接层的权重
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # 将偏置初始化为零
                init.constant_(m.bias, 0.0)

    def forward(self, x):
        '''
        x: 输入形状为 (batch_size, 1, input_dim)
        '''
        x = x.unsqueeze(1)  # 添加一维通道信息，变为 (batch_size, 1, input_dim)
        x = self.conv_layer1(x)  # 第一次卷积和池化
        x = self.conv_layer2(x)  # 第二次卷积和池化
        x = self.conv_layer3(x)  # 第三次卷积和池化
        x = self.conv_layer4(x)
        x = self.flatten(x)  # 展平
        x = self.fc_layer(x)  # 全连接层
        return x


def load_model(model_path):
    model = MultiLabelCNN()
    if os.path.exists(model_path):
        # model=torch.load(model_path)
        # for name, param in model.items():
        #     print(f"Parameter name: {name}, Size: {param.size()}")
        model.load_state_dict(torch.load(model_path))
        return model
    else:
        print("[ERROR]\tModel not found at the specified path.")
        return None


def browse_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
    entry.delete(0, tk.END)
    entry.insert(0, file_path)


def browse_model(entry):
    file_path = filedialog.askopenfilename(filetypes=[("PTH files", "*.pth")])
    entry.delete(0, tk.END)
    entry.insert(0, file_path)


def predict_threshold(model, inputs, threshold=0.5):
    model.eval()
    inputs = inputs
    with torch.no_grad():
        test_outputs = model(inputs)
        probs = torch.sigmoid(test_outputs)  # 计算sigmoid概率
        binary_preds = (probs >= threshold).float()  # 根据阈值进行二分
    return binary_preds


def map_labels(vector):
    labels = ["A", "C", "M", "S"]
    mapped_labels = [labels[i] for i, val in enumerate(vector) if val == 1]
    return ",".join(mapped_labels)


def display_results(results, targets):
    # 创建新窗口来展示结果
    result_window = tk.Toplevel(root)
    result_window.title("Prediction Results")

    # 创建 Treeview 组件
    tree = ttk.Treeview(result_window)
    tree["columns"] = ("results", "targets")

    # 设置列名
    tree.heading("#0", text="Sample")
    tree.heading("results", text="Results")
    tree.heading("targets", text="Targets")

    # 添加滑动滚轮
    scroll = ttk.Scrollbar(result_window, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)

    # 将结果添加到 Treeview 中
    for idx, (sample_result, target) in enumerate(zip(results, targets)):
        mapped_result = map_labels(sample_result)
        mapped_target = map_labels(target)
        tree.insert("", idx, text=f"Sample {idx + 1}", values=(mapped_result, mapped_target))

    # 添加图表说明
    label_info = tk.Label(result_window, text="A:acetyllysine(乙酰化) ,C:crotonyllysine(巴豆酰化) ,M:methyllysine(甲基化) ,S:succinyllysine(琥珀酰化) ")
    label_info.grid(row=1, column=0, columnspan=2)

    # 将 Treeview 和滑动滚轮添加到窗口中，并设置自适应大小
    tree.grid(row=0, column=0, sticky="nsew")
    scroll.grid(row=0, column=1, sticky="ns")
    result_window.grid_rowconfigure(0, weight=1)
    result_window.grid_columnconfigure(0, weight=1)

    # 设置每行内容居中显示
    for col in tree["columns"]:
        tree.heading(col, anchor=tk.CENTER)
        tree.column(col, anchor=tk.CENTER)

    # 调整窗口大小以适应内容
    result_window.update()


def integrate_and_predict():
    test_file_path = test_entry.get()
    model_file_path = model_entry.get()

    # 加载测试集数据和模型
    X_test, targets_test = load_data(test_path=test_file_path)
    model = load_model(model_file_path)
    # 使用模型进行预测
    res=predict_threshold(model=model,inputs=X_test)
    targets_test=make_ylabel(targets_test)
    display_results(res,targets_test)



# 创建GUI界面
root = tk.Tk()
root.title("MultiLabelCNN Prediction")

# 添加测试集文件路径选择框
test_label = tk.Label(root, text="Choose .mat Test Set:")
test_label.grid(row=0, column=0)
test_entry = tk.Entry(root, width=100)
test_entry.grid(row=0, column=1)
test_button = tk.Button(root, text="Browse", command=lambda: browse_file(test_entry))
test_button.grid(row=0, column=2)

# 添加模型文件路径选择框
model_label = tk.Label(root, text="Choose .pth Model:")
model_label.grid(row=1, column=0)
model_entry = tk.Entry(root, width=100)
model_entry.grid(row=1, column=1)
model_button = tk.Button(root, text="Browse", command=lambda: browse_model(model_entry))
model_button.grid(row=1, column=2)

# 添加预测按钮
predict_button = tk.Button(root, text="Predict", command=integrate_and_predict)
predict_button.grid(row=2, column=1)

root.mainloop()
