import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init
from DataProcess import load_data, data_resample, make_ylabel
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, hamming_loss
import numpy as np
import os
from torchviz import make_dot



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
            nn.Linear(128, 32),  # 根据卷积和池化后的维度计算全连接层输入维度
            nn.ReLU(),
            nn.Linear(32, num_classes)  # 输出层，对应四个标签
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        x = x.unsqueeze(1)  # 添加一维通道信息，变为 (batch_size, 1, input_dim)
        x = self.conv_layer1(x)  # 第一次卷积和池化
        x = self.conv_layer2(x)  # 第二次卷积和池化
        x = self.conv_layer3(x)  # 第三次卷积和池化
        x = self.conv_layer4(x)
        x = self.flatten(x)  # 展平
        x = self.fc_layer(x)  # 全连接层
        return x


def class_weight(target_train):
    sum=torch.tensor([0,0,0,0],dtype=torch.float32)
    for t in target_train:
        sum+=t
    # 对于每一个类别的正类权重=0.5/正样本占比
    ratios=sum/len(target_train)
    print("ratios:\t",ratios)
    weights=0.5/ratios
    print("weights:\t",weights)
    return weights


def class_weight_all(Y_train,a):
    each_label=np.zeros(11)
    for t in Y_train:
        each_label[t-1]+=1
    print(each_label)
    each_label/=np.sum(each_label)
    print(each_label)
    weight=1/each_label*a
    print(weight)
    return weight


def data_preprocess(batch_size=128, shuffle=True,sample_strategy=1):
    # load data
    X_train, X_test, Y_train, Y_test = load_data("mydata/PSTAAP_train.mat", "mydata/PSTAAP_test.mat")
    X_ori=X_train
    Y_ori=make_ylabel(Y_train)
    X_train, Y_train = data_resample(X_train, Y_train, sample_strategy=sample_strategy)
    # 转换为适合模型输入的形状
    targets_train = make_ylabel(Y_train)
    targets_test = make_ylabel(Y_test)
    print("[INFO]\tinputs.shape:", X_train.shape)
    train_dataset = list(zip(X_train, targets_train,Y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return X_train, targets_train, X_test, targets_test, train_loader,X_ori,Y_ori,Y_train


# 训练过程中计算单个样本的平均二元交叉熵损失
def compute_avg_bce_loss(model, inputs, targets):
    outputs = model(inputs)
    bce_losses = nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
    avg_bce_loss = bce_losses.mean(dim=1)  # 按样本求平均
    print(bce_losses)
    return avg_bce_loss


def train_for_test(model, train_loader, X_train, targets_train, optimizer, num_epoch,is_train=True):
    num_epochs = [10, 30, 40,50,60, 80,100, 150, 200, 250, 300,350,400,450,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
    for epoch in range(num_epoch):
        model.train()
        loss_Fn=nn.BCEWithLogitsLoss(reduction='mean')
        for batch_inputs, batch_targets,batch_y in train_loader:
            # weights = torch.tensor(class_weights[batch_y - 1].reshape(-1, 1), dtype=torch.float32).to(device)
            # loss_fn = nn.BCEWithLogitsLoss(reduction='mean', weight=weights)
            loss_fn=loss_Fn
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = loss_fn(outputs, batch_targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train)
                targets_train=targets_train.to(device)
                train_loss = loss_Fn(train_outputs, targets_train)
                print(f"[INFO]\tEpoch {epoch + 1}, Train Loss: {train_loss.item()}")
        if (epoch+1) in num_epochs:
            save_model(model=model, save_path='./ckpt',num_epoch=epoch+1,is_train=is_train)
    # 计算每个类的正确率分别是多少
    train_calculator = Metrics()
    train_predictions = predict_threshold(model, X_train, targets_train)
    train_calculator.calculate_metrics(targets_train.cpu(), train_predictions.cpu())
    train_calculator.transform_format()
    train_calculator.accumulate_counts(targets_train.cpu(), train_predictions.cpu())
    ratio = train_calculator.calculate_each_class_absolute_true_rate()
    print(f"[INFO]\teach class absolute true rate in train set:{ratio}")


def train_for_val(model, train_loader,X_train,targets_train, X_val,
                  targets_val, optimizer, num_epochs,train_calculator,class_weights,fold,is_train=True):
    # 创建一个StepLR调度器，将在epoch数每增加200时将学习率缩小10倍
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    loss_Fn=nn.BCEWithLogitsLoss(reduction='mean')
    for epoch in range(num_epochs):
        model.train()
        for batch_inputs, batch_targets,batch_y in train_loader:
            # 对每个训练批次的不同样本设置不同的损失权重
            # weights=torch.tensor(class_weights[batch_y-1].reshape(-1,1),dtype=torch.float32).to(device)
            # loss_fn = nn.BCEWithLogitsLoss(reduction='mean', weight=weights)
            loss_fn=loss_Fn
            batch_inputs=batch_inputs.to(device)
            batch_targets=batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = loss_fn(outputs, batch_targets)
            loss.backward()
            optimizer.step()
        # scheduler.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # 计算训练误差
                train_outputs=model(X_train)
                targets_train=targets_train.to(device)
                train_loss=loss_Fn(train_outputs, targets_train)
                # 计算验证误差
                val_outputs = model(X_val)
                targets_val=targets_val.to(device)
                val_loss = loss_Fn(val_outputs, targets_val)
                print(f"[INFO]\tEpoch {epoch + 1}\tTrain Loss: {train_loss.item():.4f}\tVal Loss: {val_loss.item():.4f}")

    save_model(model=model, save_path='./ckpt',is_train=is_train,fold=fold,num_epoch=num_epochs)
    train_predictions = predict_threshold(model, X_val, targets_val)
    train_calculator.calculate_metrics(targets_val.cpu(), train_predictions.cpu())
    train_calculator.accumulate_counts(targets_val.cpu(), train_predictions.cpu())


def train_in_k_fold(X_train, Y_train,X_ori,Y_ori, Y_label,lr,weight_decay, num_epochs, class_weights,k=5, batch_size=128):
    # 定义交叉验证
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 0

    train_calculator = Metrics()
    for train_idx, val_idx in kf.split(X_train):
        fold += 1
        print(f"[INFO]\tTraining on fold {fold}")

        # 划分训练集和验证集
        X_train_fold = X_train[train_idx]
        Y_train_fold= Y_train[train_idx]
        Y_label_fold=Y_label[train_idx]
        # 创建数据加载器
        train_dataset = list(zip(X_train_fold, Y_train_fold,Y_label_fold))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 创建模型实例
        model = MultiLabelCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        # 训练模型
        train_for_val(model, train_loader, X_train_fold,Y_train_fold,X_ori, Y_ori, optimizer, num_epochs,
                      train_calculator,class_weights=class_weights,is_train=True,fold=fold)

    train_calculator.transform_format(is_kfold=True)
    ratio = train_calculator.calculate_each_class_absolute_true_rate()
    print(f"[INFO]\t5-fold:each class absolute true rate in train set:{ratio}")


def save_model(model, save_path,num_epoch,is_train=True,fold=None):
    os.makedirs(save_path, exist_ok=True)
    if is_train:
        torch.save(model.state_dict(), os.path.join(save_path, f'{fold}fold_Adam_lr{lr}_weightdecay{weight_decay}_epochs{num_epoch}.pth'))
    else:
        torch.save(model.state_dict(),
                   os.path.join(save_path, f'Adam_lr{lr}_weightdecay{weight_decay}_epochs{num_epoch}.pth'))


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


# 预测时的阈值处理
def predict_threshold(model, inputs, targets, threshold=0.5):
    model.eval()
    inputs = inputs.to(device)
    targets = targets.to(device)
    with torch.no_grad():
        test_outputs = model(inputs)
        # test_loss = loss_fn(test_outputs, targets)
        # print(f"[INFO]\tTest Loss: {test_loss.item()}")
        probs = torch.sigmoid(test_outputs)  # 计算sigmoid概率
        binary_preds = (probs >= threshold).float()  # 根据阈值进行二分

        # # 根据每个标签的索引应用不同的阈值
        # # 初始化二分预测结果
        # binary_preds = torch.zeros_like(probs)
        # # 设置不同标签的阈值
        # thresholds = [0.5, 0.1, 0.1, 0.1]
        # # 根据阈值对每个标签的概率进行二分类
        # for i in range(4):  # 假设有4个标签
        #     binary_preds[:, i] = (probs[:, i] >= thresholds[i]).float()

    return binary_preds


class Metrics:
    '''
    This class used to calculate each class's predicted absolute true rate
    '''

    def __init__(self):
        # 初始化五个指标
        self.Aiming = 0
        self.Coverage = 0
        self.Acc = 0
        self.A_T = 0
        self.A_F = 0
        # 初始化类别计数器
        self.class_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0}
        self.class_correct_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0}
        self.class_absolute_true_rates = {}
        # 创建哈希表，将多标签映射到对应的类别
        self.label_mapping = {
            np.array([1, 0, 0, 0], dtype=np.float64).tobytes(): 1,
            np.array([0, 1, 0, 0], dtype=np.float64).tobytes(): 2,
            np.array([0, 0, 1, 0], dtype=np.float64).tobytes(): 3,
            np.array([0, 0, 0, 1], dtype=np.float64).tobytes(): 4,
            np.array([1, 1, 0, 0], dtype=np.float64).tobytes(): 5,
            np.array([1, 0, 1, 0], dtype=np.float64).tobytes(): 6,
            np.array([1, 0, 0, 1], dtype=np.float64).tobytes(): 7,
            np.array([0, 1, 1, 0], dtype=np.float64).tobytes(): 8,
            np.array([1, 1, 1, 0], dtype=np.float64).tobytes(): 9,
            np.array([1, 1, 0, 1], dtype=np.float64).tobytes(): 10,
            np.array([1, 1, 1, 1], dtype=np.float64).tobytes(): 11
        }

    # 计算评价指标
    @staticmethod
    def accuracy(y_true, y_pred):  # Hamming Score
        temp = 0
        for i in range(y_true.shape[0]):
            temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
        return temp / y_true.shape[0]

    @staticmethod
    def absolute_true(y_true, y_pred):
        count = 0
        for i in range(0, y_pred.shape[0]):
            if (y_pred[i] == y_true[i]).all():
                count += 1
        return count / y_true.shape[0]

    def calculate_metrics(self, y_true, y_pred):
        self.Aiming += precision_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=1)
        self.Coverage += recall_score(y_true=y_true, y_pred=y_pred, average='samples')
        self.Acc += self.accuracy(y_true, y_pred)
        self.A_T += self.absolute_true(y_true, y_pred)
        self.A_F += hamming_loss(y_true, y_pred)

    def transform_format(self,is_kfold=False):
        if is_kfold:
            self.Aiming = "{:.2f}%".format(self.Aiming /5 * 100)
            self.Coverage = "{:.2f}%".format(self.Coverage /5 * 100)
            self.Acc = "{:.2f}%".format(self.Acc /5 * 100)
            self.A_T = "{:.2f}%".format(self.A_T /5 * 100)
            self.A_F = "{:.2f}%".format(self.A_F /5 * 100)
        else:
            self.Aiming = "{:.2f}%".format(self.Aiming * 100)
            self.Coverage = "{:.2f}%".format(self.Coverage * 100)
            self.Acc = "{:.2f}%".format(self.Acc * 100)
            self.A_T = "{:.2f}%".format(self.A_T * 100)
            self.A_F = "{:.2f}%".format(self.A_F * 100)
        # print(
        #     f"[INFO]\tAiming:{self.Aiming},Coverage:{self.Coverage},Accuracy:{self.Acc},Absolute_True:{self.A_T},Absolute_False:{self.A_F}")
        print(
            f"[INFO]\t{self.Aiming},{self.Coverage},{self.Acc},{self.A_T},{self.A_F}")

    def accumulate_counts(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
        for i in range(y_true.shape[0]):
            # 判断当前样本的真实类别
            label_idx = self.label_mapping.get(y_true[i].tobytes())
            # 如果预测和真实标签完全一致，则对应类别计数器加1
            if (y_true[i] == y_pred[i]).all():
                self.class_correct_counts[label_idx] += 1
            # 对对应的类别进行计数
            self.class_counts[label_idx] += 1

    def calculate_each_class_absolute_true_rate(self):
        for label_idx in self.class_counts:
            if self.class_counts[label_idx] != 0:
                self.class_absolute_true_rates[label_idx] = "{:.2f}%".format(
                    (self.class_correct_counts[label_idx] / self.class_counts[label_idx]) * 100)
            else:
                self.class_absolute_true_rates[label_idx] = "{:.2f}%".format(0 * 100)
        return list(self.class_absolute_true_rates.values())

if __name__ == '__main__':
    # 检查是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]\tUsing device: {device}")
    X_train, targets_train, X_test, targets_test, train_loader,X_ori,Y_ori ,Y_label= data_preprocess(sample_strategy=2)
    # 进行5折交叉验证，检查在不同迭代次数下验证集误差的变化，确定是否过拟合
    lr=0.001;num_epoch=100;weight_decay=0.000001
    train_in_k_fold(X_train=X_train, Y_train=targets_train, lr=lr, weight_decay=weight_decay,
                    num_epochs=num_epoch,X_ori=X_ori,Y_ori=Y_ori,Y_label=Y_label,class_weights=class_weight_all(Y_label,1))

    # model=MultiLabelCNN().to(device)
    # num_epoch=500;lr=0.001;weight_decay=0.000001
    # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    # # class_weights=class_weight_all(Y_label,0.1)
    # train_for_test(model=model, train_loader=train_loader, X_train=X_train, targets_train=targets_train,
    #                optimizer=optimizer, num_epoch=num_epoch, is_train=False)
    # torch.onnx.export(model, X_train, "MultiLabelCNN.onnx")

    file_list=os.listdir('./ckpt')
    adam_files = [file_name for file_name in file_list if file_name.startswith('Adam')]
    sorted_adam_files = sorted(adam_files, key=lambda x: int(x.split('_epochs')[1].split('.pth')[0]))
    for file in sorted_adam_files:
        if file.startswith('Adam'):
            model_path=os.path.join('./ckpt',file)
            model = load_model(model_path=model_path).to(device)

            # 预测阶段
            # 获取最终的二元输出
            predictions = predict_threshold(model, X_test, targets_test)
            # 计算评估指标Aiming等，以及每个类的正确率分别是多少
            test_calculator = Metrics()
            test_calculator.calculate_metrics(targets_test.cpu(),predictions.cpu())
            test_calculator.transform_format()
            test_calculator.accumulate_counts(targets_test.cpu(), predictions.cpu())
            ratio = test_calculator.calculate_each_class_absolute_true_rate()
            print(f"[INFO]{file}\teach class absolute true rate in test set:{ratio}")