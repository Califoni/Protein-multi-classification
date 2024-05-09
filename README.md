# Protein-multi-classification  
**本研究结合ClusterCentroids下采样算法和CNN构造了一个多标签预测模型，对蛋白质翻译后修饰中的多重赖氨酸修饰位点进行预测**
### 1.文件结构与功能说明
* **Data**  
    Train dataset：原始训练数据  
    Test dataset：原始测试数据  
    PSTAAP_train.m：对于训练集进行特征提取的脚本->PSTAAP_train.mat  
    PSTAAP_test.m：对于测试集进行特征提取的脚本->PSTAAP_test.mat
* **DataProcess.py**：实现样本数据的多标签构造和训练数据下采样  
* **PSTAAP.py**：实现模型构建，进行模型训练，实现5折交叉验证与性能测试
* **predictor.py**：实现用户友好界面，通过选择模型与数据集进行预测演示
### 2.使用示例
#### 2.1 使用本实验数据进行预测
* 运行predictor.py，预测数据选取`Data/PSTAAP_test.mat`，模型选取`model/`下的pth文件，直接predict即可
#### 2.2 自定义数据进行预测
* 首先将数据转化成`Data/Test dataset`中数据的格式，然后使用matlab修改`Data/PSTAAP_test.m`脚本中的文件路径为你自己的文件路径后运行，获得特征提取后的mat文件
* 运行predictor.py，其余步骤同2.1，数据选择你自己的