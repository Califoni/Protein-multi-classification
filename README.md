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
### 2.使用示例
