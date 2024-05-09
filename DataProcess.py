import scipy
import numpy as np
from collections import Counter
import torch
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler
from imblearn.over_sampling import  SMOTE, BorderlineSMOTE,RandomOverSampler
from sklearn.decomposition import PCA


def load_HraaC(train_path,test_path):
    data = scipy.io.loadmat(train_path)
    data_test = scipy.io.loadmat(test_path)
    d_1 = np.array(data['PPT1'])
    d_2 = np.array(data[f'PPT2'])
    d_3 = np.array(data[f'PPT3'])
    d_4 = np.array(data[f'PPT4'])
    d_5 = np.array(data[f'PPT5'])
    d_6 = np.array(data[f'PPT6'])
    d_7 = np.array(data[f'PPT7'])
    d_8 = np.array(data[f'PPT8'])
    d_9 = np.array(data[f'PPT9'])
    d_10 = np.array(data[f'PPT10'])
    d_11 = np.array(data[f'PPT11'])

    dt_1 = np.array(data_test[f'PPT1'])
    dt_2 = np.array(data_test[f'PPT2'])
    dt_3 = np.array(data_test[f'PPT3'])
    dt_4 = np.array(data_test[f'PPT4'])
    dt_5 = np.array(data_test[f'PPT5'])
    dt_6 = np.array(data_test[f'PPT6'])
    dt_7 = np.array(data_test[f'PPT7'])
    dt_8 = np.array(data_test[f'PPT8'])
    dt_9 = np.array(data_test[f'PPT9'])
    dt_10 = np.array(data_test[f'PPT10'])
    dt_11 = np.array(data_test[f'PPT11'])

    X_train = np.concatenate((d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_11), axis=0)
    X_test = np.concatenate((dt_1, dt_2, dt_3, dt_4, dt_5, dt_6, dt_7, dt_8, dt_9, dt_10, dt_11), axis=0)

    y_dir = []
    yt_dir = []

    for i in range(1, 12):
        exec(f'y_dir.append(np.full(d_{i}.shape[0],i))')
        exec(f'yt_dir.append(np.full(dt_{i}.shape[0],i))')
    Y_train = np.concatenate((y_dir), axis=0)
    Y_test = np.concatenate((yt_dir), axis=0)
    print(
        f"[INFO]\tData Preview: X_train:{X_train.shape},X_test:{X_test.shape},Y_train:{Y_train.shape},Y_test:{Y_test.shape}")
    print("[INFO]\tData Load Finished")
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), Y_train, Y_test


def load_Kmer2(train_path, test_path):
    data = scipy.io.loadmat(train_path)
    data_test = scipy.io.loadmat(test_path)
    d_1 = data['Kmer1_2']
    dt_1 = data_test['Kmer1_2']
    d_2 = data['Kmer2_2']
    dt_2 = data_test['Kmer2_2']
    d_3 = data['Kmer3_2']
    dt_3 = data_test['Kmer3_2']
    d_4 = data['Kmer4_2']
    dt_4 = data_test['Kmer4_2']
    d_5 = data['Kmer5_2']
    dt_5 = data_test['Kmer5_2']
    d_6 = data['Kmer6_2']
    dt_6 = data_test['Kmer6_2']
    d_7 = data['Kmer7_2']
    dt_7 = data_test['Kmer7_2']
    d_8 = data['Kmer8_2']
    dt_8 = data_test['Kmer8_2']
    d_9 = data['Kmer9_2']
    dt_9 = data_test['Kmer9_2']
    d_10 = data['Kmer10_2']
    dt_10 = data_test['Kmer10_2']
    d_11 = data['Kmer11_2']
    dt_11 = data_test['Kmer11_2']
    X_train = np.concatenate((d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_11), axis=0)
    X_test = np.concatenate((dt_1, dt_2, dt_3, dt_4, dt_5, dt_6, dt_7, dt_8, dt_9, dt_10, dt_11), axis=0)
    y_dir = []
    yt_dir = []

    for i in range(1, 12):
        exec(f'y_dir.append(np.full(d_{i}.shape[0],i))')
        exec(f'yt_dir.append(np.full(dt_{i}.shape[0],i))')
    Y_train = np.concatenate((y_dir), axis=0)
    Y_test = np.concatenate((yt_dir), axis=0)
    print(
        f"[INFO]\tData Preview: X_train:{X_train.shape},X_test:{X_test.shape},Y_train:{Y_train.shape},Y_test:{Y_test.shape}")
    print("[INFO]\tData Load Finished")
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), Y_train, Y_test


def load_Kmer1(train_path, test_path):
    data = scipy.io.loadmat(train_path)
    data_test = scipy.io.loadmat(test_path)
    d_1 = data['Kmer1_1']
    dt_1 = data_test['Kmer1_1']
    d_2 = data['Kmer2_1']
    dt_2 = data_test['Kmer2_1']
    d_3 = data['Kmer3_1']
    dt_3 = data_test['Kmer3_1']
    d_4 = data['Kmer4_1']
    dt_4 = data_test['Kmer4_1']
    d_5 = data['Kmer5_1']
    dt_5 = data_test['Kmer5_1']
    d_6 = data['Kmer6_1']
    dt_6 = data_test['Kmer6_1']
    d_7 = data['Kmer7_1']
    dt_7 = data_test['Kmer7_1']
    d_8 = data['Kmer8_1']
    dt_8 = data_test['Kmer8_1']
    d_9 = data['Kmer9_1']
    dt_9 = data_test['Kmer9_1']
    d_10 = data['Kmer10_1']
    dt_10 = data_test['Kmer10_1']
    d_11 = data['Kmer11_1']
    dt_11 = data_test['Kmer11_1']
    X_train = np.concatenate((d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_11), axis=0)
    X_test = np.concatenate((dt_1, dt_2, dt_3, dt_4, dt_5, dt_6, dt_7, dt_8, dt_9, dt_10, dt_11), axis=0)
    y_dir = []
    yt_dir = []

    for i in range(1, 12):
        exec(f'y_dir.append(np.full(d_{i}.shape[0],i))')
        exec(f'yt_dir.append(np.full(dt_{i}.shape[0],i))')
    Y_train = np.concatenate((y_dir), axis=0)
    Y_test = np.concatenate((yt_dir), axis=0)
    print(
        f"[INFO]\tData Preview: X_train:{X_train.shape},X_test:{X_test.shape},Y_train:{Y_train.shape},Y_test:{Y_test.shape}")
    print("[INFO]\tData Load Finished")
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), Y_train, Y_test


def load_DR2(train_path, test_path):
    data = scipy.io.loadmat(train_path)
    data_test = scipy.io.loadmat(test_path)
    d_1 = np.array(data['DR1_2'])
    d_2 = np.array(data['DR2_2'])
    d_3 = np.array(data['DR3_2'])
    d_4 = np.array(data['DR4_2'])
    d_5 = np.array(data['DR5_2'])
    d_6 = np.array(data['DR6_2'])
    d_7 = np.array(data['DR7_2'])
    d_8 = np.array(data['DR8_2'])
    d_9 = np.array(data['DR9_2'])
    d_10 = np.array(data['DR10_2'])
    d_11 = np.array(data['DR11_2'])

    dt_1 = np.array(data_test['DR1_2'])
    dt_2 = np.array(data_test['DR2_2'])
    dt_3 = np.array(data_test['DR3_2'])
    dt_4 = np.array(data_test['DR4_2'])
    dt_5 = np.array(data_test['DR5_2'])
    dt_6 = np.array(data_test['DR6_2'])
    dt_7 = np.array(data_test['DR7_2'])
    dt_8 = np.array(data_test['DR8_2'])
    dt_9 = np.array(data_test['DR9_2'])
    dt_10 = np.array(data_test['DR10_2'])
    dt_11 = np.array(data_test['DR11_2'])

    X_train = np.concatenate((d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_11), axis=0)
    X_test = np.concatenate((dt_1, dt_2, dt_3, dt_4, dt_5, dt_6, dt_7, dt_8, dt_9, dt_10, dt_11), axis=0)
    y_dir = []
    yt_dir = []

    for i in range(1, 12):
        exec(f'y_dir.append(np.full(d_{i}.shape[0],i))')
        exec(f'yt_dir.append(np.full(dt_{i}.shape[0],i))')
    Y_train = np.concatenate((y_dir), axis=0)
    Y_test = np.concatenate((yt_dir), axis=0)
    print(
        f"[INFO]\tData Preview: X_train:{X_train.shape},X_test:{X_test.shape},Y_train:{Y_train.shape},Y_test:{Y_test.shape}")
    print("[INFO]\tData Load Finished")
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), Y_train, Y_test


def load_DR1(train_path, test_path):
    data = scipy.io.loadmat(train_path)
    data_test = scipy.io.loadmat(test_path)
    d_1 = np.array(data['DR1_1'])
    d_2 = np.array(data['DR2_1'])
    d_3 = np.array(data['DR3_1'])
    d_4 = np.array(data['DR4_1'])
    d_5 = np.array(data['DR5_1'])
    d_6 = np.array(data['DR6_1'])
    d_7 = np.array(data['DR7_1'])
    d_8 = np.array(data['DR8_1'])
    d_9 = np.array(data['DR9_1'])
    d_10 = np.array(data['DR10_1'])
    d_11 = np.array(data['DR11_1'])

    dt_1 = np.array(data_test['DR1_1'])
    dt_2 = np.array(data_test['DR2_1'])
    dt_3 = np.array(data_test['DR3_1'])
    dt_4 = np.array(data_test['DR4_1'])
    dt_5 = np.array(data_test['DR5_1'])
    dt_6 = np.array(data_test['DR6_1'])
    dt_7 = np.array(data_test['DR7_1'])
    dt_8 = np.array(data_test['DR8_1'])
    dt_9 = np.array(data_test['DR9_1'])
    dt_10 = np.array(data_test['DR10_1'])
    dt_11 = np.array(data_test['DR11_1'])
    X_train = np.concatenate((d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_11), axis=0)
    X_test = np.concatenate((dt_1, dt_2, dt_3, dt_4, dt_5, dt_6, dt_7, dt_8, dt_9, dt_10, dt_11), axis=0)
    y_dir = []
    yt_dir = []

    for i in range(1, 12):
        exec(f'y_dir.append(np.full(d_{i}.shape[0],i))')
        exec(f'yt_dir.append(np.full(dt_{i}.shape[0],i))')
    Y_train = np.concatenate((y_dir), axis=0)
    Y_test = np.concatenate((yt_dir), axis=0)
    print(
        f"[INFO]\tData Preview: X_train:{X_train.shape},X_test:{X_test.shape},Y_train:{Y_train.shape},Y_test:{Y_test.shape}")
    print("[INFO]\tData Load Finished")
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), Y_train, Y_test


def load_data(train_path, test_path):
    print("[Data Loading]")
    data = scipy.io.loadmat(train_path)
    data_test = scipy.io.loadmat(test_path)
    name=train_path.split('/')[1].split('.')[0]
    d_1 = np.array(data[f'{name}1'])
    d_2 = np.array(data[f'{name}2'])
    d_3 = np.array(data[f'{name}3'])
    d_4 = np.array(data[f'{name}4'])
    d_5 = np.array(data[f'{name}5'])
    d_6 = np.array(data[f'{name}6'])
    d_7 = np.array(data[f'{name}7'])
    d_8 = np.array(data[f'{name}8'])
    d_9 = np.array(data[f'{name}9'])
    d_10 = np.array(data[f'{name}10'])
    d_11 = np.array(data[f'{name}11'])

    dt_1 = np.array(data_test[f'{name}1'])
    dt_2 = np.array(data_test[f'{name}2'])
    dt_3 = np.array(data_test[f'{name}3'])
    dt_4 = np.array(data_test[f'{name}4'])
    dt_5 = np.array(data_test[f'{name}5'])
    dt_6 = np.array(data_test[f'{name}6'])
    dt_7 = np.array(data_test[f'{name}7'])
    dt_8 = np.array(data_test[f'{name}8'])
    dt_9 = np.array(data_test[f'{name}9'])
    dt_10 = np.array(data_test[f'{name}10'])
    dt_11 = np.array(data_test[f'{name}11'])
    X_train = np.concatenate((d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_11), axis=0)
    X_test = np.concatenate((dt_1, dt_2, dt_3, dt_4, dt_5, dt_6, dt_7, dt_8, dt_9, dt_10, dt_11), axis=0)

    y_dir = []
    yt_dir = []

    for i in range(1, 12):
        exec(f'y_dir.append(np.full(d_{i}.shape[0],i))')
        exec(f'yt_dir.append(np.full(dt_{i}.shape[0],i))')

    Y_train = np.concatenate((y_dir), axis=0)
    Y_test = np.concatenate((yt_dir), axis=0)
    print(
        f"[INFO]\tData Preview: X_train:{X_train.shape},X_test:{X_test.shape},Y_train:{Y_train.shape},Y_test:{Y_test.shape}")
    print("[INFO]\tData Load Finished")
    return torch.tensor(X_train,dtype=torch.float32), torch.tensor(X_test,dtype=torch.float32), Y_train, Y_test


def ratio_multiplier(y):
    # set under resample ratio
    multiplier = {1: 0.1}
    target_stats = Counter(y)
    for key, value in target_stats.items():
        if key in multiplier:
            target_stats[key] = int(value * multiplier[key])
    return target_stats


def data_resample(X_train, y, sample_strategy=0):
    """
    We get four resample strategies:
        0.nothing happen
        1.under resample
        2.up resample
        3.under resample + up resample
    """
    if sample_strategy == 1:
        under = ClusterCentroids(
            sampling_strategy=ratio_multiplier, random_state=1,
            estimator=MiniBatchKMeans(n_init=1, random_state=1)
        )
        # under=RandomUnderSampler(sampling_strategy=ratio_multiplier,random_state=2)
        x_resampled, y_resampled = under.fit_resample(X_train, y)

    elif sample_strategy == 2:
        sampling_strategy = Counter(
            {1: 9279, 2: 3099, 3: 3099, 4: 3099, 5: 3099, 6: 3099, 7: 3099, 8: 3099, 9: 3099, 10: 3099, 11: 3099})
        # over1 = BorderlineSMOTE(random_state=1, sampling_strategy=sampling_strategy)
        over2 = SMOTE(random_state=1, sampling_strategy=sampling_strategy)
        # x_resampled, y_resampled = over1.fit_resample(X_train, y)
        x_resampled, y_resampled = over2.fit_resample(X_train, y)
    elif sample_strategy > 2:
        print("resample")
        under = ClusterCentroids(
            sampling_strategy=ratio_multiplier, random_state=1,
            estimator=MiniBatchKMeans(n_init=1, random_state=1)
        )
        x_resampled, y_resampled = under.fit_resample(X_train, y)
        over1 = BorderlineSMOTE(random_state=1)
        over2 = SMOTE(random_state=1)
        x_resampled, y_resampled = over1.fit_resample(x_resampled, y_resampled)
        x_resampled, y_resampled = over2.fit_resample(x_resampled, y_resampled)
    else:
        return X_train, y
    return torch.tensor(x_resampled,dtype=torch.float32), y_resampled


def data_reshape(X,shape):
    pca = PCA(n_components=shape)
    data_2d_pca = pca.fit_transform(X)
    return torch.tensor(data_2d_pca,dtype=torch.float32)


def make_ylabel(y):
    # transform y into multi-label
    Y = np.zeros((y.shape[0], 4))
    for i in range(y.shape[0]):
        if y[i] == 1:
            Y[i] = np.array([1, 0, 0, 0])  # a
        elif y[i] == 2:
            Y[i] = np.array([0, 1, 0, 0])  # c
        elif y[i] == 3:
            Y[i] = np.array([0, 0, 1, 0])  # m
        elif y[i] == 4:
            Y[i] = np.array([0, 0, 0, 1])  # s
        elif y[i] == 5:
            Y[i] = np.array([1, 1, 0, 0])  # ac
        elif y[i] == 6:
            Y[i] = np.array([1, 0, 1, 0])  # am
        elif y[i] == 7:
            Y[i] = np.array([1, 0, 0, 1])  # as
        elif y[i] == 8:
            Y[i] = np.array([0, 1, 1, 0])  # cm
        elif y[i] == 9:
            Y[i] = np.array([1, 1, 1, 0])  # acm
        elif y[i] == 10:
            Y[i] = np.array([1, 1, 0, 1])  # acs
        elif y[i] == 11:
            Y[i] = np.array([1, 1, 1, 1])  # acms
    print("[INFO]\tmulti-label Y:", Y.shape)
    return torch.tensor(Y,dtype=torch.float32)


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_data("mydata/PSTAAP_train.mat", "mydata/PSTAAP_test.mat")
    # print("under resample ratio:", ratio_multiplier((Y_train)))
    # make_ylabel(Y_train)
    # load_DR2('mydata/DR2.mat','mydata/DR2_test.mat')
    # load_Kmer2('mydata/Kmer2.mat','mydata/Kmer2_test.mat')
    # load_HraaC('./mydata/HraaC.mat','./mydata/HraaC_test.mat')