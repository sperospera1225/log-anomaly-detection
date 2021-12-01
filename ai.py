import pandas as pd
import json
import numpy as np

test_file = pd.read_csv('https://docs.google.com/uc?export=download&id=1GDq0dqwYhcsOeqcCen06YwQRi43ZuVoX')
need_col = ['request', 'response', 'ip', 'useragent', 'timestamp']

df = test_file[need_col]

df = df.dropna(axis=0)

df['request'] = df['request'].apply(lambda x: str(x).split('- -')[0].strip())

df['timestamp'] = pd.to_datetime(df['timestamp'])

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()  # string을 정수로 변환할 인코더

df['request'] = LE.fit_transform(df['request'])  # sql_syntax 정수 인코딩

df = df.sort_values(by=['timestamp'])
df['label'] = df['useragent'].apply(lambda x: 1 if 'bot' in str(x) else (1 if 'crawl' in str(x) else (
    1 if 'BUbiNG' in str(x) else (1 if 'Bot' in str(x) else (1 if 'Crawl' in str(x) else 0)))))

num_classes = len(df['request'].unique())

df.reset_index(drop=True, inplace=True)
log_entry = "ip"
unique_key = df[log_entry].unique()
unique_n = len(df[log_entry].unique())
## log_entry값에 따라서 이벤트의 sequence와 정답 라벨링 결과를 튜플형식으로 정렬한다
param_bot = dict()
param_man = dict()

log_key_bot = dict()
log_key_man = dict()

for i in range(unique_n):
    param_bot[unique_key[i]] = []

for i in range(unique_n):
    param_man[unique_key[i]] = []

for i in range(unique_n):
    log_key_bot[unique_key[i]] = []

for i in range(unique_n):
    log_key_man[unique_key[i]] = []

for idx in range(len(df)):
    if df.iloc[idx]['label'] == 1:
        param_bot[df[log_entry][idx]].append(
            (df.iloc[idx]['timestamp'] - df.iloc[idx - 1]['timestamp'], df.iloc[idx]["response"]))

for idx in range(len(df)):
    if df.iloc[idx]['label'] == 0:
        param_man[df[log_entry][idx]].append(
            (df.iloc[idx]['timestamp'] - df.iloc[idx - 1]['timestamp'], df.iloc[idx]["response"]))

for idx in range(len(df)):
    if df.iloc[idx]['label'] == 1:
        log_key_bot[df[log_entry][idx]].append((df.iloc[idx]['request'], df.iloc[idx]["label"]))

for idx in range(len(df)):
    if df.iloc[idx]['label'] == 0:
        log_key_man[df[log_entry][idx]].append((df.iloc[idx]['request'], df.iloc[idx]["label"]))

# 이벤트의 sequence만을 담는 배열을 만든다
seq_param_bot = []
seq_param_man = []
seq_log_key_bot = []
seq_log_key_man = []

for k in param_bot.keys():
    seq_param_bot.append(param_bot[k])

for k in param_man.keys():
    seq_param_man.append(param_man[k])

for k in log_key_bot.keys():
    seq_log_key_bot.append(log_key_bot[k])

for k in log_key_man.keys():
    seq_log_key_man.append(log_key_man[k])

seq_param_bot.sort(key=len, reverse=True)
seq_param_man.sort(key=len, reverse=True)
seq_log_key_bot.sort(key=len, reverse=True)
seq_log_key_man.sort(key=len, reverse=True)

# 빈 배열 삭제
idx = 0
for item in range(len(seq_param_man)):
    if len(seq_param_man[item]) <= 5:  # time window의 값으로 설정해야하진 않을까?
        idx = item
        break

seq_param_man = seq_param_man[:idx]
seq_param_man

# 빈 배열 삭제
idx = 0
for item in range(len(seq_log_key_bot)):
    if len(seq_log_key_bot[item]) <= 5:
        idx = item
        break

seq_log_key_bot = seq_log_key_bot[:idx]
seq_log_key_bot

# 빈 배열 삭제
idx = 0
for item in range(len(seq_log_key_man)):
    if len(seq_log_key_man[item]) <= 5:
        idx = item
        break

seq_log_key_man = seq_log_key_man[:idx]
seq_log_key_man

nums = len(seq_log_key_man)
ratio = 0.8
train_num = int(nums * ratio)

key_train = seq_log_key_man[:train_num]
key_valid = seq_log_key_man[train_num:]

param_train = seq_param_man[:train_num]
param_valid = seq_param_man[train_num:]

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import tensorflow as tf
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def generate(name, window_size):
    num_sessions = 0
    inputs = []
    outputs = []

    for line in name:
        num_sessions += 1
        #         line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
        for i in range(len(line) - window_size):
            inputs_tmp = []
            for j in line[i:i + window_size]:
                inputs_tmp.append(j[0])
            inputs.append(inputs_tmp)
            outputs.append(line[i + window_size][0])

    #     print('Number of sessions({}): {}'.format(name, num_sessions))
    #     print('Number of seqs({}): {}'.format(name, len(inputs)))
    return inputs, outputs


start_time = time.time()
num_epochs = 100
batch_size = 32
window_size = 10
num_classes = len(df['request'].unique())

TP = 0
FP = 0
n_candidates = 9  # top n probability of the next tag

X, Y = generate(key_train, window_size)
X = np.reshape(X, (len(X), window_size, 1))
X = X / float(num_classes)
Y = to_categorical(Y, num_classes)
print(X)
print(Y)

from tensorflow.keras.optimizers import Adam

optimizer = Adam(lr=0.01)

model = Sequential()
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

checkpoint_filepath = '/ai'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(X, Y, epochs=100, batch_size=batch_size, shuffle=True)


# window_size에 맞추어서 input과 label의 라벨값을 리턴하는 함수
def generate_pred(file, window_size):
    hdfs = list()
    hhhh = list()

    uri = []
    trid = []

    for line in file:
        uri_tmp = []
        trid_tmp = []
        for i in line:
            uri_tmp.append(i[0])
            trid_tmp.append(i[1])
        uri.append(uri_tmp)
        trid.append(trid_tmp)
    # 길이가 windowsize보다 짧을 경우 패딩을 시킴
    for ln in uri:
        line = list(map(lambda n: n - 1, ln))
        ln = line + [-2] * (window_size + 1 - len(line))
        hdfs.append(tuple(ln))

    for ll in trid:
        line = list(ll)
        ll = line + [-2] * (window_size + 1 - len(line))
        hhhh.append(tuple(ll))

    return hdfs, hhhh

# 다음에 올거라고 예측하는 event 라벨이 모델의 candidates 안에 존재하면 0, 존재하지않으면 1을 가지는 배열 생성
# 추후 룰 기반 결과와 비교해서 성능평가
def get_list(labeled, input_pred, total):
    pred_result = np.zeros((total, 1))
    idx = 0
    for i, j in zip(labeled, input_pred):
        if i in j:
            pred_result[idx] = 0 #정상
        else:
            pred_result[idx] = 1 #비정상
        idx += 1
    return pred_result


from tqdm import tqdm
from keras.activations import softmax

test_normal_loader, was_trid = generate_pred(key_train, window_size)
# test_abnormal_loader = generate1('hdfs_test_abnormal', window_size)

# for line in test_abnormal_loader:
#     for i in range(len(line) - window_size):
#         X = np.reshape(seq, (1, window_size, 1))
#         X = X / float(num_classes)
#         Y = to_categorical(label, num_classes)
#         prediction = model.predict(X, verbose=0)
#         if np.argmax(Y) not in prediction.argsort()[0][::-1][: n_candidates]:

#             TP += 1
#             break
total = 0
correct = 0
proba = []
trid_list = []
inputed = []
labeled = []
input_pred = []
start_time = time.time()

for line, y in tqdm(zip(test_normal_loader, was_trid)):
    for i in range(len(line) - window_size):
        seq = line[i:i + window_size]
        label = line[i + window_size]
        trid = y[i + window_size]
        if label == -2:
            continue

        X = np.reshape(seq, (1, window_size, 1))
        X = X / float(num_classes)
        Y = to_categorical(label, num_classes)
        prediction = model.predict(X, verbose=0)

        predicted = prediction.argsort()[0][::-1][: n_candidates]
        print(predicted)
        y_pred = prediction

        proba.append(y_pred)

        inputed.append(seq)
        labeled.append(label)
        trid_list.append(trid)
        input_pred.append(predicted)
        total += 1

        if np.argmax(Y) not in prediction.argsort()[0][::-1][: n_candidates]:
            FP += 1
            break

elapsed_time = time.time() - start_time
print('elapsed_time: {:.3f}s'.format(elapsed_time))



print("elapsed_time: {:.3f}s".format(elapsed_time))
print("correct_number : %d" % correct)
print("total : %d" % total)
accu = (total - correct) / total
# 전체 개수중에서 제대로 예측한 개수
print("accuracy : %f" % accu)
# Compute precision, recall and F1-measure
pred_list = get_list(labeled, input_pred, total)
trid_list = np.array(trid_list)


import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import pandas as pd
import numpy as np
import sys
#오차행렬, 정확도, 정밀도, 재현율 구하는 함수
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}'.format(accuracy, precision, recall))
    print('f1 스코어: {0: .4f}'.format(f1))

    get_clf_eval(trid_list, pred_list)

    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    TN = len(test_normal_loader) - FP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)

    print(f"FP:{FP}")
    print(f"FN: {FN}")
    print(f"TP: {TP}")
    print(f"TN: {TN}")

    print(
        'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
            FP, FN, P, R, F1))
    print('Finished Predicting')
