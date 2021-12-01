from __future__ import print_function
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Lambda
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.utils.np_utils import to_categorical
#from keras.utils import np_utils
#from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

import numpy
import h5py
from tensorflow.keras import callbacks
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import plot_confusion_matrix, precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error
from sklearn import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
from numpy import nan
import datetime
import sys
import os
import matplotlib.pyplot as plt
import json
import sqlalchemy
from elasticsearch import Elasticsearch
from sklearn.preprocessing import LabelEncoder

with open('./setting.json','r', encoding='utf-8') as json_file:
    setting_data = json.load(json_file)

db_host, db_user, db_port, db_pwd, db_database = setting_data['db_host'], setting_data['db_user'], setting_data['db_port'], setting_data['db_pwd'], setting_data['db_database']
engine = sqlalchemy.create_engine("mysql+pymysql://"+db_user+":"+db_pwd+"@"+db_host+":"+str(db_port)+"/"+db_database)

setting_data = pd.read_sql_table(table_name="N_CODE_INFO", con=engine, columns=['CD1','CD2','DATAJSON'])
setting_es_data = setting_data[(setting_data['CD1'] == '0007') & (setting_data['CD2'] == 'setting')]
setting_es_data = eval(list(setting_es_data['DATAJSON'])[0])



input_data_ip = setting_es_data['input_data_ip']
input_data_port = setting_es_data['input_data_port']
acc_data_index = setting_es_data['input_data_index']
sql_data_index = setting_es_data['sql_data_index']

setting_data = setting_data[(setting_data['CD1'] == '0007') & (setting_data['CD2'] == 'cnnsetting')]
setting_data = eval(list(setting_data['DATAJSON'])[0])
rule = setting_data['rule']
feature_selection = setting_data['feature_selection']
K = setting_data['selectKBest']
start_date = setting_data['train_start_date']
finish_date = setting_data['train_finish_date']
test_start_date = setting_data['model_test_start_date']
test_finish_date = setting_data['model_test_finish_date']
epochs = setting_data['epochs']
learning_rate = setting_data['learning_rate']
model_name = setting_data['model_name']

from common import Common
cm = Common()
print("acc log 로딩중.... ")
data = cm.es_data_search_date(input_data_ip, input_data_port, acc_data_index, 'str_time', start_date, test_finish_date)
data = pd.DataFrame(data)
data = data[["str_time", "user_sn","dept_code", "position_code", "was_trid", "uri",  "work_time"]]
data = data.sort_values(by="str_time")
print("acc log 완료 ")


#결측행 삭제
print("acc log 결측치 삭제 ")
data = data[(data['position_code']!='UNKNOWN') & (data['dept_code']!='UNKNOWN') ]
data[['position_code', 'dept_code']] = data[['position_code', 'dept_code']].apply(pd.to_numeric)
data = data.dropna(subset=['str_time'])

from common import Common

if len(data) == 0:
    print(acc_data_index + " 는 비어있습니다. 변수 설정을 확인해주세요.")
    quit()

if rule == "RS02" or rule == "RS03":
    print("sql_data 로딩중.... ")
    sql_data = cm.es_data_search_date(input_data_ip, input_data_port, sql_data_index, 'exec_str_time', start_date,
                                      test_finish_date)
    sql_data = pd.DataFrame(sql_data)
    sql_data = sql_data.sort_values(by="exec_str_time")
    sql_data = sql_data[["was_trid", "rst_row_cnt", "sql_syntax"]]

    print("sql_data 완료 ")

print("acc_log와 sql_data 병합중.... ")
LE = LabelEncoder()
sql_data['rst_row_cnt']=sql_data['rst_row_cnt'].apply(pd.to_numeric)
sql_data['sql_syntax'] = LE.fit_transform(sql_data['sql_syntax']) # sql_syntax 정수 인코딩
sql_data = sql_data.groupby('was_trid').std().fillna(0).reset_index() # 표준편차로 groupping
data = pd.merge(data,sql_data,how='inner',on=['was_trid'])
print("acc_log와 sql_data 병합 완료")

if len(sql_data)==0:
    print(sql_data_index+" 는 비어있습니다. 변수 설정을 확인해주세요.")
    quit()

#mariaDB에서 라벨링 데이터 받아오기
#
#
#
print("mariaDB에서 라벨 데이터 불러오기..... ")
label = pd.read_sql_table(table_name="N_VIOLATE_ML", con=engine,columns=['was_trid','rule_total'])
label['rule_total']=label['rule_total'].apply(lambda x:eval(x))
label = label.join(pd.DataFrame(label.pop('rule_total').values.tolist()))
if len(label)==0:
    print("N_VIOLATE_ML이 비어있습니다.")
    quit()
print("mariaDB에서 라벨 데이터 불러오기 완료")

data = pd.merge(data,label[['was_trid',"RT01","RT02","RU01","RU02","RS01","RS02"]],how='inner',on='was_trid')
data.sort_values(by=['str_time'])

change_value_dict = {'정상':1, '비정상':0, '점검대상':0}
df = data.replace({rule:change_value_dict})

ratio = 0.8
train_num = int(len(df)*ratio)

train_df = df[:train_num]
test_df = df[train_num:]

train_df['RT01'] = train_df['RT01'].apply(lambda x: 1 if x=="정상" else 0)
test_df['RT01'] = test_df['RT01'].apply(lambda x: 1 if x=="정상" else 0)

train_df = train_df[train_df['RT01']==1]
train_df['str_time']=pd.to_datetime(train_df['str_time'])
train_df.sort_values(by=['str_time'], inplace = True)

train_df = train_df.astype({'dept_code':'string','position_code':'string'})
test_df = test_df.astype({'dept_code':'string','position_code':'string'})

train_df['log_key'] = train_df['user_sn']+train_df['dept_code']+train_df['position_code']
test_df['log_key'] = test_df['user_sn']+test_df['dept_code']+test_df['position_code']

# log_key 컬럼을 정수 인코딩 하여 모델에 사용함
key_unique = np.append(train_df['log_key'].unique(), test_df['log_key'].unique())
key_unique = np.unique(key_unique)
logk_n = len(key_unique)

key_dict = dict()
for i in range(len(key_unique)):
    key_dict[key_unique[i]]=i

num_keys = len(key_dict)
key_id = []
for i in range(len(train_df)):
    key_id.append(key_dict[train_df.iloc[i]['log_key']])
train_df.reset_index(drop=True, inplace=True)
train_df['log_key_id']=pd.DataFrame(key_id)

train_df['str_time']=pd.to_datetime(train_df['str_time'])
train_df['str_time'] = train_df.str_time.dt.floor('30min')
train_df.sort_values(by=['str_time'], inplace = True)
train_df['str_time'] = train_df['str_time'].dt.hour*100 + train_df['str_time'].dt.minute
time_unique = train_df['str_time'].unique()

print(len(train_df['str_time'].value_counts()),"가 48이여야함")
time_list = [0, 30, 100, 130, 200, 230, 300, 330, 400, 430, 500, 530, 600, 630, 700, 730, 800, 830, 900, 930, 1000, 1030, 1100, 1130, 1200, 1230, 1300, 1330, 1400, 1430, 1500, 1530, 1600, 1630, 1700, 1730, 1800, 1830, 1900, 1930, 2000, 2030, 2100, 2130, 2200, 2230, 2300, 2330]
add_time = set(time_list) - set(time_unique.tolist())
add_time = list(add_time)
time_unique = np.append(time_unique, np.array(add_time))
uri_n = len(time_unique)

time_dict = dict()
for i in range(len(time_unique)):
    time_dict[time_unique[i]]=i
# event_id의 배열은 각각의 로그가 log_entry로 정렬된 sorted_df_acc_log_**파일에 대해서 전체 모든 행에 대해서 라벨링 한값을 나타내고
# 해당 배열은 event_id라는 컬럼으로 df에다가 추가된다.
event_id = []
for i in range(len(train_df)):
    event_id.append(time_dict[train_df.iloc[i]['str_time']])
#각각의 uri값에 따라서 라벨링된 event_id의 컬럼이 기존 df 데이터 프레임에 추가됨
train_df.reset_index(drop=True, inplace=True)
train_df['param_id']=pd.DataFrame(event_id)

unique_key = train_df["log_key_id"].unique()
unique_n = len(unique_key)
## log_entry값에 따라서 이벤트의 sequence와 정답 라벨링 결과를 튜플형식으로 정렬한다
param_dict=dict()
for i in range(unique_n):
    param_dict[unique_key[i]]=[]

for idx in range(len(train_df)):
    param_dict[train_df["log_key_id"][idx]].append((train_df.iloc[idx]['param_id'], train_df.iloc[idx]["RT01"]))
# 이벤트의 sequence만을 담는 배열을 만든다
uri_seq = []
for k in param_dict.keys():
    uri_seq.append(param_dict[k])
uri_seq.sort(key = len, reverse = True)

key_seq = []
for idx in range(len(train_df)):
    key_seq.append((train_df.iloc[idx]['log_key_id'], train_df.iloc[idx]['RT01']))

test_df['str_time']=pd.to_datetime(test_df['str_time'])
test_df.sort_values(by=['str_time'], inplace = True)

key_id = []
for i in range(len(test_df)):
    key_id.append(key_dict[test_df.iloc[i]['log_key']])
test_df.reset_index(drop=True, inplace=True)
test_df['log_key_id']=pd.DataFrame(key_id)

test_df['str_time']=pd.to_datetime(test_df['str_time'])
test_df['str_time'] = test_df.str_time.dt.floor('30min')
test_df.sort_values(by=['str_time'], inplace = True)
test_df['str_time'] = test_df['str_time'].dt.hour*100 + test_df['str_time'].dt.minute
time_unique = test_df['str_time'].unique()

print(len(test_df['str_time'].value_counts()),"가 48이여야함")
time_list = [0, 30, 100, 130, 200, 230, 300, 330, 400, 430, 500, 530, 600, 630, 700, 730, 800, 830, 900, 930, 1000, 1030, 1100, 1130, 1200, 1230, 1300, 1330, 1400, 1430, 1500, 1530, 1600, 1630, 1700, 1730, 1800, 1830, 1900, 1930, 2000, 2030, 2100, 2130, 2200, 2230, 2300, 2330]
add_time = set(time_list) - set(time_unique.tolist())
add_time = list(add_time)
time_unique = np.append(time_unique, np.array(add_time))
uri_n = len(time_unique)

time_dict = dict()
for i in range(len(time_unique)):
    time_dict[time_unique[i]]=i
# event_id의 배열은 각각의 로그가 log_entry로 정렬된 sorted_df_acc_log_**파일에 대해서 전체 모든 행에 대해서 라벨링 한값을 나타내고
# 해당 배열은 event_id라는 컬럼으로 df에다가 추가된다.
event_id = []
for i in range(len(test_df)):
    event_id.append(time_dict[test_df.iloc[i]['str_time']])
#각각의 uri값에 따라서 라벨링된 event_id의 컬럼이 기존 df 데이터 프레임에 추가됨
test_df.reset_index(drop=True, inplace=True)
test_df['param_id']=pd.DataFrame(event_id)

unique_key = test_df["log_key_id"].unique()
unique_n = len(unique_key)
## log_entry값에 따라서 이벤트의 sequence와 정답 라벨링 결과를 튜플형식으로 정렬한다
param_dict=dict()
for i in range(unique_n):
    param_dict[unique_key[i]]=[]

for idx in range(len(test_df)):
    param_dict[test_df["log_key_id"][idx]].append((test_df.iloc[idx]['param_id'], test_df.iloc[idx]["RT01"]))
# 이벤트의 sequence만을 담는 배열을 만든다
uri_seq_test = []
for k in param_dict.keys():
    uri_seq_test.append(param_dict[k])
uri_seq_test.sort(key = len, reverse = True)

key_seq_test = []
for idx in range(len(test_df)):
    key_seq_test.append((test_df.iloc[idx]['log_key_id'], test_df.iloc[idx]['RT01']))

import os
import sys
import time
os.chdir("/workspace/deeplog")
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from Params import Hyperparameters
from log_key_detection_models import DeepLog
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 시퀀스를 담고있는 배열을 불러와서 input(앞서 나오는 n개의 로그)과 output(다음으로 나올 실제라벨값)으로 분류
def generate_param(seq, window_size, num_classes):
    num_sessions = 0
    inputs = []
    outputs = []

    # 각 sequence단위별로 반복
    for line in seq:
        num_sessions += 1
        for i in range(len(line) - window_size):
            inputs_tmp = []

            for j in line[i:i + window_size]:
                inputs_tmp.append(j[0])
            inputs.append(inputs_tmp)

            outputs.append(line[i + window_size][0])
    #     print("Number of sessions({}): {}".format(seq, num_classes))
    #     print("Number of seqs({}): {}".format(seq, len(inputs)))
    print(len(inputs))
    print(len(outputs))
    dataset = TensorDataset(
        torch.tensor(inputs, dtype=torch.float, device=device),
        torch.tensor(outputs, device=device)
    )
    return dataset

num_classes = 48 # unique한 시간의 클래스 개수
hparams = Hyperparameters()
pl.seed_everything(hparams.seed)

train_dset_param = generate_param(uri_seq, hparams.window_size, num_classes)
# dataloader를 이용해서 batch별로 불러옴
train_loader_param = DataLoader(
    train_dset_param, batch_size=hparams.batch_size, shuffle=True
)

valid_dset_param = generate_param(uri_seq_test, hparams.window_size, num_classes)
valid_loader_param = DataLoader(
    valid_dset_param, batch_size=hparams.batch_size, shuffle=False
)

# 모델 초기화
model = DeepLog(
    input_size=hparams.input_size,
    hidden_size=hparams.hidden_size,
    window_size=hparams.window_size,
    num_layers=hparams.num_layers,
    num_classes=num_classes,
    lr=hparams.lr
)
# early_stopping 조건
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, strict=False, verbose=True, mode="min"
)
logger = TensorBoardLogger("logs", name="deeplog")
# 모델에 대한 체크포인트 생성
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="edit/",
    filename="param_checkpoint-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min"
)
# 트레이너 정의
trainer = pl.Trainer(
    gpus=hparams.gpus,
    deterministic=True,
    logger=logger,
    callbacks=[early_stopping, checkpoint_callback],
    max_epochs=hparams.epoch
)
# 모델 학습
trainer.fit(model, train_loader_param, valid_loader_param)

train_key_seq = list(train_df['log_key_id'])
test_key_seq = list(test_df['log_key_id'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def generate_key(seq, window_size, num_classes):
    num_sessions = 0
    inputs = []
    outputs = []

    for i in range(len(seq) - window_size):
        inputs.append(seq[i: i + window_size])
        outputs.append(seq[i + window_size])

    dataset = TensorDataset(
        torch.tensor(inputs, dtype=torch.float, device=device),
        torch.tensor(outputs, device=device)
    )
    print(len(inputs))
    print(len(outputs))
    return dataset

num_classes = num_keys
hparams = Hyperparameters()
pl.seed_everything(hparams.seed)
train_dset = generate_key(train_key_seq, hparams.window_size, num_classes)

# dataloader를 이용해서 batch별로 불러옴
train_loader = DataLoader(
    train_dset, batch_size=hparams.batch_size, shuffle=True
)

valid_dset = generate_key(test_key_seq, hparams.window_size, num_classes)
valid_loader = DataLoader(
    valid_dset, batch_size=hparams.batch_size, shuffle=False
)

# 모델 초기화
model2 = DeepLog(
    input_size=hparams.input_size,
    hidden_size=hparams.hidden_size,
    window_size=hparams.window_size,
    num_layers=hparams.num_layers,
    num_classes=num_classes,
    lr=hparams.lr
)
# early_stopping 조건
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, strict=False, verbose=True, mode="min"
)
logger = TensorBoardLogger("logs", name="deeplog")
# 모델에 대한 체크포인트 생성
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="edit/",
    filename="key_checkpoint-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min"
)
# 트레이너 정의
trainer2 = pl.Trainer(
    gpus=hparams.gpus,
    deterministic=True,
    logger=logger,
    callbacks=[early_stopping, checkpoint_callback],
    max_epochs=hparams.epoch
)
# 모델 학습
trainer2.fit(model2, train_loader, valid_loader)

os.chdir("/workspace/deeplog")
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import pandas as pd
import numpy as np
import sys


# window_size에 맞추어서 input과 label의 라벨값을 리턴하는 함수
def generate_pred_param(file, window_size):
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
        line = list(map(lambda n: n, ln))
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
    correct = 0
    for i, j in zip(labeled, input_pred):
        if i in j:
            pred_result[idx] = 1
            correct += 1
        else:
            pred_result[idx] = 0
        idx += 1
    print(correct/total)
    return pred_result

#오차행렬, 정확도, 정밀도, 재현율 구하는 함수
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='weighted')
    recall = recall_score(y_test, pred, average='weighted')
    f1 = f1_score(y_test, pred, average='weighted')
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}'.format(accuracy, precision, recall))
    print('f1 스코어: {0: .4f}'.format(f1))


# window_size에 맞추어서 input과 label의 라벨값을 리턴하는 함수
def generate_pred_key(file, window_size):
    hdfs = list()
    hhhh = list()

    uri = []
    trid = []

    for line in file:
        uri.append(line[0])
        trid.append(line[1])

    hdfs.append(tuple(uri))
    hhhh.append(tuple(trid))

    return hdfs, hhhh

#모델의 오로지 성능평가를 위한 sequence
final_seq = []
for idx in range(len(test_df)):
    final_seq.append((test_df.iloc[idx]['log_key_id'], test_df.iloc[idx]['param_id'] ,test_df.iloc[idx]['RT01'], test_df.iloc[idx]['was_trid']))


# window_size에 맞추어서 input과 label의 라벨값을 리턴하는 함수
def generate_pred_total(file, window_size):
    hdfs = list()
    haaa = list()
    hhhh = list()

    uri = []
    time = []
    trid = []

    for line in file:
        uri.append(line[0])
        time.append(line[1])
        trid.append(line[2])

    hdfs.append(tuple(uri))
    haaa.append(tuple(time))
    hhhh.append(tuple(trid))

    return hdfs, haaa, hhhh

#sys.stdout = open('performance/model_result.txt', 'w')
device = DeepLog.device
hparams = Hyperparameters()
# 모델을 gpu로 불러온다
model_path = "/workspace/deeplog/edit"
key_model_name = "key_checkpoint-epoch=92-val_loss=1.63.ckpt"
key_bestmodel = DeepLog.load_from_checkpoint(os.path.join(model_path, key_model_name))
key_bestmodel.to(device)

test_key_normal_loader, test_normal_loader, y_test = generate_pred_total(final_seq, hparams.window_size)

model_path = "/workspace/deeplog/edit"
param_model_name = "param_checkpoint-epoch=99-val_loss=0.90.ckpt"
param_bestmodel = DeepLog.load_from_checkpoint(os.path.join(model_path, param_model_name))
param_bestmodel.to(device)

total = 0
correct = 0
fail = 0
proba = []
y_labeled = []
inputed = []
labeled = []
input_pred = []
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 모델에 sequence를 window_size별로 나누어서 학습시킨다
f = open(os.path.join("/workspace/deeplog/result", "final_model" + str(start_time) + ".txt"), "w")
with torch.no_grad():
    for line, line2, y in tqdm(zip(test_key_normal_loader, test_normal_loader, y_test)):
        for i in range(len(line) - hparams.window_size):
            seq_key = line[i:i + hparams.window_size]
            label_key = line[i + hparams.window_size]
            y_label = y[i + hparams.window_size]

            seq_param = line2[i:i + hparams.window_size]
            label_param = line2[i + hparams.window_size]

            total += 1

            if label_key == -2:
                continue

            seq_key = (
                torch.tensor(seq_key, dtype=torch.float, device=device).view(
                    -1, hparams.window_size, hparams.input_size
                )
            )

            label_key = torch.tensor(label_key, device=device).view(-1)
            label_param = torch.tensor(label_param, device=device).view(-1)
            y_label = torch.tensor(y_label, device=device).view(-1)
            output = key_bestmodel(seq_key)
            # 모델 output을 이용해서 상위 n개의 라벨을 predicted배열에 추가한다
            predicted = torch.argsort(output, 1)[0][-hparams.num_candidates:]
            # 모델 output을 softmax함수에 적용시켜서 각각의 클래스의 확률을 담는 proba배열을 만든다.
            y_pred = F.softmax(output, dim=1)
            proba.append(y_pred.cpu().numpy()[0].tolist())

            if label_key in predicted:
                if label_param == -2:
                    continue
                seq_param = (
                    torch.tensor(seq_param, dtype=torch.float, device=device).view(
                        -1, hparams.window_size, hparams.input_size
                    )
                )
                label_param = torch.tensor(label_param, device=device).view(-1)
                output_p = param_bestmodel(seq_param)
                predicted_p = torch.argsort(output_p, 1)[0][-hparams.num_candidates:]
                y_labeled.append(np.array(y_label.cpu()))
                #                 y_pred = F.softmax(output_p, dim=1)
                #                 proba.append(y_pred.cpu().numpy()[0].tolist())
                if label_param in predicted_p:
                    correct += 1
                    labeled.append(np.array(label_param.cpu())[0])
                    input_pred.append(np.array(predicted_p.cpu()))
                else:
                    fail += 1
                    labeled.append(np.array(label_param.cpu())[0])
                    input_pred.append(np.array(predicted_p.cpu()))

            else:
                fail += 1
                labeled.append(np.array(label_key.cpu())[0])
                input_pred.append(np.array(predicted.cpu()))
                y_labeled.append(np.array(y_label.cpu()))

#             inputed.append(np.array(seq_key.cpu()))
#             labeled.append(np.array(label_key.cpu())[0])
#             y_labeled.append(np.array(y_label.cpu()))
#             input_pred.append(np.array(predicted.cpu()))
#             total += 1
#             f.write(str(seq_key))
#             f.write("\n")
#             f.write(str(label_key))
#             f.write("\n")
#             f.write(str(predicted))
#             f.write("\n")
#             f.write(str(y_label))
#             f.write("\n")
#             # 후보군안에 실제 라벨이 존재할 경우 correct를 1 증가시킨다.
#             if label_key in predicted:
#                 correct += 1
#                 f.write("Predict Success")
#                 f.write("\n\n")
#             else:
#                 f.write("Predict Fail")
#                 f.write("\n\n")

f.close()
elapsed_time = time.time() - start_time

print("elapsed_time: {:.3f}s".format(elapsed_time))
print("correct_number : %d" % correct)
print("total : %d" % total)
accu = correct / total
# 전체 개수중에서 제대로 예측한 개수
print("accuracy : %f" % accu)

# Compute precision, recall and F1-measure
pred_list = get_list(labeled, input_pred, total)
y_labeled = np.array(y_labeled)
# 룰기반으로 생성된 정답데이터 셋을 이용해서 모델의 예측값과 성능평가
get_clf_eval(y_labeled, pred_list)

# 전체 데이터에 대해서 예측을 진행해서 마리아 디비에 넣을거임
df['RT01'] = df['RT01'].apply(lambda x: 1 if x=="정상" else 0)
df['str_time']=pd.to_datetime(df['str_time'])
df.sort_values(by=['str_time'], inplace = True)
df = df.astype({'dept_code':'string','position_code':'string'})
df['log_key'] = df['user_sn']+df['dept_code']+df['position_code']
key_unique = np.append(df['log_key'].unique(), df['log_key'].unique())
key_unique = np.unique(key_unique)
logk_n = len(key_unique)
key_dict = dict()
for i in range(len(key_unique)):
    key_dict[key_unique[i]]=i
num_keys = len(key_dict)

key_id = []
for i in range(len(df)):
    key_id.append(key_dict[df.iloc[i]['log_key']])
df.reset_index(drop=True, inplace=True)
df['log_key_id']=pd.DataFrame(key_id)
df['str_time']=pd.to_datetime(df['str_time'])
df['str_time'] = df.str_time.dt.floor('30min')
df.sort_values(by=['str_time'], inplace = True)
df['str_time'] = df['str_time'].dt.hour*100 + df['str_time'].dt.minute
time_unique = df['str_time'].unique()

time_list = [0, 30, 100, 130, 200, 230, 300, 330, 400, 430, 500, 530, 600, 630, 700, 730, 800, 830, 900, 930, 1000, 1030, 1100, 1130, 1200, 1230, 1300, 1330, 1400, 1430, 1500, 1530, 1600, 1630, 1700, 1730, 1800, 1830, 1900, 1930, 2000, 2030, 2100, 2130, 2200, 2230, 2300, 2330]
add_time = set(time_list) - set(time_unique.tolist())
add_time = list(add_time)
time_unique = np.append(time_unique, np.array(add_time))
uri_n = len(time_unique)

time_dict = dict()
for i in range(len(time_unique)):
    time_dict[time_unique[i]]=i
# event_id의 배열은 각각의 로그가 log_entry로 정렬된 sorted_df_acc_log_**파일에 대해서 전체 모든 행에 대해서 라벨링 한값을 나타내고
# 해당 배열은 event_id라는 컬럼으로 df에다가 추가된다.
event_id = []
for i in range(len(df)):
    event_id.append(time_dict[df.iloc[i]['str_time']])
#각각의 uri값에 따라서 라벨링된 event_id의 컬럼이 기존 df 데이터 프레임에 추가됨
df.reset_index(drop=True, inplace=True)
df['param_id']=pd.DataFrame(event_id)

maria_seq = []
for idx in range(len(df)):
    maria_seq.append((df.iloc[idx]['log_key_id'], df.iloc[idx]['param_id'] ,df.iloc[idx]['RT01'], df.iloc[idx]['was_trid']))


# window_size에 맞추어서 input과 label의 라벨값을 리턴하는 함수
def generate_pred_total_maria(file, window_size):
    hdfs = list()
    haaa = list()
    hhhh = list()
    hhll = list()

    uri = []
    time = []
    rt = []
    trid = []

    for line in file:
        uri.append(line[0])
        time.append(line[1])
        rt.append(line[2])
        trid.append(line[3])

    hdfs.append(tuple(uri))
    haaa.append(tuple(time))
    hhhh.append(tuple(rt))
    hhll.append(tuple(trid))

    return hdfs, haaa, hhhh, hhll

test_key_normal_loader, test_normal_loader, y_test, was_trid = generate_pred_total_maria(maria_seq, hparams.window_size)
total = 0
correct = 0
fail = 0
proba = []
y_labeled = []
inputed = []
labeled = []
input_pred = []
trid_list = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 모델에 sequence를 window_size별로 나누어서 학습시킨다
f = open(os.path.join("/workspace/deeplog/result", "final_model" + str(start_time) + ".txt"), "w")
with torch.no_grad():
    for line, line2, y, was_trid in tqdm(zip(test_key_normal_loader, test_normal_loader, y_test, was_trid)):
        for i in range(len(line) - hparams.window_size):
            seq_key = line[i:i + hparams.window_size]
            label_key = line[i + hparams.window_size]
            y_label = y[i + hparams.window_size]
            trid = was_trid[i + hparams.window_size]

            seq_param = line2[i:i + hparams.window_size]
            label_param = line2[i + hparams.window_size]

            total += 1

            if label_key == -2:
                continue

            seq_key = (
                torch.tensor(seq_key, dtype=torch.float, device=device).view(
                    -1, hparams.window_size, hparams.input_size
                )
            )

            label_key = torch.tensor(label_key, device=device).view(-1)
            label_param = torch.tensor(label_param, device=device).view(-1)
            y_label = torch.tensor(y_label, device=device).view(-1)
            output = key_bestmodel(seq_key)
            # 모델 output을 이용해서 상위 n개의 라벨을 predicted배열에 추가한다
            predicted = torch.argsort(output, 1)[0][-hparams.num_candidates:]
            # 모델 output을 softmax함수에 적용시켜서 각각의 클래스의 확률을 담는 proba배열을 만든다.
            y_pred = F.softmax(output, dim=1)
            proba.append(y_pred.cpu().numpy()[0].tolist())
            trid_list.append(trid)

            if label_key in predicted:
                if label_param == -2:
                    continue
                seq_param = (
                    torch.tensor(seq_param, dtype=torch.float, device=device).view(
                        -1, hparams.window_size, hparams.input_size
                    )
                )
                label_param = torch.tensor(label_param, device=device).view(-1)
                output_p = param_bestmodel(seq_param)
                predicted_p = torch.argsort(output_p, 1)[0][-hparams.num_candidates:]
                y_labeled.append(np.array(y_label.cpu()))

                if label_param in predicted_p:
                    correct += 1
                    labeled.append(np.array(label_param.cpu())[0])
                    input_pred.append(np.array(predicted_p.cpu()))
                else:
                    fail += 1
                    labeled.append(np.array(label_param.cpu())[0])
                    input_pred.append(np.array(predicted_p.cpu()))

            else:
                fail += 1
                labeled.append(np.array(label_key.cpu())[0])
                input_pred.append(np.array(predicted.cpu()))
                y_labeled.append(np.array(y_label.cpu()))

f.close()

# 다음에 올거라고 예측하는 event 라벨이 모델의 candidates 안에 존재하면 0, 존재하지않으면 1을 가지는 배열 생성
# 추후 룰 기반 결과와 비교해서 성능평가
def get_list_maria(labeled, input_pred, total):
    pred_result = []
    idx = 0
    correct = 0
    for i, j in zip(labeled, input_pred):
        if i in j:
            pred_result.append("정상")
            correct += 1
        else:
            pred_result.append("비정상")
        idx += 1
    print(correct/total)
    return pred_result

pred_list = get_list_maria(labeled, input_pred, total)
trid_list = np.array(trid_list)
deeplog_result = pd.DataFrame({'was_trid':trid_list, 'prediction':pred_list})

import pymysql
import datetime
deeplog_result.to_sql(name='N_VIOLATET_ML_DEEPLOG',con=engine, index=False, if_exists='replace')