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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

import numpy
import h5py
from tensorflow.keras import callbacks
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import plot_confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    mean_squared_error, mean_absolute_error
from sklearn import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
from numpy import nan
import datetime as dt
import sys
import os
import matplotlib.pyplot as plt
import json
import sqlalchemy
from elasticsearch import Elasticsearch

import numpy
import h5py
from tensorflow.keras import callbacks
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import plot_confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    mean_squared_error, mean_absolute_error
from sklearn import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
from numpy import nan
import datetime as dt
import sys
import os
import matplotlib.pyplot as plt
import json
import sqlalchemy
from elasticsearch import Elasticsearch
from sklearn.preprocessing import LabelEncoder
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

import json
import sqlalchemy
import datetime
from common import Common

########## DB정보 불러오기 ######################
with open('setting.json','r', encoding='utf-8') as json_file:
         setting_data = json.load(json_file)
db_host, db_user, db_port, db_pwd, db_database = setting_data['db_host'], setting_data['db_user'], setting_data['db_port'], setting_data['db_pwd'], setting_data['db_database']
engine = sqlalchemy.create_engine("mysql+pymysql://"+db_user+":"+db_pwd+"@"+db_host+":"+str(db_port)+"/"+db_database)


########## N_code_info 테이블에서 설정값 불러오기 ######################
setting_data = pd.read_sql_table(table_name="N_CODE_INFO", con=engine)
setting_data = setting_data[setting_data['CD1'] == '0007']
setting_data = setting_data[setting_data['CD2'] == 'setting']
setting_data = eval(list(setting_data['DATAJSON'])[0])
input_data_ip = setting_data['input_data_ip']
input_data_port = setting_data['input_data_port']
input_data_index = setting_data['input_data_index']
sql_data_index = setting_data['sql_data_index']
rule_start_date = setting_data['rule_start_date']
rule_finish_date = setting_data['rule_finish_date']
test_start_date = setting_data['test_start_date']
test_finish_date = setting_data['test_finish_date']
threshold_check = setting_data['threshold_check']
time_threshold_check = setting_data['time_threshold_check']
visualize_check = eval(setting_data['visualize_check'])
depth = setting_data['depth']
criteria = setting_data['criteria']
time_id_threshold, time_department_threshold, time_position_threshold = setting_data['time_id_threshold'], setting_data['time_department_threshold'], setting_data['time_position_threshold']
url_id_threshold, url_department_threshold, url_position_threshold = setting_data['url_id_threshold'], setting_data['url_department_threshold'], setting_data['url_position_threshold']
sql_id_threshold, sql_department_threshold, sql_position_threshold = setting_data['sql_id_threshold'], setting_data['sql_department_threshold'], setting_data['sql_position_threshold']
visualize_ip, visualize_port = setting_data['visualize_ip'], setting_data['visualize_port']
rtime_num = setting_data['rtime_num']
cutline = setting_data['cutline']
hour_type = setting_data['hour_type']

########## ES 데이터 불러오기 ######################
cm = Common()

print("data loading" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
rule_data = cm.es_data_search_date(input_data_ip, input_data_port, input_data_index, 'str_time', rule_start_date, rule_finish_date)
sql_rule_data = cm.es_data_search_date(input_data_ip, input_data_port, sql_data_index, 'exec_str_time', rule_start_date, rule_finish_date)
print("data loading finish" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

print("test_data loading" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
test_data = cm.es_data_search_date(input_data_ip, input_data_port, input_data_index,  'str_time', test_start_date, test_finish_date)
sql_test_data = cm.es_data_search_date(input_data_ip, input_data_port, sql_data_index, 'exec_str_time', test_start_date, test_finish_date)
print("test_data loading finish" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

acc_log = pd.DataFrame.from_dict(rule_data)
sql_log = pd.DataFrame.from_dict(sql_rule_data)
LE = LabelEncoder() #string을 정수로 변환할 인코더

merge_data = pd.merge(acc_log[['dept_code','was_trid', 'position_code', 'user_sn', 'uri', 'str_time']],sql_log[['was_trid','sql_syntax','rst_row_cnt']],how='inner',on=['was_trid'])
merge_data['sql_syntax'] = LE.fit_transform(merge_data['sql_syntax']) # sql_syntax 정수 인코딩

df = merge_data.sort_values(by=['dept_code', 'str_time'])
# 분석을 위한 기준값을 1. dept_code 2. position_code 3.user_sn 중 선택
log_entry = 'dept_code'

# event를 시간기준으로 설정한뒤 30분단위로 시간을 라벨링(총 48개의 클래스)
df['str_time']=pd.to_datetime(df['str_time'])
df['str_time'] = df.str_time.dt.floor('30min')
df.sort_values(by=['str_time'], inplace = True)
df['str_time'] = df['str_time'].dt.hour*100 + df['str_time'].dt.minute
uri_unique = df['str_time'].unique()
time_list = [0, 30, 100, 130, 200, 230, 300, 330, 400, 430, 500, 530, 600, 630, 700, 730, 800, 830, 900, 930, 1000, 1030, 1100, 1130, 1200, 1230, 1300, 1330, 1400, 1430, 1500, 1530, 1600, 1630, 1700, 1730, 1800, 1830, 1900, 1930, 2000, 2030, 2100, 2130, 2200, 2230, 2300, 2330]
add_time = set(time_list) - set(uri_unique.tolist())
add_time = list(add_time)
uri_unique = np.append(uri_unique, np.array(add_time))
uri_n = len(uri_unique)
# 딕셔러리 형식에 uri_unique key값으로 숫자순서대로 라벨링
uri_dict = dict()
for i in range(len(uri_unique)):
    uri_dict[uri_unique[i]]=i
# event_id의 배열은 각각의 로그가 log_entry로 정렬된 sorted_df_acc_log_**파일에 대해서 전체 모든 행에 대해서 라벨링 한값을 나타내고
# 해당 배열은 event_id라는 컬럼으로 df에다가 추가된다.
event_id = []
for i in range(len(df)):
    event_id.append(uri_dict[df.iloc[i]['str_time']])
#각각의 uri값에 따라서 라벨링된 event_id의 컬럼이 기존 df 데이터 프레임에 추가됨
df.reset_index(drop=True, inplace=True)
df['event_id']=pd.DataFrame(event_id)

#UNKNOWN제거
indexNames = df[df['dept_code']>="UNKNOWN"].index
df.drop(indexNames, inplace=True)
df.reset_index(drop=True, inplace=True)

unique_key = df[log_entry].unique()
unique_n = len(df[log_entry].unique())
## log_entry값에 따라서 이벤트의 sequence와 정답 라벨링 결과를 튜플형식으로 정렬한다
dept_dict=dict()
for i in range(unique_n):
    dept_dict[unique_key[i]]=[]

for idx in range(len(df)):
    dept_dict[df[log_entry][idx]].append((df.iloc[idx]['event_id'], df.iloc[idx]["was_trid"]))

# 이벤트의 sequence만을 담는 배열을 만든다
uri_seq = []
for k in dept_dict.keys():
    uri_seq.append(dept_dict[k])
uri_seq.sort(key = len, reverse = True)

# train과 test데이터 셋을 비율에 따라 분리한다
ratio = 0.8
train_num = int(len(df)*ratio)
tmp = 0
seq_train = []
seq_test = []

for seq in uri_seq:
    tmp = len(seq)
    idx = int(ratio * tmp)
    seq_train.append(seq[:idx])
    seq_test.append(seq[idx:])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 시퀀스를 담고있는 배열을 불러와서 input(앞서 나오는 n개의 로그)과 output(다음으로 나올 실제라벨값)으로 분류
def generate(seq, window_size, num_classes):
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
    print("Number of sessions({}): {}".format(seq, num_classes))
    print("Number of seqs({}): {}".format(seq, len(inputs)))

    dataset = TensorDataset(
        torch.tensor(inputs, dtype=torch.float, device=device),
        torch.tensor(outputs, device=device)
    )
    return dataset

hparams = Hyperparameters()
pl.seed_everything(hparams.seed)
train_dset = generate(seq_train, hparams.window_size, hparams.num_classes)
# dataloader를 이용해서 batch별로 불러옴
train_loader = DataLoader(
    train_dset, batch_size=hparams.batch_size, shuffle=True
)
valid_dset = generate(seq_test, hparams.window_size, hparams.num_classes)
valid_loader = DataLoader(
    valid_dset, batch_size=hparams.batch_size, shuffle=False
)
# 모델 초기화
model = DeepLog(
    input_size=hparams.input_size,
    hidden_size=hparams.hidden_size,
    window_size=hparams.window_size,
    num_layers=hparams.num_layers,
    num_classes=hparams.num_classes,
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
    dirpath="deeplog/",
    filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
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
trainer.fit(model, train_loader, valid_loader)

import os
import sys
import time
os.chdir("/workspace/deeplog")
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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import pandas as pd
import numpy as np
import sys


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

import numpy as np

sys.stdout = open('performance/model_result.txt', 'w')
device = DeepLog.device
hparams = Hyperparameters()
# 모델을 gpu로 불러온다
model_path = "/workspace/deeplog/deeplog"
# best model 지정
model_name = "checkpoint-epoch=63-val_loss=0.03.ckpt"

bestmodel = DeepLog.load_from_checkpoint(os.path.join(model_path, model_name))
bestmodel.to(device)
print("model_path: {}".format(os.path.join(model_path, model_name)))
test_normal_loader, was_trid = generate_pred(seq_test, hparams.window_size)

total = 0
correct = 0
proba = []
trid_list = []
inputed = []
labeled = []
input_pred = []
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 모델에 sequence를 window_size별로 나누어서 학습시킨다
f = open(os.path.join("/workspace/deeplog/result", "result_" + "seq_test" + str(start_time) + ".txt"), "w")
with torch.no_grad():
    for line, y in tqdm(zip(test_normal_loader, was_trid)):
        for i in range(len(line) - hparams.window_size):
            seq = line[i:i + hparams.window_size]
            label = line[i + hparams.window_size]
            trid = y[i + hparams.window_size]
            if label == -2:
                continue

            seq = (
                torch.tensor(seq, dtype=torch.float, device=device).view(
                    -1, hparams.window_size, hparams.input_size
                )
            )
            label = torch.tensor(label, device=device).view(-1)
            #trid = torch.tensor(trid, device=device).view(-1)
            output = bestmodel(seq)
            # 모델 output을 이용해서 상위 n개의 라벨을 predicted배열에 추가한다
            predicted = torch.argsort(output, 1)[0][-hparams.num_candidates:]
            # 모델 output을 softmax함수에 적용시켜서 각각의 클래스의 확률을 담는 proba배열을 만든다.
            y_pred = F.softmax(output, dim=1)
            proba.append(y_pred.cpu().numpy()[0].tolist())

            inputed.append(np.array(seq.cpu()))
            labeled.append(np.array(label.cpu())[0])
            trid_list.append(np.array(trid))
            input_pred.append(np.array(predicted.cpu()))
            total += 1
            f.write(str(seq))
            f.write("\n")
            f.write(str(label))
            f.write("\n")
            f.write(str(predicted))
            f.write("\n")
            f.write(str(trid))
            f.write("\n")
            # 후보군안에 실제 라벨이 존재할 경우 correct를 1 증가시킨다.
            if label in predicted:
                correct += 1
                f.write("Predict Success")
                f.write("\n\n")
            else:
                f.write("Predict Fail")
                f.write("\n\n")

f.close()
elapsed_time = time.time() - start_time

print("elapsed_time: {:.3f}s".format(elapsed_time))
print("correct_number : %d" % correct)
print("total : %d" % total)
accu = (total - correct) / total
# 전체 개수중에서 제대로 예측한 개수
print("accuracy : %f" % accu)
# Compute precision, recall and F1-measure
pred_list = get_list(labeled, input_pred, total)
trid_list = np.array(trid_list)

#mariaDB에 넣을 최종 데이터 프레임 (TRANSCATION ID랑 정상/비정상 유무 포함)
deeplog_result = pd.DataFrame({'was_trid':trid_list, 'prediction':pred_list.flatten()})

import pymysql
import datetime

#데이터 프레임 생성후에 to_sql해서 넣기 (transaction ID에 대한 정보를 유지해서 새로운 데이터프레임을 만듬)
deeplog_result.to_sql("N_VIOLATE_ML_DEEPLOG", con = engine, index = False, if_exists = 'append')