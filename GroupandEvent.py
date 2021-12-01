import pandas as pd
import numpy as np
import pickle
import os

# 로그데이터 불러와서 전처리하기
data = pd.read_csv('/home/user/deeplog/acc_log.csv')
data=data.iloc[:,1:]
data['exec_str_time']=pd.to_datetime(data['exec_str_time'])
# 룰 기반 라벨링결과 불러오기
rtime_result_train = pd.read_csv("/home/user/deeplog/RT01.csv")
rtime_result_train = rtime_result_train.iloc[:,1:]
df_acc_log=pd.merge(data,rtime_result_train,how='right',left_on='was_trid',right_on='trid')

# 룰 기반 라벨링 결과를 숫자로 변환
for i in range(df_acc_log.shape[0]):
    if df_acc_log['rt'].iloc[i] == "비정상":
        df_acc_log['rt'].iloc[i] = 1
    elif df_acc_log['rt'].iloc[i] == "점검대상":
        df_acc_log['rt'].iloc[i] = 0
    elif df_acc_log['rt'].iloc[i] == "정상":
        df_acc_log['rt'].iloc[i] = 0
# 데이터가 저장될 위치 지정
Data_root = "/home/user/deeplog/data"
# 원하는 컬럼만 추출
col = df_acc_log.columns
remove_cols=['Unnamed: 0', 'src_ip_long','end_time','was_ip','sg_keyword_reason_bit',
             'logging_time_ui','str_time_ui','elapsed_time','logging_time','was_ip_long','download_check',
             'sg_multi_reason_bit','src_ip','sg_reason','was_country_lat','work_name','sys_crud_auth','was_trid',
             'sg_multi_reason','src_country_lat','security_grade','params','sys_id_auth','security_score',
             'work_time', 'end_time_ui','sg_keyword_reason','keyword_count','sg_reason_bit','acc_luid', 'trid']
cols = list(set(col)-set(remove_cols))
df = df_acc_log[cols]
# null데이터를 제거한 known_df파일을 생성
with open(os.path.join(Data_root, "known_df"), "wb") as file:
    pickle.dump(df, file)
# 부서코드별 정렬
df.sort_values(by=['dept_code', 'str_time'], inplace = True)
with open(os.path.join(Data_root, "sorted_df_acc_log_dept_code"), "wb") as file:
    pickle.dump(df, file)

# 직급코드별 정렬
with open(os.path.join(Data_root, "known_df"),"rb") as file:
    df = pickle.load(file)
df.sort_values(by=['position_code', 'str_time'], inplace = True)
with open(os.path.join(Data_root, "sorted_df_acc_log_position_code"), "wb") as file:
    pickle.dump(df, file)

# 개인별 정렬
with open(os.path.join(Data_root, "known_df"), "rb") as file:
    df = pickle.load(file)
df.sort_values(by=['user_sn', 'str_time'], inplace=True)
with open(os.path.join(Data_root, "sorted_df_acc_log_user_sn"), "wb") as file:
    pickle.dump(df, file)

# 분석을 위한 log_key값을 1. dept_code 2. position_code 3.user_sn 중 선택
log_entry = 'dept_code'
# log_key값에 따라 앞서 정렬한 데이터 셋 로드
with open(os.path.join(Data_root, "sorted_df_acc_log_"+log_entry), "rb") as file:
    df = pickle.load(file)
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
dfvc = df["str_time"].value_counts()

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

unique_key = df[log_entry].unique()
unique_n = len(df[log_entry].unique())
## log_entry값에 따라서 이벤트의 sequence와 정답 라벨링 결과를 튜플형식으로 정렬한다
dept_dict=dict()
for i in range(unique_n):
    dept_dict[unique_key[i]]=[]

for idx in range(len(df)):
    dept_dict[df[log_entry][idx]].append((df.iloc[idx]['event_id'], df.iloc[idx]["rt"]))
# 이벤트의 sequence만을 담는 배열을 만든다
uri_seq = []
for k in dept_dict.keys():
    uri_seq.append(dept_dict[k])
uri_seq.sort(key = len, reverse = True)
# train과 test데이터 셋을 비율에 따라 분리한다
ratio = 0.8
train_num = int(len(df)*ratio)
tmp = 0
uri_train = []
uri_test = []

for seq in uri_seq:
    tmp = len(seq)
    idx = int(ratio * tmp)
    uri_train.append(seq[:idx])
    uri_test.append(seq[idx:])

# train 데이터셋 저장
with open(os.path.join(Data_root, "uri_train_" + log_entry), "wb") as file:
    pickle.dump(uri_train, file)
# test 데이터셋 저장
with open(os.path.join(Data_root, "uri_test_" + log_entry), "wb") as file:
    pickle.dump(uri_test, file)
