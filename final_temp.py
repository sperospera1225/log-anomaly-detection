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





