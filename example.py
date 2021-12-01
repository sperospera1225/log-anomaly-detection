# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 02:11:15 2021

@author: 85311
"""

import json
import sqlalchemy
import datetime

from common import Common
import pandas as pd

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