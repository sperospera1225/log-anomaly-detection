# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:58:42 2021

@author: 85311
"""

import pymysql
import datetime
import pandas as pd

#conn = pymysql.connect(host = '112.220.209.210', port = 13008, user = 'root', password = 'samocns15', db = 'CAMOMILE')
#dbcon = {'host': db_host, 'port' : db_port,'user' : db_user, 'pwd' : db_pwd, 'db' : db_database}
#    db_host = '112.220.209.210'
#   db_port = 13008
#   db_user = 'root'
#   db_pwd = 'samocns15' 
#   db_database = 'CAMOMILE'
def dbcreate(dbcon):
    conn = pymysql.connect(host = dbcon['host'], port = dbcon['port'], user = dbcon['user'], password = dbcon['pwd'], db = dbcon['db'])
    cursor = conn.cursor()
    
    sql = """CREATE TABLE `N_VIOLATE` (
    	`was_trid` VARCHAR(125) NOT NULL COMMENT '로그아이디\\n' COLLATE 'utf8_general_ci',
    	`sdate` VARCHAR(8) NOT NULL COMMENT 'yyyyMMdd' COLLATE 'utf8_general_ci',
    	`str_time_ui` DATETIME NOT NULL COMMENT '접속시작시간',
    	`end_time_ui` DATETIME NULL DEFAULT NULL COMMENT '접속종료시간',
    	`app_nm` VARCHAR(45) NULL DEFAULT NULL COMMENT '시스템-어플리케이션 이름' COLLATE 'utf8_general_ci',
    	`user_id` VARCHAR(45) NULL DEFAULT NULL COMMENT '접속아이디' COLLATE 'utf8_general_ci',
    	`user_name` VARCHAR(125) NULL DEFAULT NULL COLLATE 'utf8_general_ci',
    	`data` LONGTEXT NULL DEFAULT NULL COMMENT 'al_was_acc_log' COLLATE 'utf8mb4_bin',
    	`violate` VARCHAR(10) NULL DEFAULT NULL COMMENT '판정상태' COLLATE 'utf8_general_ci',
    	`comment` VARCHAR(4000) NULL DEFAULT NULL COMMENT '사유' COLLATE 'utf8_general_ci',
    	`report_type` VARCHAR(100) NOT NULL COMMENT '보고서 종류' COLLATE 'utf8_general_ci',
    	`type` VARCHAR(100) NOT NULL COMMENT '항목' COLLATE 'utf8_general_ci',
    	`inspec_detail` VARCHAR(100) NOT NULL COMMENT '점검내용' COLLATE 'utf8_general_ci',
    	`change_dt` TIMESTAMP NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
    	PRIMARY KEY (`was_trid`, `inspec_detail`, `report_type`) USING BTREE,
    	INDEX `sdate_ix` (`sdate`) USING BTREE
    )
    COLLATE='utf8_general_ci'
    ENGINE=MyISAM
    ;"""
    cursor.execute(sql)
    return

def usertablecreate(dbcon):
    conn = pymysql.connect(host = dbcon['host'], port = dbcon['port'], user = dbcon['user'], password = dbcon['pwd'], db = dbcon['db'])
    cursor = conn.cursor()
    sql = """CREATE TABLE `N_VIOLATEUserTable` (
    `user_id` VARCHAR(45) NOT NULL COMMENT '접속아이디' COLLATE 'utf8_general_ci',
    `user_name` VARCHAR(125) NOT NULL COMMENT '접속사용자' COLLATE 'utf8_general_ci',
    `week` VARCHAR(45) NOT NULL COMMENT '평일/주말' COLLATE 'utf8_general_ci',
    `0H` VARCHAR(45) NOT NULL COMMENT '0~1시' COLLATE 'utf8_general_ci',
    `1H` VARCHAR(45) NOT NULL COMMENT '1~2시' COLLATE 'utf8_general_ci',
    `2H` VARCHAR(45) NOT NULL COMMENT '2~3시' COLLATE 'utf8_general_ci',
    `3H` VARCHAR(45) NOT NULL COMMENT '3~4시' COLLATE 'utf8_general_ci',
    `4H` VARCHAR(45) NOT NULL COMMENT '4~5시' COLLATE 'utf8_general_ci',
    `5H` VARCHAR(45) NOT NULL COMMENT '5~6시' COLLATE 'utf8_general_ci',
    `6H` VARCHAR(45) NOT NULL COMMENT '6~7시' COLLATE 'utf8_general_ci',
    `7H` VARCHAR(45) NOT NULL COMMENT '7~8시' COLLATE 'utf8_general_ci',
    `8H` VARCHAR(45) NOT NULL COMMENT '8~9시' COLLATE 'utf8_general_ci',
    `9H` VARCHAR(45) NOT NULL COMMENT '9~10시' COLLATE 'utf8_general_ci',
    `10H` VARCHAR(45) NOT NULL COMMENT '10~11시' COLLATE 'utf8_general_ci',
    `11H` VARCHAR(45) NOT NULL COMMENT '11~12시' COLLATE 'utf8_general_ci',
    `12H` VARCHAR(45) NOT NULL COMMENT '12~13시' COLLATE 'utf8_general_ci',
    `13H` VARCHAR(45) NOT NULL COMMENT '13~14시' COLLATE 'utf8_general_ci',
    `14H` VARCHAR(45) NOT NULL COMMENT '14~15시' COLLATE 'utf8_general_ci',
    `15H` VARCHAR(45) NOT NULL COMMENT '15~16시' COLLATE 'utf8_general_ci',
    `16H` VARCHAR(45) NOT NULL COMMENT '16~17시' COLLATE 'utf8_general_ci',
    `17H` VARCHAR(45) NOT NULL COMMENT '17~18시' COLLATE 'utf8_general_ci',
    `18H` VARCHAR(45) NOT NULL COMMENT '18~19시' COLLATE 'utf8_general_ci',
    `19H` VARCHAR(45) NOT NULL COMMENT '19~20시' COLLATE 'utf8_general_ci',
    `20H` VARCHAR(45) NOT NULL COMMENT '20~21시' COLLATE 'utf8_general_ci',
    `21H` VARCHAR(45) NOT NULL COMMENT '21~22시' COLLATE 'utf8_general_ci',
    `22H` VARCHAR(45) NOT NULL COMMENT '22~23시' COLLATE 'utf8_general_ci',
    `23H` VARCHAR(45) NOT NULL COMMENT '23~24시' COLLATE 'utf8_general_ci',
    PRIMARY KEY (`user_id`, `user_name`, `week`) USING BTREE
    )
    COLLATE='utf8_general_ci'
    ENGINE=MyISAM
    ;"""
    cursor.execute(sql)
    return

def positiontablecreate(dbcon):
    conn = pymysql.connect(host = dbcon['host'], port = dbcon['port'], user = dbcon['user'], password = dbcon['pwd'], db = dbcon['db'])
    cursor = conn.cursor()
    sql = """CREATE TABLE `N_VIOLATEPositionTable` (
    `position_name` VARCHAR(125) NOT NULL COMMENT '직급' COLLATE 'utf8_general_ci',
    `week` VARCHAR(45) NOT NULL COMMENT '평일/주말' COLLATE 'utf8_general_ci',
    `0H` VARCHAR(45) NOT NULL COMMENT '0~1시' COLLATE 'utf8_general_ci',
    `1H` VARCHAR(45) NOT NULL COMMENT '1~2시' COLLATE 'utf8_general_ci',
    `2H` VARCHAR(45) NOT NULL COMMENT '2~3시' COLLATE 'utf8_general_ci',
    `3H` VARCHAR(45) NOT NULL COMMENT '3~4시' COLLATE 'utf8_general_ci',
    `4H` VARCHAR(45) NOT NULL COMMENT '4~5시' COLLATE 'utf8_general_ci',
    `5H` VARCHAR(45) NOT NULL COMMENT '5~6시' COLLATE 'utf8_general_ci',
    `6H` VARCHAR(45) NOT NULL COMMENT '6~7시' COLLATE 'utf8_general_ci',
    `7H` VARCHAR(45) NOT NULL COMMENT '7~8시' COLLATE 'utf8_general_ci',
    `8H` VARCHAR(45) NOT NULL COMMENT '8~9시' COLLATE 'utf8_general_ci',
    `9H` VARCHAR(45) NOT NULL COMMENT '9~10시' COLLATE 'utf8_general_ci',
    `10H` VARCHAR(45) NOT NULL COMMENT '10~11시' COLLATE 'utf8_general_ci',
    `11H` VARCHAR(45) NOT NULL COMMENT '11~12시' COLLATE 'utf8_general_ci',
    `12H` VARCHAR(45) NOT NULL COMMENT '12~13시' COLLATE 'utf8_general_ci',
    `13H` VARCHAR(45) NOT NULL COMMENT '13~14시' COLLATE 'utf8_general_ci',
    `14H` VARCHAR(45) NOT NULL COMMENT '14~15시' COLLATE 'utf8_general_ci',
    `15H` VARCHAR(45) NOT NULL COMMENT '15~16시' COLLATE 'utf8_general_ci',
    `16H` VARCHAR(45) NOT NULL COMMENT '16~17시' COLLATE 'utf8_general_ci',
    `17H` VARCHAR(45) NOT NULL COMMENT '17~18시' COLLATE 'utf8_general_ci',
    `18H` VARCHAR(45) NOT NULL COMMENT '18~19시' COLLATE 'utf8_general_ci',
    `19H` VARCHAR(45) NOT NULL COMMENT '19~20시' COLLATE 'utf8_general_ci',
    `20H` VARCHAR(45) NOT NULL COMMENT '20~21시' COLLATE 'utf8_general_ci',
    `21H` VARCHAR(45) NOT NULL COMMENT '21~22시' COLLATE 'utf8_general_ci',
    `22H` VARCHAR(45) NOT NULL COMMENT '22~23시' COLLATE 'utf8_general_ci',
    `23H` VARCHAR(45) NOT NULL COMMENT '23~24시' COLLATE 'utf8_general_ci',
    PRIMARY KEY (`position_name`, `week`) USING BTREE
    )
    COLLATE='utf8_general_ci'
    ENGINE=MyISAM
    ;"""
    cursor.execute(sql)
    return

def departmenttablecreate(dbcon):
    conn = pymysql.connect(host = dbcon['host'], port = dbcon['port'], user = dbcon['user'], password = dbcon['pwd'], db = dbcon['db'])
    cursor = conn.cursor()
    sql = """CREATE TABLE `N_VIOLATEDepartmentTable` (
    `department_name` VARCHAR(125) NOT NULL COMMENT '부서' COLLATE 'utf8_general_ci',
    `week` VARCHAR(45) NOT NULL COMMENT '평일/주말' COLLATE 'utf8_general_ci',
    `0H` VARCHAR(45) NOT NULL COMMENT '0~1시' COLLATE 'utf8_general_ci',
    `1H` VARCHAR(45) NOT NULL COMMENT '1~2시' COLLATE 'utf8_general_ci',
    `2H` VARCHAR(45) NOT NULL COMMENT '2~3시' COLLATE 'utf8_general_ci',
    `3H` VARCHAR(45) NOT NULL COMMENT '3~4시' COLLATE 'utf8_general_ci',
    `4H` VARCHAR(45) NOT NULL COMMENT '4~5시' COLLATE 'utf8_general_ci',
    `5H` VARCHAR(45) NOT NULL COMMENT '5~6시' COLLATE 'utf8_general_ci',
    `6H` VARCHAR(45) NOT NULL COMMENT '6~7시' COLLATE 'utf8_general_ci',
    `7H` VARCHAR(45) NOT NULL COMMENT '7~8시' COLLATE 'utf8_general_ci',
    `8H` VARCHAR(45) NOT NULL COMMENT '8~9시' COLLATE 'utf8_general_ci',
    `9H` VARCHAR(45) NOT NULL COMMENT '9~10시' COLLATE 'utf8_general_ci',
    `10H` VARCHAR(45) NOT NULL COMMENT '10~11시' COLLATE 'utf8_general_ci',
    `11H` VARCHAR(45) NOT NULL COMMENT '11~12시' COLLATE 'utf8_general_ci',
    `12H` VARCHAR(45) NOT NULL COMMENT '12~13시' COLLATE 'utf8_general_ci',
    `13H` VARCHAR(45) NOT NULL COMMENT '13~14시' COLLATE 'utf8_general_ci',
    `14H` VARCHAR(45) NOT NULL COMMENT '14~15시' COLLATE 'utf8_general_ci',
    `15H` VARCHAR(45) NOT NULL COMMENT '15~16시' COLLATE 'utf8_general_ci',
    `16H` VARCHAR(45) NOT NULL COMMENT '16~17시' COLLATE 'utf8_general_ci',
    `17H` VARCHAR(45) NOT NULL COMMENT '17~18시' COLLATE 'utf8_general_ci',
    `18H` VARCHAR(45) NOT NULL COMMENT '18~19시' COLLATE 'utf8_general_ci',
    `19H` VARCHAR(45) NOT NULL COMMENT '19~20시' COLLATE 'utf8_general_ci',
    `20H` VARCHAR(45) NOT NULL COMMENT '20~21시' COLLATE 'utf8_general_ci',
    `21H` VARCHAR(45) NOT NULL COMMENT '21~22시' COLLATE 'utf8_general_ci',
    `22H` VARCHAR(45) NOT NULL COMMENT '22~23시' COLLATE 'utf8_general_ci',
    `23H` VARCHAR(45) NOT NULL COMMENT '23~24시' COLLATE 'utf8_general_ci',
    PRIMARY KEY (`department_name`, `week`) USING BTREE
    )
    COLLATE='utf8_general_ci'
    ENGINE=MyISAM
    ;"""
    cursor.execute(sql)
    return

def dbinsert(dbcon,data,time_threshold_check):
    
    conn = pymysql.connect(host = dbcon['host'], port = dbcon['port'], user = dbcon['user'], password = dbcon['pwd'], db = dbcon['db'])
    cursor = conn.cursor()
    if time_threshold_check == 0:
        threshold = "1사분위수"
    elif time_threshold_check == 1:
        threshold = "평균"
    elif time_threshold_check == 2:
        threshold = "사용자 정의"
    for temp in data:
        log = temp[0]
        trid = log['was_trid']
        sdate = log['str_time_ui'].split(" ")[0].replace("-","")
        str_time_ui = log['str_time_ui'] 
        end_time_ui = log['end_time_ui']
        app_nm = log['app_nm']
        user_id = log['user_id']
        user_name = log['user_name']
        jdata = str(log).replace("'",'"')
        violate = "S"
        comment = "통계분석"
        report_type = "statistics"
        ttype = "통계분석"
        inspect_detail = "RULE"
        rule_total = str(temp[1]).replace("'",'"')
        change_dt = str(pd.Timestamp(datetime.datetime.now()))
        sql = "INSERT INTO N_VIOLATEML VALUES ('" + trid + \
         "','" +  sdate + "','" + str_time_ui + "','" + end_time_ui + "','" + \
         app_nm + "','" + user_id + "','" + user_name + "','" + jdata + "','" + \
         violate + "','" + comment + "','" + report_type + "','" + ttype + "','" + \
         inspect_detail +"','" + change_dt +"','" + threshold + "','" + rule_total + "')" + \
        "ON DUPLICATE KEY UPDATE rule_total = "+"'" + rule_total+"'"
        cursor.execute(sql)

    return

def usertableinsert(dbcon,table):
    
    conn = pymysql.connect(host = dbcon['host'], port = dbcon['port'], user = dbcon['user'], password = dbcon['pwd'], db = dbcon['db'])
    cursor = conn.cursor()
    
    for key in table.keys():
        temp = table[key]
        for i in range(2):
            if(i == 0):
                week = "평일"
            else:
                week = "주말"
            sql = "REPLACE INTO N_VIOLATEUserTable VALUES ('" + key + \
            "','" + temp[2] + "','" + week + "','" + \
            str(temp[i][0]) + "','" + str(temp[i][1]) + "','" + str(temp[i][2]) + "','" + str(temp[i][3]) + "','" + \
            str(temp[i][4]) + "','" + str(temp[i][5]) + "','" + str(temp[i][6]) + "','" + str(temp[i][7]) + "','" + \
            str(temp[i][8]) + "','" + str(temp[i][9]) + "','" + str(temp[i][10]) + "','" + str(temp[i][11]) + "','" + \
            str(temp[i][12]) + "','" + str(temp[i][13]) + "','" + str(temp[i][14]) + "','" + str(temp[i][15]) + "','" + \
            str(temp[i][16]) + "','" + str(temp[i][17]) + "','" + str(temp[i][18]) + "','" + str(temp[i][19]) + "','" + \
            str(temp[i][20]) + "','" + str(temp[i][21]) + "','" + str(temp[i][22]) + "','" + str(temp[i][23]) + "')"
            cursor.execute(sql)
    return

def departmenttableinsert(dbcon,table):
    
    conn = pymysql.connect(host = dbcon['host'], port = dbcon['port'], user = dbcon['user'], password = dbcon['pwd'], db = dbcon['db'])
    cursor = conn.cursor()
    
    for key in table.keys():
        temp = table[key]
        for i in range(2):
            if(i == 0):
                week = "평일"
            else:
                week = "주말"
            sql = "REPLACE INTO N_VIOLATEDepartmentTable VALUES ('" + key + \
            "','" + week + "','" + \
            str(temp[i][0]) + "','" + str(temp[i][1]) + "','" + str(temp[i][2]) + "','" + str(temp[i][3]) + "','" + \
            str(temp[i][4]) + "','" + str(temp[i][5]) + "','" + str(temp[i][6]) + "','" + str(temp[i][7]) + "','" + \
            str(temp[i][8]) + "','" + str(temp[i][9]) + "','" + str(temp[i][10]) + "','" + str(temp[i][11]) + "','" + \
            str(temp[i][12]) + "','" + str(temp[i][13]) + "','" + str(temp[i][14]) + "','" + str(temp[i][15]) + "','" + \
            str(temp[i][16]) + "','" + str(temp[i][17]) + "','" + str(temp[i][18]) + "','" + str(temp[i][19]) + "','" + \
            str(temp[i][20]) + "','" + str(temp[i][21]) + "','" + str(temp[i][22]) + "','" + str(temp[i][23]) + "')"
            cursor.execute(sql)
    return


def postitiontableinsert(dbcon,table):
    
    conn = pymysql.connect(host = dbcon['host'], port = dbcon['port'], user = dbcon['user'], password = dbcon['pwd'], db = dbcon['db'])
    cursor = conn.cursor()
    
    for key in table.keys():
        temp = table[key]
        for i in range(2):
            if(i == 0):
                week = "평일"
            else:
                week = "주말"
            sql = "REPLACE INTO N_VIOLATEPositionTable VALUES ('" + key + \
             "','" + week + "','" + \
            str(temp[i][0]) + "','" + str(temp[i][1]) + "','" + str(temp[i][2]) + "','" + str(temp[i][3]) + "','" + \
            str(temp[i][4]) + "','" + str(temp[i][5]) + "','" + str(temp[i][6]) + "','" + str(temp[i][7]) + "','" + \
            str(temp[i][8]) + "','" + str(temp[i][9]) + "','" + str(temp[i][10]) + "','" + str(temp[i][11]) + "','" + \
            str(temp[i][12]) + "','" + str(temp[i][13]) + "','" + str(temp[i][14]) + "','" + str(temp[i][15]) + "','" + \
            str(temp[i][16]) + "','" + str(temp[i][17]) + "','" + str(temp[i][18]) + "','" + str(temp[i][19]) + "','" + \
            str(temp[i][20]) + "','" + str(temp[i][21]) + "','" + str(temp[i][22]) + "','" + str(temp[i][23]) + "')"
            cursor.execute(sql)
    return


def usertableselect(dbcon):
    conn = pymysql.connect(host = dbcon['host'], port = dbcon['port'], user = dbcon['user'], password = dbcon['pwd'], db = dbcon['db'])
    cursor = conn.cursor()
    sql = "SELECT * FROM `N_VIOLATEUserTable`;"
    cursor.execute(sql)
    result = cursor.fetchall()
    return result
def postableselect(dbcon):
    conn = pymysql.connect(host = dbcon['host'], port = dbcon['port'], user = dbcon['user'], password = dbcon['pwd'], db = dbcon['db'])
    cursor = conn.cursor()
    sql = "SELECT * FROM `N_VIOLATEPositionTable`;"
    cursor.execute(sql)
    result = cursor.fetchall()
    return result
def depttableselect(dbcon):
    conn = pymysql.connect(host = dbcon['host'], port = dbcon['port'], user = dbcon['user'], password = dbcon['pwd'], db = dbcon['db'])
    cursor = conn.cursor()
    sql = "SELECT * FROM `N_VIOLATEDepartmentTable`;"
    cursor.execute(sql)
    result = cursor.fetchall()
    return result


# +
def tablereset(dbcon):
    conn = pymysql.connect(host = dbcon['host'], port = dbcon['port'], user = dbcon['user'], password = dbcon['pwd'], db = dbcon['db'])
    cursor = conn.cursor()
    
    sql = "Drop Table IF EXISTS N_VIOLATE_URL_WhiteList_Result"
    cursor.execute(sql)
    
    sql = "Drop Table IF EXISTS N_VIOLATE_URL_Pattern_WhiteList_Result"
    cursor.execute(sql)
    
    sql = "Drop Table IF EXISTS N_VIOLATE_SQL_Pattern_WhiteList_Result"
    cursor.execute(sql)
    
    sql = "Drop Table IF EXISTS N_VIOLATE_SQL_WhiteList_Result"
    cursor.execute(sql)
    
    return
    
    
