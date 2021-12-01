# -*- coding: utf-8 -*-
import json
import pandas as pd
from collections import Counter, defaultdict
from elasticsearch import Elasticsearch
import datetime

class Common:

    # standard를 기준으로 해당하는 standard에 해당하는 info filtering
    # standard는 칼럼 이름으로 해줄 것.
    def data_filtering(self, column, standard, info, data):

        result = defaultdict(list)
        st = data[column]
        
        for i in range(len(st)):
            if standard == st[i]:
                for inf in info:
                    result[inf].append(data[inf][i])
            
        return result
    # df_ex = data_filtering('사용자 IP','210.119.148.36',['호출시간','URI'],data)
    # df_ex = data_filtering('직급명','연구원',['호출시간','URI'],data)
    # df_ex = data_filtering('부서명','대학교육혁신원',['호출시간','URI'],data)

    # URL 패턴 생성
    def pattern(self, column, standard, pattern_info, data, dedup_op, depth, filter_r, is_url):
        
        pattern = defaultdict(list)
        
        if not filter_r:
            print('NO DATA')
            return  
        
        time = filter_r[pattern_info[0]]
        url = filter_r[pattern_info[1]]

        # True면 중복제거O / False면 중복제거X
        if dedup_op:

            dedup_url = [url[0]]
            dedup_time = [time[0]]

            for i in range(1, len(url)):
                if url[i] != url[i-1]:
                    dedup_url.append(url[i])
                    dedup_time.append(time[i])

            url = dedup_url
            time = dedup_time
        
        for i in range(len(url)-depth+1):
            p = list()
            for k in range(depth):
                if is_url:
                    p.append(url[i+k].split('/')[1])
                else:
                    p.append(url[i+k])
            pattern[tuple(p)].append(time[i])

        return sorted(pattern.items(), key = lambda x : len(x[1]), reverse=True)
    # pattern_ex = pattern('사용자 IP','222.116.158.167',['호출시간','URI'], data, True, 2)



    def es_data_search(self, ip, port, index, doc_type):
        
        es = Elasticsearch(hosts=ip, port=port)

        resp = es.search(
            index = index,
            doc_type=doc_type,
            body = {"size": 1000,"query": {"match_all": {}}},
            scroll = '25m'
        )

        old_scroll_id = resp['_scroll_id']

        # 처음 요청에서 출력된 결과 저장
        result = []
        for doc in resp['hits']['hits']:
            result.append(doc['_source'])

        # 이어서 남은 Data Scroll
        while len(resp['hits']['hits']):
            resp = es.scroll(
                scroll_id = old_scroll_id,
                scroll = '25m'
            )
            old_scroll_id = resp['_scroll_id']
    
            for doc in resp['hits']['hits']:
                result.append(doc['_source'])
        
        return result
    # input_data = es_data_search('112.220.209.210', '13017', 'camo_pagent','al_was_acc_log')


    def es_data_search_date(self, ip, port, index, time_column, str_date, end_date):
    
        ip,port,index = ip,port,index
        
        es = Elasticsearch(hosts=ip, port=port)
    
        resp = es.search(
            index = index,
            body = {"size": 1000,"query": {
                "range": {
                    time_column:{
                        "gte":str_date,
                        "lte":end_date,
                    "format": "yyyy-MM-dd"}}}},
            scroll = '25m' 
        )
    
        old_scroll_id = resp['_scroll_id']
    
        # 처음 요청에서 출력된 결과 저장
        result = []
        for doc in resp['hits']['hits']:
            result.append(doc['_source'])
    
        # 이어서 남은 Data Scroll
        while len(resp['hits']['hits']):
            resp = es.scroll(
                scroll_id = old_scroll_id,
                scroll = '25m'
            )
            old_scroll_id = resp['_scroll_id']
    
            for doc in resp['hits']['hits']:
                result.append(doc['_source'])
        
        return result
    # input_data = es_data_search_date('112.220.209.210', '13017', 'camo_pagent','al_was_acc_log', '2021-03-10', '2021-03-12')


    def pattern_check(self, pattern, st, threshold, urls):

        split_urls = [url.split('/')[1] for url in urls]
        result = list()
        
        for second in range(len(urls)):
            check = True
            if st[second] in pattern.keys():
                for first in range(second-1,-1,-1):
                    if st[first] == st[second] and (split_urls[first],split_urls[second]) in pattern[st[first]].keys() and pattern[st[first]][(split_urls[first],split_urls[second])] >= threshold:
                        result.append(True)
                        check = False
                        break
                if check:
                    result.append(False)
            else:
                result.append(None)
        
        return result

        
    # starting URL 찾기
    def find_starting_url(self, column, standard, pattern_info, data, filter_r):

        date_result = filter_r[pattern_info[0]]
        url_result = filter_r[pattern_info[1]]

        date_dic = defaultdict(str)

        for i in range(len(date_result)):
            d = str(date_result[i]).split()[0]
            if d not in date_dic:
                date_dic[d] = url_result[i]
        
        try:
            return Counter(date_dic.values()).most_common(1)[0][0]
        except:
            return ''
    # fsu_ex = find_starting_url('사용자 IP','222.116.158.167',['호출시간','URI'], data)



    # Common Starting URL 찾기
    def find_common_starting_url(self, column, pattern_info, data):

        result = list()
        standard_list = list(set(data[column]))

        for i in range(len(standard_list)):
            print(str(i)+' / '+str(len(standard_list)))
            starting_url = self.find_starting_url(column, standard_list[i], pattern_info, data)
            result.append(starting_url)

        return Counter(result).most_common(1)[0][0].split('/')[1]
    # fcsu_ex = find_common_starting_url('사용자 IP',['호출시간','URI'], data)



    # 생성된 URL 패턴 중, Second URL 위치에 starting URL이 있는 패턴은 제외시키기
    def starting_url_filtering(self, starting_url, pattern_result):
        
        starting_url_filtered = list()
        
        for p in pattern_result:
            if p[0][1] != starting_url:
                starting_url_filtered.append(p)
        
        return sorted(starting_url_filtered, key = lambda x : len(x[1]), reverse=True)
    # suf_ex = starting_url_filtering(fsu_ex, pattern_ex)

    def input_es(self, result, ip, port, index_name):
        
        tmp = pd.DataFrame(result).to_json(orient='records')
        data_json = json.loads(tmp)

        es = Elasticsearch('http://'+ip+':'+port)
        date = datetime.datetime.now().strftime('%y-%m-%d')
        for doc in data_json:
            try:
                es.index(index = index_name+"_"+date, doc_type = '_doc', body = doc)
            except:
                print('ERROR')

    # input_es(result, '117.17.189.6', '4200', 'test')

    def input_db(self, test_data, rtime_num ,rt, ru, rs, engine):
        test_data = [log for log in test_data if log['dept_name'] != 'UNKNOWN']
        test_data = pd.DataFrame(test_data)
        test_data = test_data.sort_values(by = ['was_trid'])
        test_data = test_data.reset_index(drop=True)
        
        
        temp_result = pd.merge(ru, rs, how='left',on='trid')
        temp_result = temp_result.fillna("해당없음")
        
        result = pd.merge(rt, temp_result, how='left',on='trid')
        result = result.sort_values(by = ['trid'])
        result = result.reset_index(drop=True)
        log = []
        admin = pd.read_sql_table(table_name="N_VIOLATE_ADMIN_DISCRIMINANT", con=engine)
        for i in range(len(test_data)):
            temp = []
            rule_temp = {"RT01" : "해당없음", "RT02" : "해당없음", "RU01" : "해당없음", "RU02" : "해당없음", "RS01" : "해당없음",
                  "RS02" : "해당없음"}
            temp_json = test_data.loc[i].to_dict()
            temp_result = list(result.loc[i])
            idx = 1
            for key in rule_temp.keys():
                
                if(rtime_num == 1):
                    rule = "RT01"
                    if(key == 'RT02'):
                        continue
                if(rtime_num == 2):
                    rule = "RT02"
                    if(key == 'RT01'):
                        continue
                rule_temp[key] = temp_result[idx]
                idx += 1
            temp.append(temp_json) 
            user_data = admin[admin['user_id'] == temp_json['user_id']]
            user_data = user_data.reset_index(drop=True)
            if(len(user_data) != 0):
                a = user_data.loc[0]['hour'].split(",")
                hour_list = [int (i) for i in a]
                time = temp_json['str_time_ui'].split(" ")[1]
                hour = int(time.split(":")[0])
                if(hour in hour_list):
                    rule_temp[rule] = "정상"
            temp.append(rule_temp)
            log.append(temp)

        return log

    
