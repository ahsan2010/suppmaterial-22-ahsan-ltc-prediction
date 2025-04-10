# -*- coding: utf-8 -*-
import pandas as pd
import csv



def mapping_user(year_value):
    xin_year_ltc_file_name = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/Xin_year_{}_java.csv".format(year_value)
    pd_year_ltc = pd.read_csv(xin_year_ltc_file_name)

    ghtorrent_user_file_name = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/users.csv"
    file = open(ghtorrent_user_file_name, mode='r', encoding='utf-8')
    csv_reader = csv.reader(file, doublequote=True, escapechar='\\',
                            quotechar="\"", delimiter=',',
                                   skipinitialspace=True)

    counter = 0
    rows = []   
    for row in csv_reader:
        counter = counter + 1
        if counter % 100000 == 0:
            print(counter)
        rows.append(row)
    
    pd_usr = pd.DataFrame(rows)
    pd_usr.columns = ['user_id', 'login', 'company', 'email', 'created_at', 'type', 'fake', 'deleted', 'long', 'lat', 'country_code', 'state', 'city']
    pd_usr['user_id'] = pd_usr['user_id'].astype(int)
    merged_pd = pd.merge(pd_year_ltc,pd_usr, on = 'user_id')
    
    merged_pd.to_csv('/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/Xin_year_{}_java_with_gt_login_name_II.csv'.format(year_value),index = False)
    

mapping_user(2)
mapping_user(3)