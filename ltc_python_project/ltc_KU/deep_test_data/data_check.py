#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:29:40 2024

@author: ahsan
"""

import pandas as pd
import numpy as np
import random 

def test_data():
    
    year_value = 1
    xin_data_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/Xin_year_{}_java_with_gt_login_name.csv'.format(year_value)
    pd_xin_data = pd.read_csv(xin_data_path)
    
    output_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/'
    pd_full = pd.read_csv('{}full_result_ltc_{}.csv'.format(output_file, year_value))
    
    
    xin_data_not_in_pd_full = pd_xin_data[~pd_xin_data['user_id'].isin(pd_full['user_id'])]
    xin_data_not_in_pd_full = xin_data_not_in_pd_full.drop_duplicates(subset=['user_id'])
    random_dev_pick = 309
    
    sample_ind = random.sample(range(0,xin_data_not_in_pd_full.shape[0]), random_dev_pick)
    
    remain_devs = xin_data_not_in_pd_full.iloc[sample_ind]
    pd_full_xin = pd_xin_data[pd_xin_data['user_id'].isin(pd_full['user_id'])]
    pd_data = pd.concat([pd_full_xin,remain_devs])
    pd_data.to_csv('/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/fixed_dataset/data_set_{}.csv'.format(year_value),index=False)
    
def main():
    print('Finish Successfully')
    pass

if __name__ == "__main__":
    main()