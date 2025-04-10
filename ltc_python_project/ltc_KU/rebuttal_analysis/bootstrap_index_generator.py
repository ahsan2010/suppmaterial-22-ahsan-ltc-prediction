#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:07:06 2025

@author: ahsan
"""
import pandas as pd
import numpy as np
import time

dev_month_ku_expertise_file_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/data/'


def bootstrap_index_generation():
    output_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/"
    year_value_list = [1,2,3]
    pd_list = []
    for year_value in year_value_list:
        
        path = '{}ltc_data_full_year_{}_30_jan_2025_rem_mult_proj.csv'.format(dev_month_ku_expertise_file_dir, year_value)
        X = pd.read_csv(path)
        indices = list(range(0,X.shape[0] ,1))
        sample_iterations = list(range(1,100 + 1,1))
        print("Working LTC Year {}".format(year_value))
        
        for i in sample_iterations:
            start_time = time.time()
            np.random.seed(i)
            train_index = list(np.random.choice(indices,len(indices)))
            test_index = np.setdiff1d(indices, train_index)
            
            d_train = pd.DataFrame(train_index, columns=['index_value'])
            d_train['boot_id'] = i
            d_train['ltc_year'] = year_value
            d_train['index_type'] = 'train_index'
            
            d_test = pd.DataFrame(test_index, columns=['index_value'])
            d_test['boot_id'] = i
            d_test['ltc_year'] = year_value
            d_test['index_type'] = 'test_index'
            
            d_merge = pd.concat([d_train, d_test], axis = 0)
            pd_list.append(d_merge)
            
        print("Finished")
        
        
    pd_final_combine_index = pd.concat(pd_list, axis = 0)
    pd_final_combine_index.to_csv("{}bootstrap_index_30_jan_2025.csv".format(output_path), index=False)

bootstrap_index_generation()