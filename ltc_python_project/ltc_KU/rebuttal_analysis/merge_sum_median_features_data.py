#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:50:20 2025

@author: ahsan
"""
import pandas as pd

data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/data/'

ltc_year_list = [1,2,3]

for year_value in ltc_year_list:
    sum_data = "{}ltc_data_full_year_{}_30_jan_2025.csv".format(data_dir, year_value)
    median_data = "{}ltc_data_full_year_{}.csv".format(data_dir, year_value)
    
    pd_median_data = pd.read_csv(median_data)
    pd_sum_data = pd.read_csv(sum_data)
    
    
    median_data_col = list(pd_median_data.columns)
    
    # median_data_col.index('proj_dim_data_type') 94
    # median_data_col.index('LTC_Developer_Cat_Year_One') 210
    
    
    non_change_columns = median_data_col[0:94]
    target_columns = median_data_col[210:]
    changed_columns = median_data_col[94:210]
    
    sum_column_name = [x + "-sum" for x in changed_columns]
    median_column_name = [x + "-med" for x in changed_columns]
    
    pd_full = pd.concat([pd_median_data[non_change_columns],
               pd_median_data[changed_columns],
               pd_sum_data[changed_columns],
               pd_median_data[target_columns]], axis = 1)
    
    pd_full.columns = non_change_columns + median_column_name + \
                            sum_column_name + target_columns
    
    
    # Removing multiple project developers
    
    duplicates = pd_full[pd_full.duplicated(subset=['dev_name'], keep=False)]
    duplicates = duplicates.drop_duplicates(subset=['Project_Name','dev_name'], keep=False)
    f = '{}duplicate_dev_project.csv'.format(dir)
    duplicates.to_csv(f,index=False)
    user_involve_multiple_project = list(set(duplicates['dev_name']))
    frequency = duplicates['dev_name'].value_counts()
    
    pd_full_removing_multiple_project = pd_full[~pd_full['dev_name'].isin(user_involve_multiple_project)]
    pd_full_removing_multiple_project.to_csv("{}ltc_data_full_year_{}_30_jan_2025_rem_mult_proj.csv".format(data_dir, year_value))
    