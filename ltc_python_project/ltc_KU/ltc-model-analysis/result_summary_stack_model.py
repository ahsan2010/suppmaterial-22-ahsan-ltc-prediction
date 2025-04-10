#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:06:42 2023

@author: ahsan
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 08:46:54 2023

@author: ahsan
"""
# -*- coding: utf-8 -*-
import pandas as pd
import os.path
import numpy as np


#model_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/August_05_2022/'
#model_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/Result/'
#model_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/ku-different-dimension/"
model_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/stack-model/"

ltc_year_list = [1,2,3]
    
STACK_KU_ALL = "stack_ku_all_dim_stack"
STACK_KU_ALL_XIN = "stack_ku_all_dim_with_xin_stack"

STACK_KU_ALL_AUTO = "stack_ku_all_dim_auto_stack_auto"
STACK_KU_ALL_XIN_AUTO = "stack_ku_all_dim_with_xin_auto_stack_auto"


model_list = [STACK_KU_ALL, 
             STACK_KU_ALL_XIN]

model_list_auto = [STACK_KU_ALL_AUTO,
                   STACK_KU_ALL_XIN_AUTO]

def analyze_model_result():
    pd_list = []
    for model in model_list:
        for ltc in ltc_year_list:
            file_name = '{}ltc_{}_{}.csv'.format(model_path, ltc, model)
            pd_data = pd.read_csv(file_name)
            pd_list.append(pd_data)
    pd_model_result = pd.concat(pd_list)
    
    result_median = pd_model_result.groupby(['ltc_year','features','classifier'])['auc'].agg('median')
    pd_result_median = pd.DataFrame(result_median).reset_index()
    #pd_result_median['key'] = pd_result_median['ltc_year'].astype(str) + '-'  + pd_result_median['features'] + '-' + pd_result_median['classifier']
    temp = pd.pivot_table(pd_result_median, index = ['ltc_year','features'], columns = 'classifier', values = 'auc')
    pd_final_result = pd.DataFrame(temp).reset_index().rename_axis(None, axis=1)
    pd_final_result.to_csv('{}model_result_bootstrap_median.csv'.format(model_path,), index = False)
   

def analyze_model_result_Autospearman():
    pd_list = []
    for model in model_list_auto:
        for ltc in ltc_year_list:
            file_name = '{}ltc_{}_{}.csv'.format(model_path, ltc, model)
            pd_data = pd.read_csv(file_name)
            pd_list.append(pd_data)
    pd_model_result = pd.concat(pd_list)
    
    result_median = pd_model_result.groupby(['ltc_year','features','classifier'])['auc'].agg('median')
    pd_result_median = pd.DataFrame(result_median).reset_index()
    temp = pd.pivot_table(pd_result_median, index = ['ltc_year','features'], columns = 'classifier', values = 'auc')
    pd_final_result = pd.DataFrame(temp).reset_index().rename_axis(None, axis=1)
    pd_final_result.to_csv('{}model_result_bootstrap_median_auto.csv'.format(model_path,), index = False)
   
def main():
    analyze_model_result()
    analyze_model_result_Autospearman()
    print('Program finishes successfully')

if __name__ == "__main__":
    main()
