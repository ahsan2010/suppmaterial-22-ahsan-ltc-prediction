# -*- coding: utf-8 -*-
import pandas as pd
import os.path
import numpy as np


#model_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/August_05_2022/'
#model_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/Result/'
model_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/Result/"


ltc_year_list = [1,2,3]
    
model_list = ['XIN_FEATURES',
              'XIN_FEATURES_KU_COUNT_NORM',
              'XIN_FEATURES_KU_COUNT',
              'XIN_FEATURES_KU_NORM',
              'KU_COUNT_NORM',
              'XIN_SIMPLIFIED',
              'XIN_SIMPLIFIED_KU_COUNT',
              'XIN_SIMPLIFIED_KU_NORM',
              'XIN_SIMPLIFIED_KU_COUNT_NORM',
              'KU_COUNT',
              'KU_NORM']

dictionary_model_file = {
                    'XIN_FEATURES'                  : 'xin_features_boostrap',
                    'XIN_FEATURES_KU_COUNT_NORM'    : 'ku_full_xin_features_boostrap',
                    'XIN_FEATURES_KU_COUNT'         : 'ku_count_xin_features_boostrap',
                    'XIN_FEATURES_KU_NORM'          : 'ku_norm_xin_features_boostrap',
                    'KU_COUNT_NORM'                 : 'ku_full_boostrap',
                    'XIN_SIMPLIFIED'                : 'xin_simp_boostrap',
                    'XIN_SIMPLIFIED_KU_COUNT'       : 'ku_count_xin_simp_boostrap',
                    'XIN_SIMPLIFIED_KU_NORM'        : 'ku_norm_xin_simp_boostrap',
                    'XIN_SIMPLIFIED_KU_COUNT_NORM'  : 'ku_full_xin_simp_boostrap',
                    'KU_COUNT'                      : 'ku_count_boostrap',
                    'KU_NORM'                       : 'ku_norm_boostrap'
                    }


def analyze_model_result():
    pd_list = []
    for model in model_list:
        for ltc in ltc_year_list:
            file_name = '{}ltc_{}_{}.csv'.format(model_path, ltc, dictionary_model_file[model])
            pd_data = pd.read_csv(file_name)
            pd_list.append(pd_data)
    pd_model_result = pd.concat(pd_list)
    
    result_median = pd_model_result.groupby(['ltc_year','features','classifier'])['auc'].agg('median')
    pd_result_median = pd.DataFrame(result_median).reset_index()
    #pd_result_median['key'] = pd_result_median['ltc_year'].astype(str) + '-'  + pd_result_median['features'] + '-' + pd_result_median['classifier']
    temp = pd.pivot_table(pd_result_median, index = ['ltc_year','features'], columns = 'classifier', values = 'auc')
    pd_final_result = pd.DataFrame(temp).reset_index().rename_axis(None, axis=1)
    pd_final_result.to_csv('{}model_result_bootstrap_median.csv'.format(model_path,), index = False)
    
def main():
    analyze_model_result()
    print('Program finishes successfully')

if __name__ == "__main__":
    main()
