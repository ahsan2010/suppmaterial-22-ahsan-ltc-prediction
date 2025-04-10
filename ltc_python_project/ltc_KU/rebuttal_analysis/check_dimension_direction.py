# -*- coding: utf-8 -*-
import pandas as pd
import time

ku_dimension_list = [
    'KU_DEV_EXP',
    'KU_CUR_PROJ',
    'KU_DEV_PREV_EXP',
    'KU_COLLAB_EXP',
    'KU_PREV_PROJ'
    ]

xin_dimension_list = [
    'BAO_DEV_PPROF',
    'BAO_REPO_PROF',
    'BAO_DEV_ACT',
    'BAO_REPO_ACT',
    'BAO_COLLAB_NET'
    ]

path_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/feature_importance_reb/'

def analyze_direction_each_dimension():
    pd_list = []
    for year_value in [1,2,3]:
        year_file = '{}year_{}.csv'.format(path_dir, year_value)
        pd_data = pd.read_csv(year_file)
        pd_data['ratio'] = [x/abs(y) for x,y in zip(pd_data['pos'],pd_data['neg'])]
        
        d = pd_data.groupby(['dim']).median().reset_index()
        d['year'] = year_value
        pd_list.append(d)
    pd_final = pd.concat(pd_list)


def main():
    
    start_time = time.time()
   
    print(" YEAR{:.4f} minutes ---".format((time.time() - start_time)/60.0))
   
    print("main function..")

if __name__ == "__main__":
    main()
    print("Program finishes successfully.")
