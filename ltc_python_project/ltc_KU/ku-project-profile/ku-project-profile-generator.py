# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
import collections

ROOT = "/home/local/SAIL/ahsan/LTC_Project/project-ku-profile/"
input_file = ROOT + "project-ku-profile.csv"
output_file = ROOT + "proj-ku-dim.csv"


LTC_PATH_DIR = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/"

LTC_ONE_FILE = "full_result_updated_label_ltc1"
LTC_TWO_FILE = "full_result_updated_label_ltc2"
LTC_THREE_FILE = "full_result_updated_label_ltc3"

ltc_file_list = [LTC_ONE_FILE, LTC_TWO_FILE, LTC_THREE_FILE]


def generate_ku_proj_profile():
    pd_other_proj_ku = pd.read_csv(input_file)
    column_list = list(pd_other_proj_ku.columns)
    selected_columns = [column_list[0]] + [column_list[1]] + column_list[3:] 
    
    pd_other_proj_ku_selected = pd_other_proj_ku[selected_columns]
    
    group_by_column = ['dev_name']
    summary_columns = list(selected_columns)[2:]
    agg_dict = {}
    
    
    for col in summary_columns:
        agg_dict[col] = np.median
    
    df = pd_other_proj_ku_selected.groupby(group_by_column).agg(
            agg_dict
            ).reset_index()
    
    final_pd = df.drop_duplicates()
    
    for i in range(len(ltc_file_list)):
        ltc_file_path = "{}{}.csv".format(LTC_PATH_DIR,ltc_file_list[i])
        pd_ltc = pd.read_csv(ltc_file_path)
        column_list = list(pd_ltc.columns)
        target = column_list[32:35]
        #selected_columns = column_list[0:2] + column_list[4:32] + column_list [35:64] + column_list [66:129] + target
        selected_columns = column_list[0:3] + column_list [66:129] + target

        pd_selected_col = pd_ltc[selected_columns]
        
        #my_dev_merge_function_investigate(pd_selected_col, final_pd)
        
        pd_ku_feature_full = pd.merge(pd_selected_col, final_pd, how = 'left', on = column_list[1:2])
        
        nan_rows = pd_ku_feature_full.isna().any(axis=1)
        nan_data = pd_ku_feature_full[nan_rows.to_list()]
        #print(nan_data.shape[0])
        pd_ku_feature_full = pd_ku_feature_full.fillna(0)
    
        pd_ku_feature_full.to_csv("{}proj-ku-dim-ltc-{}.csv".format(ROOT, i + 1), index = False)
       

def main():
    start_time = time.time()
    
    generate_ku_proj_profile()
    
    print("Full Time <--- {:.5} minutes --->".format((time.time() - start_time)/60))
    print("Program finishes successfully!")
    
if __name__ == "__main__":
    main()
