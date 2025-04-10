# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

initial_java_project_users = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/user_git_repos/json_to_csv/user_project_list.csv'
ltc_data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/'
output_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/selected_other_projects.csv'


def load_ltc_data():
    pd_list = []
    for year in [1,2,3]:
        file_path = "{}full_result_updated_label_ltc{}.csv".format(ltc_data_dir, year)
        data = pd.read_csv(file_path)
        data["ltc_year"] = year
        pd_list.append(data)
    pd_full = pd.concat(pd_list)
    return pd_full

def select_other_project_users():
    pd_ltc = load_ltc_data()
    pd_initial_project = pd.read_csv(initial_java_project_users)
    login_commit_data = pd_ltc[["login_x","FirstJavaCommit", "dev_name"]]
    login_commit_data.columns = ['user_login','first_java_commit', 'dev_name']
    
    pd_initial_project = pd_initial_project.drop_duplicates()
    login_commit_data = login_commit_data.drop_duplicates()
    login_commit_data= login_commit_data.drop_duplicates(subset='user_login', keep="first")
    
    merged_data = pd.merge(pd_initial_project, login_commit_data, on = ['user_login'], how = 'left')
    merged_data['created_at'] = pd.to_datetime(merged_data['created_at'], utc=True)
    merged_data['first_java_commit'] = pd.to_datetime(merged_data['first_java_commit'], utc=True)
    
    merged_data['project_before_first_commit'] = merged_data['first_java_commit'] > merged_data['created_at']

    merged_data['diff_years'] = (merged_data['first_java_commit'] - merged_data['created_at']) / np.timedelta64(1, 'Y')
    

    selected_rows = merged_data[(merged_data['language'] == "Java") & (merged_data['diff_years'] >= 0) & (merged_data['diff_years'] <=2)]
    selected_rows.nunique()
    selected_rows[['user_login','dev_name','repo_url','created_at', 'first_java_commit', 'project_before_first_commit', 'diff_years']].to_csv(output_file, index = False)

    login_commit_data[login_commit_data['user_login'] == 'lzyzsd']

if __name__ == "__main__":
    select_other_project_users()
    print('Successfully completed')