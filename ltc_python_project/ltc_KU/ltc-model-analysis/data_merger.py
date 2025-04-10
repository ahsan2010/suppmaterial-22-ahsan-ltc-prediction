# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from functools import reduce
import os
import time


#QuantumBadger_RedReader_file_change
#h2oai_h2o-3-basic_commit_history

def get_list_files(file_str):
    file_list = []
    for x in file_str.split(']-'):
        x = x.replace(',','').strip()
        file_list.append(x[0:x.find('[')])
    return file_list

def url_project_name(url_str):
    str_x = url_str.split("/")
    x = "{}-{}".format(str_x[3], str_x[4])
    return x

def merge_github_project_full_data():
    ROOT = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2021/ahsan_gustavo/dev_knowledge/'
    
    #commit_updated_result_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2021/ahsan_gustavo/dev_knowledge/feasilibility_result/commit_results_updated/'
    #commit_meta_data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2021/ahsan_gustavo/dev_knowledge/feasilibility_result/commit_meta_data/'
    #commit_meta_data_dir = os.path.join(ROOT,'feasilibility_result','commit_merge_data')
    
    commit_updated_result_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/commit_file_change/'
    commit_meta_data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/commit_data/'
    commit_merge_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/commit_merge_data/'
    
    project_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'
    pd_proj_data = pd.read_csv(project_path)
    pd_proj_data['repository']= [url_project_name(x) for x in pd_proj_data['html_url']]
    proj_list = list(pd_proj_data['repository'])
    total = 0
    
    
    column_list= [
            'project_name',
            'commit_id',
            'commit_msg',
            'commit_author_name',
            'commit_committer_name',
            'commit_author_email',
            'commit_committer_email',
            'commit_is_merge',
            'commit_author_date',
            'commit_committer_date',
            'is_commit_merge',
            'has_java_file',
            'no_modified_java_file',
            'file_change_list']
    
    for proj in proj_list:
        total = total + 1
        
        commit_file_change_path = '{}{}_file_change.csv'.format(commit_updated_result_dir, proj)
        commit_meta_file_path = '{}{}-basic_commit_history.csv'.format(commit_meta_data_dir, proj)
        
        if (os.path.exists(commit_file_change_path) & os.path.exists(commit_meta_file_path)):
            print(proj)
            pd_data_commit_change = pd.read_csv(commit_file_change_path)
            print(pd_data_commit_change.shape)
            pd_data_commit_meta = pd.read_csv(commit_meta_file_path)
            print(pd_data_commit_meta.shape)
    
            project_commit_data = pd.merge(pd_data_commit_meta, pd_data_commit_change, on = 'commit_id')
            project_commit_data['file_change_list'] = [get_list_files(x) for x in project_commit_data['file_info_string']]
            project_commit_data['project_name'] = proj
            project_commit_data[column_list].to_csv(os.path.join(commit_merge_dir,'{}_full_commit_data.csv'.format(proj)), index = False)

        else:
            print('Project: {} Change File: {}, Meta File: {}'.format(proj, commit_file_change_path, commit_meta_file_path))
        
        
        print('Finish project {}'.format(total))
        


def main():
    start_time = time.time()
    merge_github_project_full_data()
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Program finishes successfully!")

if __name__ == "__main__":
    main()