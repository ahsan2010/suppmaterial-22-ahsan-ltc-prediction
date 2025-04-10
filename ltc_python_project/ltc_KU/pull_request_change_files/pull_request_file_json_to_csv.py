
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 10:45:59 2021

@author: ahsan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 09:29:56 2021

@author: ahsan
"""

import json
import pandas as pd
import csv

#project_list = ['apache_lucene', 'apache_wicket', 'apache_hbase', 'apache_groovy', 'apache_hive', 'apache_activemq',
#'apache_storm','apache_stratos', 'elastic_elasticsearch', 'jruby_jruby', 'caskdata_cdap']

project_list = ['elastic_elasticsearch']

pr_column_list = [       'Proj_Name',
                         'Git_Repo_Name',
                         'Git_Proj_Name',
                         'PR_Url',
                         'PR_Number',
                         'commit_id',
                         'change_file_name',
                         'additions',
                         'deletions',
                         'changes',
                         'change_status']

project_file_location = '/scratch/ahsan/Java_Exam_Work/Result/pull_request/studied_project.csv'    
json_pull_change_file_dir = '/scratch/ahsan/Java_Exam_Work/Result/pull_request_changed_files/'
json_to_csv_folder = '/scratch/ahsan/Java_Exam_Work/Result/pull_request_changed_files/pull_request_files_csv/'
pull_request_folder = '/scratch/ahsan/Java_Exam_Work/Result/pull_request/'

def convert_pull_request_files_json_to_csv(project):
    pull_request_file = '{}pr_reports_{}.csv'.format(pull_request_folder, project)
    change_file_list = []
    with open(pull_request_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            try:
                git_repo_name = row['Git_Repo_Name']
                git_proj_name = row['Git_Proj_Name']
                proj_name = row['Proj_Name']
                pr_url = row['PR_Url']
                pr_number = row['PR_Number']        
                
                json_output_dir = '{}{}/'.format(json_pull_change_file_dir, proj_name)
                pull_channge_json_file = "{}pull_request_files_{}_{}.json".format(json_output_dir,proj_name, pr_number)
                #print(reviewer_json_file)
                with open(pull_channge_json_file,encoding="utf8") as jsonfile:
                    json_data = json.load(jsonfile)
                    item_num = 0
                    for item in json_data['items']:
                        item_num = item_num + 1
                        
                        commit_sha = item['sha']
                        changed_file_name = item['filename']
                        total_addition = item['additions']
                        total_deletion = item['deletions']
                        total_changes = item['changes']
                        change_status = item['status']
                           
                        change_file_info = [
                               proj_name,
                               git_repo_name,
                               git_proj_name,
                               pr_url,
                               pr_number,
                               commit_sha,
                                changed_file_name,
                                total_addition,
                                total_deletion,
                                total_changes,
                                change_status
                                ]
                        
                        change_file_list.append(change_file_info)
            except Exception as e:
                print(str(e))
        
        print(f'change file list Length {len(change_file_list)}')
        output_file = '{}{}_files.csv'.format(json_to_csv_folder, proj_name)
        pd_data_frame = pd.DataFrame(change_file_list)
        if pd_data_frame.shape[0] == 0:
            pd_data_frame = pd.DataFrame(columns = pr_column_list)
        else:    
            pd_data_frame.columns = pr_column_list
        pd_data_frame.to_csv(output_file, index = False)


if __name__ == "__main__":
    for project in project_list:
        print(project)
        convert_pull_request_files_json_to_csv(project)
    print('Program finishes successfully')# -*- coding: utf-8 -*-

