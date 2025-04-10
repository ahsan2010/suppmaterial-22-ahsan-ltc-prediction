# -*- coding: utf-8 -*-

import json
import pandas as pd
import csv


def convert_json_git_issue_to_csv(json_pr_file, pr_column_list, csv_pr_file, proj_name, git_repo_name):
    with open(json_pr_file,encoding="utf8") as jsonfile:
        json_data = json.load(jsonfile)
        item_num = 0
        pr_csv_list = []
        for item in json_data['items'][0]:
            item_num = item_num + 1
            pr_id = item['id']
            pr_url = item['url']
            pr_html_url = item['html_url']
            pr_title = item['title']
            pr_body = item['body']
            pr_state =  item['state']
            pr_created_at = item['created_at']
            pr_updated_at = item['updated_at']
            pr_closed_at = item['closed_at']
            pr_merged_at = item['merged_at']
            pr_user = item['user']['login']
            pr_number = item['number']
            pr_merge_commit = item['merge_commit_sha']
            
            pr_list = [proj_name,
                            git_repo_name,
                           pr_id,
                           pr_url,
                           pr_html_url,
                           pr_title,
                           pr_body,
                           pr_state,
                           pr_created_at,
                           pr_updated_at,
                           pr_closed_at,
                           pr_merged_at,
                           pr_user,
                           pr_number,
                           pr_merge_commit]
            
            pr_csv_list.append(pr_list)
        
        pd_data_frame = pd.DataFrame(pr_csv_list)
        pd_data_frame.columns = pr_column_list
        pd_data_frame.to_csv(csv_pr_file, index = False)
        
        return pd_data_frame

if __name__ == "__main__":
    
    pr_column_list = ['Proj_Name',
                         'Git_Repo_Name',
                         'PR_Id',
                         'PR_Url',
                         'PR_Html_Url',
                         'PR_Title',
                         'PR_Body',
                         'PR_State',
                         'PR_Created_At',
                         'PR_Updated_At',
                         'PR_Closed_At',
                         'PR_Merged_At',
                         'PR_Creator_Login',
                         'PR_Number',
                         'PR_Merge_Commit_SHA']

    pd_data_frame_list = []
    
    project_file_location = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'
    json_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/pull-request/pr-data/'
    csv_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/pull-request/pr-data/csv-pr-data/"
    github_string = "https://api.github.com/repos/"
    with open(project_file_location) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            git_repo_name = row['url']
            proj_name = git_repo_name[len(github_string):]
            proj_name = proj_name.replace("/","-")
            
            json_pr_file = json_dir + "pr_{}.json".format(proj_name)
            csv_pr_file =  csv_dir + "pr_{}.csv".format(proj_name)
            
            print('Working.. ' , proj_name)
            try:
                pd_data = convert_json_git_issue_to_csv(json_pr_file,pr_column_list,csv_pr_file,proj_name, git_repo_name)
                pd_data_frame_list.append(pd_data)
            except Exception as e:
                print('Failed to upload to ftp: '+ str(e) + ' ' + proj_name)
    
    final_pd = pd.concat(pd_data_frame_list)      
    final_pd.to_csv(json_dir + 'full_pull_request.csv', index = False)
    print('Program finishes successfully')