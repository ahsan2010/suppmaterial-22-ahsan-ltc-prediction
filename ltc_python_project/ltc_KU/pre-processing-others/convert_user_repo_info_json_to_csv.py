import json
import pandas as pd
import csv
import time

user_repo_data_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/user_git_repos/'
uniq_user_list = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/uniq_dev_login.csv'
output_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/user_git_repos/json_to_csv/'


def convert_json_user_repo_to_csv():
    user_data = pd.read_csv(uniq_user_list)
    user_name_list = list(user_data['login_x'])
    col_names = ['user_login', 'full_repo_name',
                 'repo_url', 'is_fork', 'created_at',
                 'updated_at', 'star_count', 'watcher_count',
                 'language', 'fork_count', 'is_archived',
                 'default_branch']
    total = 0
    user_project_information = []
    for user_login in user_name_list:
        start_time = time.time()
        total = total + 1
        print("[Working {} total = {}/{}]".format(user_login,total,len(user_name_list)))
        
        json_file = "{}user_repos_{}.json".format(user_repo_data_path,user_login)
        
        try:
            with open(json_file, encoding="utf8") as jsonfile:
                json_data = json.load(jsonfile)
                item_num = 0
                for item in json_data['items'][item_num]:
                    full_repo_name = item['full_name']
                    repo_url = item['url']
                    is_fork = item['fork']
                    created_at = item['created_at']
                    updated_at = item['updated_at']
                    star_count = item['stargazers_count']
                    watcher_count = item['watchers_count']
                    language = item['language']
                    fork_count = item['forks_count']
                    is_archived = item['archived']
                    default_branch = item['default_branch']
                    
                    temp = [user_login, full_repo_name, repo_url, is_fork, created_at, updated_at,
                            star_count, watcher_count, language, fork_count, is_archived,
                            default_branch]
                    user_project_information.append(temp)
                    #contributor_list.append([user_id,login_name,dev_name,location])
                    
                    
        except Exception as e:
            print('Problem in json file')
            print(str(e))
            continue;
        
        print("Time <--- {:.5} minutes --->".format((time.time() - start_time)/60))
        
    pr_column_list = col_names
    output_file = '{}{}.csv'.format(output_path, "user_project_list")
    pd_data_frame = pd.DataFrame(user_project_information)
    pd_data_frame.columns = pr_column_list
    pd_data_frame.to_csv(output_file, index = False)
    
convert_json_user_repo_to_csv()