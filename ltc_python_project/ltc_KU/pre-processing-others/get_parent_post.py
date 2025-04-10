import requests
import time
import os
import datetime
import json
import glob
import copy
import csv
import os.path
import pandas as pd

output_json_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/parent_project/'
selected_project_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/selected_other_projects.csv'


def request(url):
    git_token = "YOUR_GITHUB_KEY"
    headers = {'Authorization': 'token {}'.format(git_token)}

    print("[REQUEST] {}".format(url))
    res = requests.get(url,
                       headers=headers)

    return res

def get_user_repo_parent():
    project_data = pd.read_csv(selected_project_file)
    repo_list = list(project_data['repo_url'])
    
    total = 0
    for repo_url_str in repo_list:
       
        print("[Working {} total = {}/{}]".format(repo_url_str,total,len(repo_list)))
        
        data = project_data[project_data['repo_url'] == repo_url_str]
        user_login_name = data.iloc[0]['user_login']
        name_key = str(user_login_name) + "_" + str(total)
        
        start_time = time.time()
        total = total + 1
        
        user_repo_list = list()
        url = repo_url_str
        res = request(url)
        time.sleep(1)
        try:
            respos = res.json()
            user_repo_list.append(respos)
        except Exception:
            print("Repository '{}' has a bad JSON format. Continuing from next repository".format(repo_url_str))
            continue
        
        i = {"gh_user": user_login_name,
                     "items": user_repo_list}
        with open("{}user_repos_{}.json".format(output_json_path,name_key), "w") as f:
            json.dump(i, f)
            
        print("Time <--- {:.5} minutes --->".format((time.time() - start_time)/60))
        
if __name__ == "__main__":
    start_time = time.time()
    get_user_repo_parent()
    print("Full Time <--- {:.5} minutes --->".format((time.time() - start_time)/60))
    print("Program finishes successfully!")