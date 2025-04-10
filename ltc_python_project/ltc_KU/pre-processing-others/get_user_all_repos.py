# -*- coding: utf-8 -*-
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

output_json_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/user_git_repos/'
uniq_user_list = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/uniq_dev_login.csv'


def request(url):
    git_token = "YOUR_GITHUB_KEY"
    headers = {'Authorization': 'token {}'.format(git_token)}

    print("[REQUEST] {}".format(url))
    res = requests.get(url,
                       headers=headers)

    return res

def get_user_repos():
    user_data = pd.read_csv(uniq_user_list)
    user_name_list = list(user_data['login_x'])
    per_page = 100
    total = 0
    for user_login in user_name_list:
        start_time = time.time()
        total = total + 1
        print("[Working {} total = {}/{}]".format(user_login,total,len(user_name_list)))
        user_repo_list = list()
        url = "https://api.github.com/users/{}/repos?per_page={}&page=1".format(user_login,per_page)
        res = request(url)
        time.sleep(1)
        try:
            respos = res.json()
            #user_repo_list.append(respos)
            while 'next' in res.links.keys():
                res = request(res.links['next']['url'])
                respos.extend(res.json())
            user_repo_list.append(respos)
        except Exception:
            print("Repository '{}' has a bad JSON format. Continuing from next repository".format(user_login))
            continue
        
        i = {"gh_user": user_login,
                     "items": user_repo_list}
        with open("{}user_repos_{}.json".format(output_json_path,user_login), "w") as f:
            json.dump(i, f)
            
        print("Time <--- {:.5} minutes --->".format((time.time() - start_time)/60))
        
if __name__ == "__main__":
    start_time = time.time()
    get_user_repos()
    print("Full Time <--- {:.5} minutes --->".format((time.time() - start_time)/60))
    print("Program finishes successfully!")