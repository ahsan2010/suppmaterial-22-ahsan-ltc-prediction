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

output_json_path = '/home/local/SAIL/ahsan/LTC_Project/Contributor_Information/'
full_contrib_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/Xin_year_1_java_with_gt_login_name.csv'


def request(url):
    
    git_token = "YOUR_GITHUB_KEY"
    headers = {'Authorization': 'token {}'.format(git_token)}

    print("[REQUEST] {}".format(url))
    res = requests.get(url,
                       headers=headers)

    return res

def get_contributor_email_from_json():
    with open(full_contrib_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        total = 0
        email_data = []
        for row in csv_reader:
            total = total + 1
            if total < 4523:
                continue
            repo_id = row['repo_id']
            user_id = row['user_id']
            login_name = row['login']
            contr_url = "{}{}".format("https://api.github.com/users/",login_name)
            
            if os.path.exists("{}contributor_{}.json".format(output_json_path,user_id)):
                print('Already crawled')
                continue
            
            print(contr_url)
            res = request(contr_url)
            time.sleep(1)
            contributor_list = list()
                   
            try:
                j = res.json()
                contributor_list.append(j)
                #contributor_list = contributor_list + j
            except Exception:
                print("Repository '{}' has a bad JSON format. Continuing from next repository".format(repo_id))
                continue
            
            i = {"gh_user": user_id,
                     "items": contributor_list}
            with open("{}contributor_{}.json".format(output_json_path,user_id), "w") as f:
                json.dump(i, f)
            
            print('Finish {}'.format(total))
        
        
        
if __name__ == "__main__":
    get_contributor_email_from_json()