# -*- coding: utf-8 -*-
import requests
import time
import os
import datetime
import json
import glob
import copy
import csv

'''def request(url):
    git_token = "c7df13288605b85568f1044c9e3896cd6ecfcca2"
    headers = {'Authorization': 'token {}'.format(git_token)}

    print("[REQUEST] {}".format(url))
    res = requests.get(url,
                       headers=headers)

    return res
'''
def request(url):
    git_token = "YOUR_GITHUB_KEY"
    headers = {'Authorization': 'token {}'.format(git_token)}

    print("[REQUEST] {}".format(url))
    res = requests.get(url,
                       headers=headers)

    return res

def get_gh_pull_request():
    project_file_location = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'
    json_output_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/pull-request/pr-data/'
    github_string = "https://api.github.com/repos/"
    total = 0
    with open(project_file_location) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            total = total + 1
            git_repo_name = row['url']
            if git_repo_name == "https://api.github.com/repos/elastic/elasticsearch":
                continue
            project_name = git_repo_name[len(github_string):]
            project_name = project_name.replace("/","-")
            
            if os.path.isfile("{}pr_{}.json".format(json_output_dir,project_name)):
                print("Pull Request from '{}' repository were already collected. Continuing from next repository".format(git_repo_name))
                continue

            print("[START] ", project_name)
            res = request("{}/pulls?state=all&per_page=100".format(git_repo_name))
            time.sleep(2)

            pull_request_list = list()

            try:
                respos = res.json()
                #pull_request_list.append(j)
                while 'next' in res.links.keys():
                    res = request(res.links['next']['url'])
                    time.sleep(2)
                    try:
                        j = res.json()
                        respos.extend(j)
                    except Exception:
                        print("Repository '{}' has a bad JSON format. Continuing from next repository".format(project_name))
                        continue
                pull_request_list.append(respos)
            except Exception:
                print("Repository '{}' has a bad JSON format. Continuing from next repository".format(project_name))
                continue
            
            i = {"project": project_name,
                 "items": pull_request_list}
            with open(json_output_dir + "pr_{}.json".format(project_name), "w") as f:
                json.dump(i, f)
                
            print("[END] ", project_name)

if __name__ == "__main__":
    get_gh_pull_request()
