import json
import pandas as pd
import csv
import time



json_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/parent_project/'
selected_project_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/selected_other_projects.csv'
output_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/'


def convert_json_parent_repo_to_csv():
    
    project_data = pd.read_csv(selected_project_file)
    repo_list = list(project_data['repo_url'])
    
    total = 0
    user_project_information = []
    
    total = 0
    for repo_url_str in repo_list:
        data = project_data[project_data['repo_url'] == repo_url_str]
        user_login_name = data.iloc[0]['user_login']
        dev_name = data.iloc[0]['dev_name']
        name_key = str(user_login_name) + "_" + str(total)
        total = total + 1
        print("[Working {} total = {}/{}]".format(repo_url_str,total,len(repo_list)))
        
        json_file = "{}user_repos_{}.json".format(json_path,name_key)
        
        try:
            with open(json_file, encoding="utf8") as jsonfile:
                json_data = json.load(jsonfile)
                item_num = 0
                parent_url = repo_url_str
                parent_repo = parent_url[len("https://api.github.com/repos/"):]
                is_fork = False
                for item in json_data['items']:
                    is_fork = item['fork']
                    if is_fork:
                        parent_url = item['parent']['url']
                        parent_repo = parent_url[len("https://api.github.com/repos/"):]
                        print(parent_url)
                temp = [user_login_name,dev_name, repo_url_str, is_fork, parent_url, parent_repo]
                user_project_information.append(temp)
        
        except Exception as e:
            print('Problem in json file')
            print(str(e))
            continue;
                
    pr_column_list = ['user_login', 'dev_name','repo_url', 'is_fork', 'parent_url', 'parent_repo']
    output_file = '{}{}.csv'.format(output_path, "select_parent_project")
    pd_data_frame = pd.DataFrame(user_project_information)
    pd_data_frame.columns = pr_column_list
    pd_data_frame.to_csv(output_file, index = False)
    
if __name__ == "__main__":
    start_time = time.time()
    convert_json_parent_repo_to_csv()
    print("Time <--- {:.5} minutes --->".format((time.time() - start_time)/60))
    print('Successfully completed')