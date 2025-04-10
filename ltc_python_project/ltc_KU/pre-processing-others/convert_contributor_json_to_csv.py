# -*- coding: utf-8 -*-
import json
import pandas as pd
import csv

output_path = '/home/local/SAIL/ahsan/LTC_Project/Contributor_Info_CSV/'
contrib_json_path = '/home/local/SAIL/ahsan/LTC_Project/Contributor_Information/'
full_contrib_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/Xin_year_1_java_with_gt_login_name.csv'

user_taken = []

def convert_json_to_csv():
    print('Hello')
    contributor_list = []
    with open(full_contrib_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        total = 0
        email_data = []
        for row in csv_reader:
            total = total + 1
            repo_id = row['repo_id']
            user_id = row['user_id']
            login_name = row['login']
           
            print(user_id)
            if user_id in user_taken:
                continue;
            
            user_taken.append(user_id)
             
            json_file = "{}contributor_{}.json".format(contrib_json_path,user_id)
            
            try:
                with open(json_file, encoding="utf8") as jsonfile:
                    json_data = json.load(jsonfile)
                    item_num = 0
                    for item in json_data['items']:
                        login_name = item['login']
                        dev_name = item['name']
                        location = item['location']
                        contributor_list.append([user_id,login_name,dev_name,location])
                    
                    
            except Exception as e:
                print('Problem in json file')
                print(str(e))
                contributor_list.append([user_id,login_name,'',''])
                continue;
    pr_column_list = ['user_id','login','dev_name','location']
    output_file = '{}{}.csv'.format(output_path, "contributors")
    pd_data_frame = pd.DataFrame(contributor_list)
    pd_data_frame.columns = pr_column_list
    pd_data_frame.to_csv(output_file, index = False)
    
if __name__ == "__main__":
    convert_json_to_csv()
    print('Successfully completed')