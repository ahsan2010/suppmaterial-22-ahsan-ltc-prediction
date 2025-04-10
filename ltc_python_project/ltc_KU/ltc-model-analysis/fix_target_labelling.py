# -*- coding: utf-8 -*-

import pandas as pd
import time

updated_label_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/"

#old_data_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/"
#output_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/"

old_data_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/"
output_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/"


def fix_target_lagelling(year):
    old_data_file = f"{old_data_path}full_result_ltc_{year}.csv"
    updated_label_file = f"{updated_label_path}dev_label_sep_2023.csv"
    old_data = pd.read_csv(old_data_file)
    updated_label = pd.read_csv(updated_label_file)
    updated_label = updated_label.drop_duplicates(subset=['Project_Name','dev_name'])
    
    merged_data = pd.merge(old_data, updated_label, on = ['Project_Name', 'dev_name'], how = 'left')
    merged_data = merged_data[-merged_data[f'LTC_YEAR_{year}'].isnull()]
    
    if year == 1:
        merged_data['LTC_Developer_Cat_Year_One'] = merged_data[f'LTC_YEAR_{year}']
    if year == 2:
        merged_data['LTC_Developer_Cat_Year_Two'] = merged_data[f'LTC_YEAR_{year}']
    if year == 3:
        merged_data['LTC_Developer_Cat_Year_Three'] = merged_data[f'LTC_YEAR_{year}']
        
    
    #condition = merged_data['is_ltc'].astype(int) - merged_data['LTC_YEAR_1'].astype(int) != 0
    #merged_data.loc[condition]
    
    # drop last three columns
    merged_data = merged_data.iloc[:, :-4]
    merged_data.to_csv(f'{output_path}full_result_updated_label_ltc{year}.csv', index=False)
    
    print(f'Year {year}')
    print(old_data.shape)
    print(merged_data.shape)
    print("----------")
    
def main():
    start_time = time.time()
    fix_target_lagelling(1)
    fix_target_lagelling(2)
    fix_target_lagelling(3)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Program finishes successfully!")

if __name__ == "__main__":
    main()