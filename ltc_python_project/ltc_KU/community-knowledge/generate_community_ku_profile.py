# -*- coding: utf-8 -*-
import requests
import time
import os
import datetime
import json
import glob
import copy
import csv
import pandas as pd
import numpy as np


LTC_PATH_DIR = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/"

LTC_ONE_FILE = "full_result_updated_label_ltc1"
LTC_TWO_FILE = "full_result_updated_label_ltc2"
LTC_THREE_FILE = "full_result_updated_label_ltc3"

ltc_file_list = [LTC_ONE_FILE, LTC_TWO_FILE, LTC_THREE_FILE]


def ku_profile_generation(path):
    data = pd.read_csv(path)
    group_by_column = ['Project_Name','MainDev']
    summary_columns = list(data.columns)[3:]
    agg_dict = {}
    
    
    for col in summary_columns:
        #agg_dict[col] = np.median
        agg_dict[col] = np.sum
    
    df = data.groupby(group_by_column).agg(
            agg_dict
            ).reset_index()
    
    return df

def community_ku_analysis():
    github_string = "https://api.github.com/repos/"
    project_file_location = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'
    project_data = pd.read_csv(project_file_location)
    project_url_list = project_data['url'].to_list()
    
    total = 0
    dir_normalized = "/home/local/SAIL/ahsan/LTC_Project/community-ku-profile/withNormalized/"
    dir_count = "/home/local/SAIL/ahsan/LTC_Project/community-ku-profile/withoutNormalized/"
    
    df_norm_list = []
    df_count_list = []
    problem = 0
    for i in range(len(project_url_list)):
        project_url = project_url_list[i]
        total = total + 1
        project = project_url[len(github_string):]
        project = project.replace("/","-")
        
        print("Working [{} {}/{}]".format( project, i + 1, len(project_url_list)))
        
        
        try:
            norm_path = "{}{}_norm.csv".format(dir_normalized,project)
            count_path = "{}{}_count.csv".format(dir_count,project)
            
            df_norm = ku_profile_generation(norm_path)
            df_count = ku_profile_generation (count_path)
            df_norm_list.append(df_norm)
            df_count_list.append(df_count)
        except Exception as e:
            problem = problem + 1
            print("Problem {} : {}".format(project, e))

    print("Problem {}/{}".format(problem, len(project_url_list)))
    
    final_norm = pd.concat(df_norm_list)
    final_count = pd.concat(df_count_list)
    
    final_norm.to_csv("{}community_summary_ku_norm.csv".format(dir_normalized), index = False)
    final_norm.to_csv("{}community_summary_ku_count.csv".format(dir_count), index = False)

def community_ku_analysis_feature_generate():
    github_string = "https://api.github.com/repos/"
    project_file_location = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'
    project_data = pd.read_csv(project_file_location)
    project_url_list = project_data['url'].to_list()
    
    total = 0
    dir_normalized = "/home/local/SAIL/ahsan/LTC_Project/community-ku-profile/withNormalized/"
    dir_count = "/home/local/SAIL/ahsan/LTC_Project/community-ku-profile/withoutNormalized/"
    
    ku_feature_list = []
    
    problem = 0
    for i in range(len(project_url_list)):
        project_url = project_url_list[i]
        total = total + 1
        project = project_url[len(github_string):]
        project = project.replace("/","-")
        
        print("Working [{} {}/{}]".format( project, i + 1, len(project_url_list)))
        
        
        try:
            norm_path = "{}{}_norm.csv".format(dir_normalized,project)
            count_path = "{}{}_count.csv".format(dir_count,project)
            
            df_norm = ku_profile_generation(norm_path)
            df_count = ku_profile_generation (count_path)
            
            col = list(df_count)
            norm_feature_list = ['norm_' + x for x in col[2:]]
            norm_feature_list = col[0:2] + norm_feature_list
            df_norm.columns = norm_feature_list
            
            pd_ku_feature = pd.merge(df_count, df_norm, how = 'inner', on = col[0:2])

            
            ku_feature_list.append(pd_ku_feature)
            
        except Exception as e:
            problem = problem + 1
            print("Problem {} : {}".format(project, e))

    print("Problem {}/{}".format(problem, len(project_url_list)))
    
    final_pd = pd.concat(ku_feature_list)
    
    final_pd = final_pd.rename(columns={'MainDev':'dev_name'})
    final_pd = final_pd.drop_duplicates()
    
    #f = final_pd[column_list[0:2]].groupby(column_list[0:2]).size().reset_index().sort_values(0)
    
    for i in range(len(ltc_file_list)):
        ltc_file_path = "{}{}.csv".format(LTC_PATH_DIR,ltc_file_list[i])
        pd_ltc = pd.read_csv(ltc_file_path)
        column_list = list(pd_ltc.columns)
        target = column_list[32:35]
        #selected_columns = column_list[0:2] + column_list[4:32] + column_list [35:64] + column_list [66:129] + target
        selected_columns = column_list[0:3] + column_list [66:129] + target

        pd_selected_col = pd_ltc[selected_columns]
        pd_ku_feature_full = pd.merge(pd_selected_col, final_pd, how = 'left', on = column_list[0:2])
        pd_ku_feature_full = pd_ku_feature_full.fillna(0)
    
        pd_ku_feature_full.to_csv("{}community_ku_count_norm_label_ltc{}.csv".format("/home/local/SAIL/ahsan/LTC_Project/community-ku-profile/", i + 1), index = False)
     
    final_pd.to_csv("{}community_ku_count_norm.csv".format("/home/local/SAIL/ahsan/LTC_Project/community-ku-profile/"), index = False)
    

if __name__ == "__main__":
    start = time.time()
    community_ku_analysis_feature_generate()
    end = time.time()
    print(f"Runtime of the program is {end - start}")
    