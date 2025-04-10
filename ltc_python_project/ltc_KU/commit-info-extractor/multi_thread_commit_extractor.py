# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
from pydriller import Git
from pydriller import Repository
from pydriller.metrics.process.contributors_count import ContributorsCount
import time
from multiprocessing import Pool
from multiprocessing import set_start_method
from multiprocessing import get_context
import os

ROOT = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/'
        
def url_project_name(url_str):
    str_x = url_str.split("/")
    x = "{}-{}".format(str_x[3], str_x[4])
    return x

def check_file_extension(file_name, file_suffix):
    return file_name.rfind(file_suffix) > 0

def run_info_extractor(th_index, commit_hash_list,repo_loc,git_repo, proj_name, total_commits):
    commit_id = commit_hash_list[th_index]
    commit = git_repo.get_commit(commit_id)
    
    commit_hash = commit.hash
    commit_is_merge = commit.merge
    commit_modified_file_list = commit.modified_files
    
    file_change_list = []
    result = []
    
    for f in commit_modified_file_list:
        file_name = f.filename
        new_path = f.new_path
        line_added = f.added_lines
        line_removed = f.deleted_lines
        if new_path == None:
            continue
        is_java = check_file_extension(file_name,'.java')
        if is_java:
            file_string = '{}/{}[+{},-{}]'.format(new_path,file_name,line_added,line_removed)
            file_change_list.append(file_string)
    
    if len(file_change_list) > 0:
        file_change_string = '-'.join(file_change_list)
        result.append([proj_name, 
                       commit_hash, 
                       commit_is_merge, 
                       1, 
                       len(file_change_list), 
                       file_change_string])
    else:
        result.append([proj_name, 
                       commit_hash, 
                       commit_is_merge, 
                       0, 
                       0,
                       " "])

    result_pd = pd.DataFrame(result)
    
    print("[Finish] {}/{} commit index.".format(th_index, total_commits))
       
    return result_pd
    
def multi_run_wrapper(args):
    return run_info_extractor(args[0],args[1],args[2],args[3], args[4], args[5]) 

    
def thread_impl_project_extractor(repo_loc, proj_name):
    output_file_loc = '{}/xin_xia_paper_data/commit_file_change/{}_file_change.csv'.format(ROOT,proj_name)
    
    if os.path.exists(repo_loc) == False:
        return
    
    git_repo = Git(repo_loc)
    total_commit = git_repo.total_commits()
    commit_hash_list = []
    commit_progress = 0
    for commit in Repository(repo_loc).traverse_commits():
        commit_hash_list.append(commit.hash)
    total_commit = len(commit_hash_list)
    print('Total commits {}'.format(total_commit))

    
    with get_context("spawn").Pool(5) as p:
        df = pd.concat(p.map(multi_run_wrapper,[[x,commit_hash_list,repo_loc,git_repo,proj_name,total_commit] for x in range (total_commit)]))
        p.close()
        p.join()
        
    
    print('[Complete] {}'.format(proj_name))

    print("Start writing the result to file....")
    df.columns = ['project_name',
                  'commit_id',
                  'is_commit_merge',
                  'has_java_file',
                  'no_modified_java_file',
                  'file_info_string']
    df.to_csv(output_file_loc, index = False)
    print('[{}] Finish commits: {}/{}'.format(proj_name,commit_progress,total_commit))
                

def collect_commit_change_info_studied_projects():
    #project_path = '/scratch/ahsan/Java_Exam_Work/Data/Studied_Project_List_200.csv'
    project_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'
    pd_proj_data = pd.read_csv(project_path)
    pd_proj_data['repository']= [url_project_name(x) for x in pd_proj_data['html_url']]
    proj_list = list(pd_proj_data['repository'])
    for i in range(20,len(proj_list)):
        proj = proj_list[i]
        #repo_path = '/scratch/ahsan/Java_Exam_Work/GitReposistories/{}'.format(proj)
        repo_path = '/home/local/SAIL/ahsan/XIN_REPOSITORY/{}'.format(proj)
        print(repo_path)
        thread_impl_project_extractor(repo_path,proj)

def main():
    start_time = time.time()
    collect_commit_change_info_studied_projects()
    #thread_impl_project_extractor(repo_path,'elastic_elasticsearch')
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Program finishes successfully!")

if __name__ == "__main__":
    main()
