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

def get_list_files(file_str):
    file_list = []
    for x in file_str.split(']-'):
        x = x.replace(',','').strip()
        file_list.append(x[0:x.find('[')])
    return file_list


def merge_github_project_full_data():
    ROOT = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2021/ahsan_gustavo/dev_knowledge/'
    
   
    commit_updated_result_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/commit_file_change_other_projects/'
    commit_meta_data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/commit_data_other_projects/'
    commit_merge_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/commit_merge_data_other_projects/'
    
    project_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/select_parent_project.csv"
    pd_proj_data = pd.read_csv(project_path)
    pd_proj_data['parent_html_repo'] = "https://github.com/" + pd_proj_data['parent_repo']
    pd_proj_data['repository']= [url_project_name(x) for x in pd_proj_data['parent_html_repo']]
    unique_parent_proj = list(pd_proj_data['repository'].unique())
    proj_list = unique_parent_proj
    
    total = 0
    
    
    column_list= [
            'project_name',
            'commit_id',
            'commit_msg',
            'commit_author_name',
            'commit_committer_name',
            'commit_author_email',
            'commit_committer_email',
            'commit_is_merge',
            'commit_author_date',
            'commit_committer_date',
            'is_commit_merge',
            'has_java_file',
            'no_modified_java_file',
            'file_change_list']
    
    project_stat = []
    
    for proj in proj_list:
        total = total + 1
       
        commit_file_change_path = '{}{}_file_change.csv'.format(commit_updated_result_dir, proj)
        commit_meta_file_path = '{}{}-basic_commit_history.csv'.format(commit_meta_data_dir, proj)
            
        if (os.path.exists(commit_file_change_path) & os.path.exists(commit_meta_file_path)):
            
            try:
                print("Working {}".format(proj))
                pd_data_commit_change = pd.read_csv(commit_file_change_path)
                #print(pd_data_commit_change.shape)
                pd_data_commit_meta = pd.read_csv(commit_meta_file_path)
                #print(pd_data_commit_meta.shape)
        
                project_commit_data = pd.merge(pd_data_commit_meta, pd_data_commit_change, on = 'commit_id')
                project_commit_data['file_change_list'] = [get_list_files(x) for x in project_commit_data['file_info_string']]
                project_commit_data['project_name'] = proj
                project_commit_data[column_list].to_csv(os.path.join(commit_merge_dir,'{}_full_commit_data.csv'.format(proj)), index = False)
            
                project_stat.append([proj, project_commit_data.shape[0]])
            
            except Exception as e:
                print(str(e))
        else:
            print('Project: {} Change File: {}, Meta File: {}'.format(proj, commit_file_change_path, commit_meta_file_path))
    
        print('Finish project {}'.format(total))
    
    project_stat_pd = pd.DataFrame(project_stat)
    project_stat_pd.columns = ['ProjName', "Rows"]
    project_stat_pd.to_csv("/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/other_project_rows.csv", index = False)

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
    output_file_loc = '{}/xin_xia_paper_data/commit_file_change_other_projects/{}_file_change.csv'.format(ROOT,proj_name)
    
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
       
class GitProjectInfoExtractor:
    
    @staticmethod
    def extract_commit_info(repo_loc, proj_name):
        ROOT = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/'
        output_file_location = ('{}{}{}-basic_commit_history.csv').format(ROOT,'xin_xia_paper_data/commit_data_other_projects/',proj_name)
        commit_list = []
        column_list = ['commit_id',
                   'commit_msg',
                   'commit_author_name',
                   'commit_committer_name',
                   'commit_author_email',
                   'commit_committer_email',
                   'commit_is_merge',
                   'commit_author_date',
                   'commit_committer_date']
        
        git_repo = Git(repo_loc)
        total_commit = git_repo.total_commits()
        commit_progress = 0
        for commit in Repository(repo_loc).traverse_commits():
            commit_hash = commit.hash
            commit_msg = commit.msg
            commit_author_name = commit.author.name
            commit_committer_name = commit.committer.name
            commit_author_email = commit.author.email
            commit_committer_email = commit.committer.email
            '''The author is the person who originally wrote the code. The committer, on the other hand, is assumed to be the person 
            who committed the code on behalf of the original author.'''
            commit_is_merge = commit.merge
            commit_author_date = commit.author_date
            commit_committer_date = commit.committer_date
           
            commit_list.append([commit_hash,commit_msg,commit_author_name,commit_committer_name,commit_author_email,commit_committer_email,commit_is_merge,commit_author_date, commit_committer_date])
            
            commit_progress = commit_progress + 1
            if commit_progress % 100 == 0:
                print('[{}] Finish commits: {}/{}'.format(proj_name,commit_progress,total_commit))
                
        print('[{}] Finish commits: {}'.format(proj_name,commit_progress))
        
        pd_data_frame = pd.DataFrame(commit_list)
        pd_data_frame.columns = column_list
        pd_data_frame.to_csv(output_file_location, index = False)
                   
def collect_studied_project_basic_data():
    ob = GitProjectInfoExtractor()
    project_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/select_parent_project.csv"
    pd_proj_data = pd.read_csv(project_path)
    pd_proj_data['parent_html_repo'] = "https://github.com/" + pd_proj_data['parent_repo']
    pd_proj_data['repository']= [url_project_name(x) for x in pd_proj_data['parent_html_repo']]
    unique_parent_proj = list(pd_proj_data['repository'].unique())
    proj_list = unique_parent_proj
    total = 0
    for i in range(0,len(proj_list)):
        proj = proj_list[i]
        total = total + 1
        print("{} {}".format(total, proj))
        #repo_path = '/scratch/ahsan/Java_Exam_Work/GitReposistories/{}'.format(proj)
        repo_path = '/home/local/SAIL/ahsan/OTHER_LTC_PROJECTS/{}'.format(proj)
        
        try:
            ob.extract_commit_info(repo_path,proj) 
        except Exception as e:
            print('Problem in path {}'.format(repo_path))
            print(str(e))
          

def collect_commit_change_info_studied_projects():
    #project_path = '/scratch/ahsan/Java_Exam_Work/Data/Studied_Project_List_200.csv'
    project_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/select_parent_project.csv"
    pd_proj_data = pd.read_csv(project_path)
    pd_proj_data['parent_html_repo'] = "https://github.com/" + pd_proj_data['parent_repo']
    pd_proj_data['repository']= [url_project_name(x) for x in pd_proj_data['parent_html_repo']]
    unique_parent_proj = list(pd_proj_data['repository'].unique())
    proj_list = unique_parent_proj
    for i in range(3589,len(proj_list)):
        proj = proj_list[i]
        #repo_path = '/scratch/ahsan/Java_Exam_Work/GitReposistories/{}'.format(proj)
        repo_path = '/home/local/SAIL/ahsan/OTHER_LTC_PROJECTS/{}'.format(proj)
        print(repo_path)
        thread_impl_project_extractor(repo_path,proj)

def merge_parent_project_other_project():
    parent_proj_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/select_parent_project.csv"
    other_proj_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/selected_other_projects.csv"
    
    parent_pd = pd.read_csv(parent_proj_path)
    other_proj_pd = pd.read_csv(other_proj_path)
    
    merge_data = pd.merge(other_proj_pd, parent_pd, on = ["user_login","repo_url"])

    merge_data.to_csv("/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/mrege_parent_other_proj.csv", index = False)

def main():
    start_time = time.time()
        
    #collect_commit_change_info_studied_projects()
    
    #collect_studied_project_basic_data()
    
    #merge_github_project_full_data()
    
    merge_parent_project_other_project()
    #thread_impl_project_extractor(repo_path,'elastic_elasticsearch')
    print("Full Time <--- {:.5} minutes --->".format((time.time() - start_time)/60))
    print("Program finishes successfully!")

def test():
    project_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/data/select_parent_project.csv"
    pd_proj_data = pd.read_csv(project_path)
    pd_proj_data['parent_html_repo'] = "https://github.com/" + pd_proj_data['parent_repo']
    pd_proj_data['repository']= [url_project_name(x) for x in pd_proj_data['parent_html_repo']]
    unique_parent_proj = list(pd_proj_data['repository'].unique())
    proj_list = unique_parent_proj
    
    print(proj_list[3000])
    print(proj_list[3000 + 587])
    
    # Problem 3588

if __name__ == "__main__":
    main()
   #test()
   
