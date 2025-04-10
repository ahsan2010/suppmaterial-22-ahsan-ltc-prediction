# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
from pydriller import Git
from pydriller import Repository
from pydriller.metrics.process.contributors_count import ContributorsCount
import time
from multiprocessing import Pool



repo_path = '/scratch/ahsan/Java_Exam_Work/GitReposistories/elastic_elasticsearch'

def url_project_name(url_str):
    str_x = url_str.split("/")
    x = "{}-{}".format(str_x[3], str_x[4])
    return x

class GitProjectInfoExtractor:
    
    @staticmethod
    def extract_commit_info(repo_loc, proj_name):
        ROOT = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/'
        output_file_location = ('{}{}{}-basic_commit_history.csv').format(ROOT,'xin_xia_paper_data/commit_data/',proj_name)
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
                   
def collect_studied_project_meta_data():
    ob = GitProjectInfoExtractor()
    #project_path = '/scratch/ahsan/Java_Exam_Work/Data/Studied_Project_List_200.csv'
    project_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/java_xin_project.csv'
    pd_proj_data = pd.read_csv(project_path)
    pd_proj_data['repository']= [url_project_name(x) for x in pd_proj_data['html_url']]
    proj_list = list(pd_proj_data['repository'])
    total = 0
    for proj in proj_list:
        total = total + 1
        print("{} {}".format(total, proj))
        #repo_path = '/scratch/ahsan/Java_Exam_Work/GitReposistories/{}'.format(proj)
        repo_path = '/home/local/SAIL/ahsan/XIN_REPOSITORY/{}'.format(proj)
        ob.extract_commit_info(repo_path,proj)


def main():
    
    start_time = time.time()
    collect_studied_project_meta_data()
    #ob.extract_single_commit_info(repo_path,'elastic_elasticsearch','b3337c312765e51cec7bde5883bbc0a08f56fb65')
    #ob.extract_commit_info(repo_path,'elastic_elasticsearch')
    #ob.thread_impl_project_extractor(repo_path,'elastic_elasticsearch')
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Program finishes successfully!")

if __name__ == "__main__":
    main()

        
        
    
    