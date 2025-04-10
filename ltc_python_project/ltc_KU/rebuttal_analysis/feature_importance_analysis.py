
import shap
import pandas as pd
import os.path
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import roc_auc_score

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

from multiprocessing import Pool
from multiprocessing import set_start_method
from multiprocessing import get_context

from lightgbm import LGBMClassifier
import time
import pickle
import sys
import_file_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/ltc_python_project/ltc_KU/ltc-model-analysis/"
sys.path.append(import_file_path)
from autospearman_kla_python_impl import AutoSpearman
import warnings
warnings.simplefilter('ignore') #we don't wanna see that

t_name = ['Ahsan_Java_Commit','Xin_Any_Commit']


sk_input_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/feature-analysis/sk-rank-input/'
data_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/data/'   

KU_ALL_AUTO          = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"
KU_DEV_EXP_XIN_AUTO = "ku_dev_exp_xin_all_auto"

# KU feature dimension index
dev_exp_features_ind = list(range(3,32))
cur_proj_features_ind = list(range(94,123))
prev_dev_exp_features_ind = list(range(123,152))
col_exp_features_ind = list(range(152,181))
prev_proj_features_ind = list(range(181,210))

#Xin feature dimension index
xin_dev_prof = list(range(33,40))
xin_repo_prof = list(range(40,59))
xin_month_repo_prof = list(range(59,77))
xin_dev_act_prof = list(range(77,89))
xin_collab_prof = list(range(89,94))

KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

bootstrap_index_file    = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/bootstrap_index.csv"
pd_bootstrap_index = pd.read_csv(bootstrap_index_file)
shap_value_output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/shap/"

save_model_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/models/" 

full_model = [KU_ALL_AUTO, KU_ALL_XIN_ALL_AUTO, KU_CUR_XIN_AUTO]

ku_dimension_list = [
    'KU_DEV_EXP',
    'KU_CUR_PROJ',
    'KU_DEV_PREV_EXP',
    'KU_COLLAB_EXP',
    'KU_PREV_PROJ'
    ]

xin_dimension_list = [
    'BAO_DEV_PPROF',
    'BAO_REPO_PROF',
    'BAO_DEV_ACT',
    'BAO_REPO_ACT',
    'BAO_COLLAB_NET'
    ]


def map_feature_to_dimension(pd_feature,model_dimension_list, map_feature_dimension):
    feature_list = list(pd_feature['feature_name'])
    dimension_list = []
    for f in feature_list:
        dimension_list.append(map_feature_dimension[f])
    return dimension_list

def find_value_direction_II(x,y,d):
    if x in list(d['feature_name']):
        vq3 = d[d['feature_name']==x]['q3'].iloc[0]
        vq1 = d[d['feature_name']==x]['q1'].iloc[0]
        vmed = d[d['feature_name']==x]['median'].iloc[0]
        if y > vq3:
            return 'H'
        elif y < vq1:
            return 'L'
        return 'M'
    return 'E'

def find_value_direction(x,y,d):
    if x in list(d['feature_name']):
        vq3 = d[d['feature_name']==x]['q3'].iloc[0]
        vq1 = d[d['feature_name']==x]['q1'].iloc[0]
        vmed = d[d['feature_name']==x]['median'].iloc[0]
        if y > vmed:
            return 'H'
        else: 
            return 'L'
        
    return 'E'


def analysis_dimensio_majority(pd_feature, row_ind):
    
    dimension_list = list(set(pd_feature['dimension']))
    result = []
    for d in dimension_list:
        shap_dir = 'M'
        data_dir = 'M'
        d_data = pd_feature[pd_feature['dimension'] == d]
        shap_pos = d_data[d_data['shap_value'] > 0].shape[0]
        shap_neg = d_data[d_data['shap_value'] < 0].shape[0]
        
        if shap_pos > shap_neg:
            shap_dir = 'P'
        elif shap_pos < shap_neg:
            shap_dir = 'N'
        else:
            shap_dir = 'M'
        
        data_high = pd_feature[pd_feature['value_direction'] == 'H'].shape[0]
        data_low = pd_feature[pd_feature['value_direction'] == 'L'].shape[0]
        if data_low > data_high:
            data_dir = 'L'
        elif data_high > data_low:
            data_dir = 'H'
        else:
            data_dir = 'M'
        
        result.append([row_ind, d, shap_dir, data_dir])
    
    return result
    
def pos(col):  
  col = [np.maximum(x,0) for x in col]
  v = np.sum(col)
  return v
  
def neg(col):
  col = [np.minimum(x,0) for x in col]
  v = np.sum(col)
  return v

def get_dimension_wise_direction(pd_full_index_data):
    dim_dic = {}
    for dim in ku_dimension_list:
        feature_dimension_data = pd_full_index_data[pd_full_index_data['dimension']==dim]
        d = feature_dimension_data.groupby(['row_ind'])
        p = d['shap_value'].agg([('negative_values', neg), 
                  ('positive_values', pos) 
                  ])
        positive_contribution = np.mean(p['positive_values'])
        negative_contribution = np.mean(p['negative_values'])
        #print('{} {} {}'.format(dim,positive_contribution, negative_contribution))
        dim_dic[dim] = (positive_contribution,negative_contribution)
    return dim_dic


def generate_singel_boot_feature_study(load_shap_value, model_dimension_list, map_feature_dimension, ku_feature_dimension_list):
    feature_list = load_shap_value.feature_names
    
    data_frame_list = []
    data_direction_list = []
    group_shap_values = np.zeros((len(load_shap_value.values), len(ku_feature_dimension_list)))
    group_data_values = np.zeros((len(load_shap_value.values), len(ku_feature_dimension_list)))

    for i, (group_name, features) in enumerate(ku_feature_dimension_list.items()):
        print('i={} {}'.format(i,group_name))
        features = [f for f in features if f in feature_list]
        indices = [feature_list.index(f) for f in features]  
        group_shap_values[:, i] = load_shap_value.values[:, indices].sum(axis=1)
        group_data_values[:, i] = load_shap_value.data[:, indices].sum(axis=1)
        
    group_shap_explanation = shap.Explanation(
        values=group_shap_values,
        data=group_data_values,  # Include actual data values
        feature_names=list(ku_feature_dimension_list.keys())
    ) 
    shap.plots.beeswarm(group_shap_explanation)

    for row_ind in range(len(load_shap_value.values)):
        print('working {}/{}'.format(row_ind,len(load_shap_value.values)))
        feature_raw_data = (load_shap_value.data[row_ind, :])
        feature_name = load_shap_value.feature_names
        feature_data = (load_shap_value.values[row_ind, :])
        pd_feature = pd.DataFrame({'feature_name':feature_name,'shap_value': feature_data, 'data_value':feature_raw_data})
        dimension_list = map_feature_to_dimension(pd_feature,  model_dimension_list, map_feature_dimension)
        pd_feature['dimension'] = dimension_list
        pd_feature['row_ind'] = row_ind
        
        v = pd_feature.groupby(['dimension']).agg({
            'shap_value' : 'sum',
            'data_value' : 'sum'
            }).reset_index()
       
        
        
def generate_single_boot_feature_ranking(load_shap_value, model_dimension_list, map_feature_dimension):
    feature_list = load_shap_value.feature_names
    feature_shap_rank = {}
    
    feature_data_list = []
    
    for i in range(0,len(feature_list)):
        feature_i_name = feature_list[i]
        feature_i_data = load_shap_value.data[:,i]
        min = np.min(feature_i_data)
        max = np.max(feature_i_data)
        median = np.median(feature_i_data)
        q1 = np.quantile(feature_i_data, 0.25)
        q3 = np.quantile(feature_i_data,0.75)
        
        if min == 0 and max == 0:
            continue
        
        feature_data_list.append([feature_i_name, map_feature_dimension[feature_i_name],min, q1, median, q3, max])
    
    pd_feature_data_distr = pd.DataFrame(feature_data_list)
    pd_feature_data_distr.columns=['feature_name','feature_dim','min','q1','median','q3','max']

    v = find_value_direction('other_proj_dim_array', 0, pd_feature_data_distr)
    
    for d in model_dimension_list:
        feature_shap_rank[d] = []
    
    data_frame_list = []
    data_direction_list = []
    for row_ind in range(len(load_shap_value.values)):
        print('working {}/{}'.format(row_ind,len(load_shap_value.values)))
        feature_raw_data = (load_shap_value.data[row_ind, :])
        feature_name = load_shap_value.feature_names
        feature_data = (load_shap_value.values[row_ind, :])
        pd_feature = pd.DataFrame({'feature_name':feature_name,'shap_value': feature_data, 'data_value':feature_raw_data})
        dimension_list = map_feature_to_dimension(pd_feature,  model_dimension_list, map_feature_dimension)
        pd_feature['dimension'] = dimension_list
        pd_feature['row_ind'] = row_ind
        pd_feature = pd_feature[pd_feature['feature_name'].isin(list(pd_feature_data_distr['feature_name']))]
        pd_feature['value_direction'] = [find_value_direction(x,y,pd_feature_data_distr) for x,y in zip(pd_feature['feature_name'],pd_feature['data_value'])]
        
        res = analysis_dimensio_majority(pd_feature, row_ind)
        pd_data_dir = pd.DataFrame(res)
        data_direction_list.append(pd_data_dir)
        data_frame_list.append(pd_feature)
        #pd_sum = pd_feature.groupby('dimension').sum().reset_index()
        '''
        for d in model_dimension_list:
            if d in list(pd_sum['dimension']):
                v = pd_sum[pd_sum['dimension'] == d].shap_value.values[0]
                feature_shap_rank[d].append(v)
        '''
    pd_full_index_data = pd.concat(data_frame_list)
    
    feature_dimension_data = pd_full_index_data[pd_full_index_data['dimension']=='KU_DEV_EXP']
    d = feature_dimension_data.groupby(['row_ind'])
    p = d['shap_value'].agg([('negative_values', neg), 
                  ('positive_values', pos) 
                  ])
    positive_contribution = np.mean(p['positive_values'])
    negative_contribution = np.mean(p['negative_values'])
    
    dim_dic = get_dimension_wise_direction(pd_full_index_data)
    
    # Determine the direction
    if positive_contribution > abs(negative_contribution):
        direction = "Positive"
    elif abs(negative_contribution) > positive_contribution:
        direction = "Negative"
    else:
        direction = "Mixed"
        
    # Print results
    #print(f"Group: {feature_group}")
    #print(f"Positive Contribution: {positive_contribution}")
    #print(f"Negative Contribution: {negative_contribution}")
    #print(f"Direction: {direction}")
    
    
    pd_data_direction = pd.concat(data_direction_list)
    pd_data_direction.columns = ['row_ind','dimension','shap_dir','data_dir']
    
    #pd_data_direction.groupby(['dimension','shap_dir','data_dir']).size()
    
    return feature_shap_rank, pd_data_direction, pd_full_index_data, positive_contribution, negative_contribution, dim_dic
    

def get_feature_names(col_list):
    dev_exp_ku_features = [col_list[i] for i in dev_exp_features_ind] 
    cur_proj_ku_features = [col_list[i] for i in cur_proj_features_ind]
    prev_dev_exp_features = [col_list[i] for i in prev_dev_exp_features_ind]
    col_exp_features = [col_list[i] for i in col_exp_features_ind]
    prev_proj_features = [col_list[i] for i in prev_proj_features_ind]
    
    xin_dev_prof_features = [col_list[i] for i in xin_dev_prof]
    xin_repo_prof_features = [col_list[i] for i in xin_repo_prof]
    xin_month_repo_prof_features = [col_list[i] for i in xin_month_repo_prof]
    xin_dev_act_prof_features = [col_list[i] for i in xin_dev_act_prof]
    xin_collab_prof_features = [col_list[i] for i in xin_collab_prof]

    ku_feature_dimension_list ={
        'KU_DEV_EXP' : dev_exp_ku_features,
        'KU_CUR_PROJ' : cur_proj_ku_features,
        'KU_DEV_PREV_EXP' : prev_dev_exp_features,
        'KU_COLLAB_EXP' : col_exp_features,
        'KU_PREV_PROJ' : prev_proj_features
        }
    
    xin_feature_dimension_list = {
        'BAO_DEV_PPROF'  : xin_dev_prof_features,
        'BAO_REPO_PROF' : xin_repo_prof_features,
        'BAO_DEV_ACT' : xin_dev_act_prof_features,
        'BAO_REPO_ACT' : xin_month_repo_prof_features,
        'BAO_COLLAB_NET' : xin_collab_prof_features
        }
    
    feature_dimension_list = {
        'KU_DEV_EXP' : dev_exp_ku_features,
        'KU_CUR_PROJ' : cur_proj_ku_features,
        'KU_DEV_PREV_EXP' : prev_dev_exp_features,
        'KU_COLLAB_EXP' : col_exp_features,
        'KU_PREV_PROJ' : prev_proj_features,
        'BAO_DEV_PPROF'  : xin_dev_prof_features,
        'BAO_REPO_PROF' : xin_repo_prof_features,
        'BAO_DEV_ACT' : xin_dev_act_prof_features,
        'BAO_REPO_ACT' : xin_month_repo_prof_features,
        'BAO_COLLAB_NET' : xin_collab_prof_features
        }
    
     
    map_feature_dimension = {}
    for key,value in feature_dimension_list.items():
        f_list = value
        for f in f_list:
            map_feature_dimension[f] = key
    
    return ku_feature_dimension_list, xin_feature_dimension_list, feature_dimension_list, map_feature_dimension

def get_feature_name_list(boot_lim,shap_value_output_dir_ku,file_name,year_value):
    feature_list = []
    for boot_id in range(1, boot_lim + 1):
        print("Working boot_id [{}]".format(boot_id))
        shap_output_file = "{}{}_shap_{}_{}.pkl".format(shap_value_output_dir_ku,file_name,year_value,boot_id)
        load_shap_value = pickle.load(open(shap_output_file, 'rb'))
        feature_list.extend(load_shap_value.feature_names)
    feature_list = list(set(feature_list))
    return feature_list

def shap_feature_analysis_boot_wise(year_value):
    
    shap_value_output_dir_combine = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/xin-ku-all-year/"
    shap_value_output_dir_ku = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/ku-all-dim-all-year/"
    
    boot_lim = 100
    m = KU_ALL_AUTO
    #year_value = 1
    shap_value_dir = shap_value_output_dir_ku
    
    path = "{}ltc_data_full_year_{}.csv".format(data_dir, year_value)
    pd_full = pd.read_csv(path)
    col_list = list(pd_full.columns)
    
    feature_list = get_feature_name_list(boot_lim, shap_value_dir, m, year_value)
    
    
    ku_feature_dimension_list, xin_feature_dimension_list, feature_dimension_list, map_feature_dimension = get_feature_names(col_list)
    model_dimension_list = {
        XIN_FEAT_AUTO : ['BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET'],
        KU_ALL_AUTO : ['KU_DEV_EXP','KU_CUR_PROJ','KU_DEV_PREV_EXP','KU_COLLAB_EXP','KU_PREV_PROJ'],
        KU_ALL_XIN_ALL_AUTO : ['KU_DEV_EXP','KU_CUR_PROJ','KU_DEV_PREV_EXP','KU_COLLAB_EXP','KU_PREV_PROJ','BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET'],
        KU_CUR_XIN_AUTO : ['KU_DEV_EXP','BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET']
        }
    
    
    result = []
    
    for boot_id in range(1, boot_lim + 1):
        print("Working boot_id [{}/{}]".format(boot_id, boot_lim))
        shap_output_file = "{}{}_shap_{}_{}.pkl".format(shap_value_dir,m,year_value,boot_id)
        load_shap_value = pickle.load(open(shap_output_file, 'rb'))
        feature_shap_rank, pd_data_direction, pd_full_index_data, positive_contribution, negative_contribution,dim_dic = generate_single_boot_feature_ranking(load_shap_value, model_dimension_list[m], map_feature_dimension)
        
        for d in ku_dimension_list:
            result.append([year_value, boot_id, d, dim_dic[d][0], dim_dic[d][1]])
        
        
    
    f_name = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/feature_importance_reb/year_{}.csv'.format(year_value)
    pd_result = pd.DataFrame(result)
    pd_result.columns = ['year','boot_id','dim','pos', 'neg']
    pd_result.to_csv(f_name,index=False)
    
    #avg_pos = np.median(pd_result['pos'])
    #avg_neg = np.median(abs(pd_result['neg']))

def shap_featue_analysis(year_value):
    path = "{}ltc_data_full_year_{}.csv".format(data_dir, year_value)
    pd_full = pd.read_csv(path)
    col_list = list(pd_full.columns)
    
    ku_feature_dimension_list, xin_feature_dimension_list, feature_dimension_list, map_feature_dimension = get_feature_names(col_list)
    model_dimension_list = {
        XIN_FEAT_AUTO : ['BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET'],
        KU_ALL_AUTO : ['KU_DEV_EXP','KU_CUR_PROJ','KU_DEV_PREV_EXP','KU_COLLAB_EXP','KU_PREV_PROJ'],
        KU_ALL_XIN_ALL_AUTO : ['KU_DEV_EXP','KU_CUR_PROJ','KU_DEV_PREV_EXP','KU_COLLAB_EXP','KU_PREV_PROJ','BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET'],
        KU_CUR_XIN_AUTO : ['KU_DEV_EXP','BAO_DEV_PPROF','BAO_REPO_PROF','BAO_DEV_ACT','BAO_REPO_ACT','BAO_COLLAB_NET']
        }
    
    
    for m in [KU_ALL_AUTO, KU_ALL_XIN_ALL_AUTO, KU_CUR_XIN_AUTO]:
        shap_file = "{}{}_shap_full_{}.pkl".format(shap_value_output_dir,m,year_value)
        load_shap_value = pickle.load(open(shap_file, 'rb'))
        feature_shap_value = generate_single_boot_feature_ranking(load_shap_value, model_dimension_list[m], map_feature_dimension)
   
        sk_output_file = "{}{}_sk_input_year_{}.txt".format(sk_input_dir,m,year_value)
        with open(sk_output_file, 'w') as fp:
            for f in feature_shap_value.keys():
                rank_string = ' '.join([str(x) for x in feature_shap_value[f]])
                fp.write(f)
                fp.write(' ')
                fp.write(rank_string)
                fp.write('\n')

        print("[Finish] file = {} year = {}".format(m, year_value))



def main():
    #year_value = 1
    start_time = time.time()
    year_value = int(sys.argv[1])
    print(sys.argv)
    shap_feature_analysis_boot_wise(year_value)
    print(" YEAR {} {:.4f} minutes ---".format(year_value, (time.time() - start_time)/60.0))
    
    #for year_value in [1,2,3]:
        #shap_featue_analysis(year_value)
        #shap_feature_analysis_boot_wise(year_value)
    print("main function..")

if __name__ == "__main__":
    main()
    print("Program finishes successfully.")
