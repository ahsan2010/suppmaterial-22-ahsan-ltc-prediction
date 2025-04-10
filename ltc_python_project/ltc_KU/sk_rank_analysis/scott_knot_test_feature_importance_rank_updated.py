#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 01:52:00 2024

@author: ahsan
"""

from copy import deepcopy as kopy
import sys,random
import pandas as pd
import csv
import pickle
from multiprocessing import Pool
from multiprocessing import set_start_method
from multiprocessing import get_context
import time

"""
Scott-Knot test + non parametric effect size + significance tests.
Tim Menzies, 2019. Share and enjoy. No warranty. Caveat Emptor.

Accepts data as per the following exmaple (you can ignore the "*n"
stuff, that is just there for the purposes of demos on larger
and larger data)

Ouputs treatments, clustered such that things that have similar
results get the same ranks.

For a demo of this code, just run

    python3 sk.py

"""

#-----------------------------------------------------
# Examples

def skDemo(n=5) :
  #Rx.data is one way to run the code
  return Rx.data( x1 =[ 0.34, 0.49 ,0.51, 0.6]*n,
                  x2  =[0.6  ,0.7 , 0.8 , 0.89]*n,
                  x3  =[0.13 ,0.23, 0.38 , 0.38]*n,
                  x4  =[0.6  ,0.7,  0.8 , 0.9]*n,
                  x5  =[0.1  ,0.2,  0.3 , 0.4]*n)

"""
Another is to make a file

x1  0.34  0.49  0.51  0.6
x2  0.6   0.7   0.8   0.9
x3  0.15  0.25  0.4   0.35
x4  0.6   0.7   0.8   0.9
x5  0.1   0.2   0.3   0.4

Then call 

   Rx.fileIn( fileName )

"""

#-----------------------------------------------------
# Config

class o:
  def __init__(i,**d) : i.__dict__.update(**d)

class THE:
  cliffs = o(dull= [0.147, # small
                    0.33,  # medium
                    0.474 # large
                    ][0])
  bs=     o( conf=0.05,
             b=500)
  mine =  o( private="_")
  char =  o( skip="?")
  rx   =  o( show="%4s %10s %s")
  tile =  o( width=50,
             chops=[0.1 ,0.3,0.5,0.7,0.9],
             marks=[" " ,"-","-","-"," "],
             bar="|",
             star="*",
             show=" %5.3f")
#-----------------------------------------------------
def cliffsDeltaSlow(lst1,lst2, dull = THE.cliffs.dull):
  """Returns true if there are more than 'dull' difference.
     Warning: O(N)^2."""
  n= gt = lt = 0.0
  for x in lst1:
    for y in lst2:
      n += 1
      if x > y:  gt += 1
      if x < y:  lt += 1
  return abs(lt - gt)/n <= dull

def cliffsDelta(lst1, lst2,  dull=THE.cliffs.dull):
  "By pre-soring the lists, this cliffsDelta runs in NlogN time"
  def runs(lst):
    for j,two in enumerate(lst):
      if j == 0: one,i = two,0
      if one!=two:
        yield j - i,one
        i = j
      one=two
    yield j - i + 1,two
  #---------------------
  m, n = len(lst1), len(lst2)
  lst2 = sorted(lst2)
  j = more = less = 0
  for repeats,x in runs(sorted(lst1)):
    while j <= (n - 1) and lst2[j] <  x: j += 1
    more += j*repeats
    while j <= (n - 1) and lst2[j] == x: j += 1
    less += (n - j)*repeats
  d= (more - less) / (m*n)
  return abs(d)  <= dull

def bootstrap(y0,z0,conf=THE.bs.conf,b=THE.bs.b):
  """
  two  lists y0,z0 are the same if the same patterns can be seen in all of them, as well
  as in 100s to 1000s  sub-samples from each. 
  From p220 to 223 of the Efron text  'introduction to the boostrap'.
  Typically, conf=0.05 and b is 100s to 1000s.
  """
  class Sum():
    def __init__(i,some=[]):
      i.sum = i.n = i.mu = 0 ; i.all=[]
      for one in some: i.put(one)
    def put(i,x):
      i.all.append(x);
      i.sum +=x; i.n += 1; i.mu = float(i.sum)/i.n
    def __add__(i1,i2): return Sum(i1.all + i2.all)
  def testStatistic(y,z):
     tmp1 = tmp2 = 0
     for y1 in y.all: tmp1 += (y1 - y.mu)**2
     for z1 in z.all: tmp2 += (z1 - z.mu)**2
     s1    = float(tmp1)/(y.n - 1)
     s2    = float(tmp2)/(z.n - 1)
     delta = z.mu - y.mu
     if s1+s2:
       delta =  delta/((s1/y.n + s2/z.n)**0.5)
     return delta
  def one(lst): return lst[ int(any(len(lst))) ]
  def any(n)  : return random.uniform(0,n)
  y,z  = Sum(y0), Sum(z0)
  x    = y + z
  
  baseline = testStatistic(y,z)
  yhat = [y1 - y.mu + x.mu for y1 in y.all]
  zhat = [z1 - z.mu + x.mu for z1 in z.all]
  bigger = 0
  for i in range(b):
    if testStatistic(Sum([one(yhat) for _ in yhat]),
                     Sum([one(zhat) for _ in zhat])) > baseline:
      bigger += 1
  return bigger / b >= conf

#-------------------------------------------------------
# misc functions
def same(x): return x

class Mine:
  "class that, amongst other times, pretty prints objects"
  oid = 0
  def identify(i):
    Mine.oid += 1
    i.oid = Mine.oid
    return i.oid
  def __repr__(i):
    pairs = sorted([(k, v) for k, v in i.__dict__.items()
                    if k[0] != THE.mine.private])
    pre = i.__class__.__name__ + '{'
    def q(z):
     if isinstance(z,str): return "'%s'" % z
     if callable(z): return "fun(%s)" % z.__name__
     return str(z)
    return pre + ", ".join(['%s=%s' % (k, q(v))])

#-------------------------------------------------------
class Rx(Mine):
  "place to manage pairs of (TreatmentName,ListofResults)"
  def __init__(i, rx="",vals=[], key=same):
    i.rx   = rx
    i.vals = sorted([x for x in vals if x != THE.char.skip])
    i.n    = len(i.vals)
    i.med  = i.vals[int(i.n/2)]
    i.mu   = sum(i.vals)/i.n
    i.rank = 1
    
  def tiles(i,lo=0,hi=1): return  xtile(i.vals,lo,hi)
  def __lt__(i,j):        return i.med < j.med
  def __eq__(i,j):
    return cliffsDelta(i.vals,j.vals) and \
            bootstrap(i.vals,j.vals)
  def __repr__(i):
    return '%4s %10s %s' % (i.rank, i.rx, i.tiles())
  def xpect(i,j,b4):
    "Expected value of difference in emans before and after a split"
    n = i.n + j.n
    return i.n/n * (b4.med- i.med)**2 + j.n/n * (j.med-b4.med)**2

  #-- end instance methods --------------------------

  @staticmethod
  def data(**d):
    "convert dictionary to list of treatments"
    return [Rx(k,v) for k,v in d.items()]

  @staticmethod
  def fileIn(f):
    d={}
    what=None
    for word in words(f):
       x = thing(word)
       
       if isinstance(x,str): 
          what=x
          d[what] = d.get(what,[])
       else:
          d[what] += [x]
    #print(d)
    result = Rx.sk(Rx.data(**d))
    return result
   
  @staticmethod
  def sum(rxs):
    "make a new rx from all the rxs' vals"
    all = []
    for rx in rxs:
        for val in rx.vals:
            all += [val]
    return Rx(vals=all)

  @staticmethod
  def show(rxs):
    "pretty print set of treatments"
    tmp=Rx.sum(rxs)
    lo,hi=tmp.vals[0], tmp.vals[-1]
    for rx in sorted(rxs):
        print(THE.rx.show % (rx.rank, rx.rx, rx.tiles()))

  @staticmethod
  def sk(rxs):
    "sort treatments and rank them"
    def divide(lo,hi,b4,rank):
      cut = left=right=None
      best = 0
      for j in range(lo+1,hi):
          left0  = Rx.sum( rxs[lo:j] )
          right0 = Rx.sum( rxs[j:hi] )
          now    = left0.xpect(right0, b4)
          if now > best:
              if left0 != right0:
                  best, cut,left,right = now,j,kopy(left0),kopy(right0)
      if cut:
        rank = divide(lo, cut, left, rank) + 1
        rank = divide(cut ,hi, right,rank)
      else:
        for rx in rxs[lo:hi]:
          rx.rank = rank
      return rank
    #-- sk main
    rxs=sorted(rxs)
    divide(0, len(rxs),Rx.sum(rxs),1)
    return rxs

#-------------------------------------------------------
def pairs(lst):
    "Return all pairs of items i,i+1 from a list."
    last=lst[0]
    for i in lst[1:]:
         yield last,i
         last = i

def words(f):
  with open(f) as fp:
    for line in fp:
       for word in line.split():
          yield word

def xtile(lst,lo,hi,
             width= THE.tile.width,
             chops= THE.tile.chops,
             marks= THE.tile.marks,
             bar=   THE.tile.bar,
             star=  THE.tile.star,
             show=  THE.tile.show):
  """The function _xtile_ takes a list of (possibly)
  unsorted numbers and presents them as a horizontal
  xtile chart (in ascii format). The default is a
  contracted _quintile_ that shows the
  10,30,50,70,90 breaks in the data (but this can be
  changed- see the optional flags of the function).
  """
  def pos(p)   : return ordered[int(len(lst)*p)]
  def place(x) :
    return int(width*float((x - lo))/(hi - lo+0.00001))
  def pretty(lst) :
    return ', '.join([show % x for x in lst])
  ordered = sorted(lst)
  lo      = min(lo,ordered[0])
  hi      = max(hi,ordered[-1])
  what    = [pos(p)   for p in chops]
  where   = [place(n) for n in  what]
  out     = [" "] * width
  for one,two in pairs(where):
    for i in range(one,two):
      out[i] = marks[0]
    marks = marks[1:]
  out[int(width/2)]    = bar
  out[place(pos(0.5))] = star
  return '('+''.join(out) +  ")," +  pretty(what)

def thing(x):
  "Numbers become numbers; every other x is a symbol."
  try: return int(x)
  except ValueError:
    try: return float(x)
    except ValueError:
      return x

#-------------------------------------------------------
def _cliffsDelta():
  "demo function"
  lst1=[1,2,3,4,5,6,7]*100
  n=1
  for _ in range(10):
      lst2=[x*n for x in lst1]
      print(cliffsDelta(lst1,lst2),n) # should return False
      n*=1.03

def bsTest(n=1000,mu1=10,sigma1=1,mu2=10.2,sigma2=1):
   def g(mu,sigma) : return random.gauss(mu,sigma)
   x = [g(mu1,sigma1) for i in range(n)]
   y = [g(mu2,sigma2) for i in range(n)]
   return n,mu1,sigma1,mu2,sigma2,\
          'same' if bootstrap(x,y) else 'different'

#-------------------------------------------------------


def analyze_ltc_feature(location, output_file):
    random.seed(1)
    project_result = []
    
    #print('File:', ' ' , location)
    result = Rx.fileIn(location)
    #Rx.show(result)
    total_elements = len(result)
    rank = 0
    prev = - 1
    for r in range(total_elements):
        if result[r].rank != prev:
            rank = rank + 1
        row_list = [result[r].rx,rank,result[r].rank]
        project_result.append(row_list)
        prev = result[r].rank 
    df = pd.DataFrame.from_records(project_result)
    df.columns = ["feature_name", "rank","sk_rank"]
    df.to_csv(output_file, index = False)
    
dev_month_ku_expertise_file_dir   = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/'
sk_rank_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values-updated/first_sk_input/"
sk_second_input_dir ='/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values-updated/second_sk_input/csv-files/'
path = '{}full_result_updated_label_ltc{}.csv'.format(dev_month_ku_expertise_file_dir, 1)
pd_full = pd.read_csv(path)
col_list = list(pd_full.columns)
ku_features_count = list(pd_full.columns)[3:32]
ku_features_norm = list(pd_full.columns)[35:64]
ku_full_features = ku_features_count + ku_features_norm
xin_features = list(pd_full.columns)[67:129]
    
ku_proj_dim_count_col = ["proj_dim_" + x for x in ku_features_count]
ku_prev_exp_count_col = ["prev_exp_" + x for x in ku_features_count]
ku_comm_exp_count_col = ["comm_exp_" + x for x in ku_features_count]
ku_other_proj_dim_count_col =["other_proj_dim_" + x for x in ku_features_count]
    
def get_feature_name_string_dictionary():
    full_map_naming = {}
    ku_count_feature_string = ["Data_Type_KU",
                               "Operator_Decision_KU",
                               "Array_KU",
                               "Loop_KU",
                               "Method_Encapsulation_KU",
                               "Inheritance_KU",
                               "Exception_KU",
                               "Advacned_Class_Design_KU",
                               "Generic_Collection_KU",
                               "Functional_Interface_KU",
                               "Stream_API_KU",
                               "Date_Time_API_KU",
                               "IO_KU",
                               "NIO_KU",
                               "Concurrency_KU",
                               "Database_KU",
                               "Localization_KU",
                               "String_Processing_KU",
                               "Java_Persistence_KU",
                               "Enterprise_Java_Bean_KU",
                               "Java_Message_Service_API_KU",
                               "SOAP_Webservice_KU",
                               "Servlet_KU",
                               "Java_REST_API_KU",
                               "Websocket_KU",
                               "Java_Server_Face_KU",
                               "CDI_KU",
                               "Enterprise_Concurrency_KU",
                               "Batch_Processing_KU"]
    for i in range(len(ku_features_count)):
        full_map_naming[ku_features_count[i]] = "KU-CUR-PROJ-EXP:" + ku_count_feature_string[i]
        full_map_naming[ku_proj_dim_count_col[i]] = "KU-CUR-PROJ-PROF:" + ku_count_feature_string[i]
        full_map_naming[ku_prev_exp_count_col[i]] = "KU-PREV-PROJ-EXP:" + ku_count_feature_string[i]
        full_map_naming[ku_comm_exp_count_col[i]] = "KU-CUR-PROJ-COLLAB-EXP:" + ku_count_feature_string[i]
        full_map_naming[ku_other_proj_dim_count_col[i]] = "KU-PREV-PROJ-PROF:" + ku_count_feature_string[i]

    return (full_map_naming)



def get_KU_feature_name(f):
    if f in ku_features_count:
        return ("KU-CUR-PROJ-EXP:" + f)
    elif f in ku_proj_dim_count_col:
        ss = f[len("proj_dim_"):]
        return ("KU-CUR-PROJ-PROF:"+ss)
    elif f in ku_prev_exp_count_col:
        ss = f[len("prev_exp_"):]
        return ("KU-PREV-PROJ-EXP:"+ss)
    elif f in ku_comm_exp_count_col:
        ss = f[len("comm_exp_"):]
        return ("KU-CUR-PROJ-COLLAB-EXP:"+ss)
    elif f in ku_other_proj_dim_count_col:
        ss = f[len("other_proj_dim_"):]
        return ("KU-REV-PROJ-PROF:"+ss)
   
    return (f)

def get_feature_name_list(boot_lim,shap_value_output_dir_ku,file_name,year_value):
    feature_list = []
    for boot_id in range(1, boot_lim + 1):
        print("Working boot_id [{}]".format(boot_id))
        shap_output_file = "{}{}_shap_{}_{}.pkl".format(shap_value_output_dir_ku,file_name,year_value,boot_id)
        load_shap_value = pickle.load(open(shap_output_file, 'rb'))
        feature_list.extend(load_shap_value.feature_names)
    feature_list = list(set(feature_list))
    return feature_list

def multi_run_wrapper(args):
    start_time = time.time()
    boot_id = args[0]
    file_name = args[1]
    year_value = args[2]
    sk_rank_dir = args[3]
    shap_value_dir = args[4]
    sk_second_input_dir = args[5]
    location = "{}{}/{}_sk_input_year_{}_{}.txt".format(sk_rank_dir,file_name,file_name,year_value,boot_id)
    output_file_csv = "{}sk_second_{}_{}_{}.csv".format(sk_second_input_dir,file_name,year_value, boot_id)
    analyze_ltc_feature(location, output_file_csv)
    print("Complete SK Rank [{}] LTC [{}] Time: {:.4f} seconds".format(boot_id,year_value, time.time() - start_time))

    
def parallel_shap_feature_sk_rank(file_name, year_value, sk_rank_dir, boot_lim, shap_value_dir, sk_second_input_dir):
    
    start_time = time.time()
    
    sample_iterations = list(range(1,boot_lim + 1,1))
    
    with get_context("spawn").Pool(10) as p:
        p.map(multi_run_wrapper,[[boot_id,file_name,year_value, sk_rank_dir, shap_value_dir, sk_second_input_dir] for boot_id in sample_iterations])
        p.close()
        p.join()
        
    print("Done All [{}] Time: {:.4f} seconds".format(file_name,time.time() - start_time))

def ltc_feature_sk_ranking():
    shap_value_output_dir_combine = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/xin-ku-all-year/"
    shap_value_output_dir_ku = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/ku-all-dim-all-year/"
    
    year_value = 1
    KU_ALL_AUTO          = "ku_all_dim_prof_auto"
    KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"
    year_list = [1,2,3]
    for year_value in year_list:
        parallel_shap_feature_sk_rank(KU_ALL_AUTO, year_value,sk_rank_dir,100, shap_value_output_dir_ku, sk_second_input_dir)
        parallel_shap_feature_sk_rank(KU_ALL_XIN_ALL_AUTO, year_value,sk_rank_dir,100, shap_value_output_dir_combine, sk_second_input_dir)

def generate_second_sk_input(sk_second_input_dir, second_sk_output_dir, shap_value_dir, file_name, year_value, boot_lim):
    full_map_naming = get_feature_name_string_dictionary()
    feature_list = get_feature_name_list(boot_lim, shap_value_dir, file_name, year_value)
    feature_shap_rank = {}
    for f in feature_list:
        feature_shap_rank[get_KU_feature_name(f)] = []
    for boot_id in range(1, boot_lim + 1):
        sk_first_result_file = "{}sk_second_{}_{}_{}.csv".format(sk_second_input_dir,file_name,year_value, boot_id)
        pd_feature = pd.read_csv(sk_first_result_file)
        for d in zip(pd_feature['feature_name'].values, pd_feature['rank'].values):
            feature_shap_rank[d[0]].append(d[1])
    sk_second_output_file = "{}{}/{}_sk_input_year_{}.txt".format(second_sk_output_dir,file_name,file_name,year_value)
    with open(sk_second_output_file, 'w') as fp:
        for f in feature_list:
            f1 = get_KU_feature_name(f)
            rank_string = ' '.join([str(x) for x in feature_shap_rank[f1]])
            if f in full_map_naming:
                fp.write(full_map_naming[f])
            else:
                fp.write(f)
            fp.write(' ')
            fp.write(rank_string)
            fp.write('\n')
        

def run_second_sk_input():
    year_list = [1,2,3]
    KU_ALL_AUTO          = "ku_all_dim_prof_auto"
    KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"
    shap_value_output_dir_combine = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/xin-ku-all-year/"
    shap_value_output_dir_ku = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/ku-all-dim-all-year/"
    second_sk_output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values-updated/second_sk_input/"
    for year_value in year_list:
        generate_second_sk_input(sk_second_input_dir, second_sk_output_dir, shap_value_output_dir_ku,KU_ALL_AUTO, year_value, 100)
        generate_second_sk_input(sk_second_input_dir, second_sk_output_dir, shap_value_output_dir_combine,KU_ALL_XIN_ALL_AUTO, year_value, 100)
        print("Complete Year {}".format(year_value))

def run_second_sk_analysis():
    KU_ALL_AUTO          = "ku_all_dim_prof_auto"
    KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"
    second_sk_output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values-updated/second_sk_input/"
    output_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values-updated/final_sk_results/'
    year_list = [1,2,3]
    for year_value in year_list:
        for file_name in [KU_ALL_AUTO, KU_ALL_XIN_ALL_AUTO]:
            location = "{}{}/{}_sk_input_year_{}.txt".format(second_sk_output_dir,file_name,file_name,year_value)
            output_file = '{}sk_result_{}_{}.csv'.format(output_dir,file_name, year_value)
            analyze_ltc_feature(location,output_file)
            print("Model {} Year {} Complete.".format(file_name, year_value))
        
if __name__ == "__main__":
    #ltc_feature_sk_ranking()
    #run_second_sk_input()
    run_second_sk_analysis()
    print('Program finishes Successful')
    
    
    
