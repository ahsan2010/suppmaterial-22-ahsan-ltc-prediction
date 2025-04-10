library(ggplot2)
library(dplyr)
library(tidyr)
library(ggh4x)

fig_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/model-sk-analysis/rank-result/figures-sk-rank/"
sk_rank_dir       = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/model-sk-analysis/rank-result/"
model_result_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/ku-different-dimension-100-bootstrap-dec-15/"

KU_ALL          = "ku_all_dim_prof"
XIN_FEAT        = "xin_feature"
KU_ALL_XIN_ALL  = "ku_all_xin_all"

KU_ALL_AUTO          = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

# ltc_1_com_exp_prof_100.csv
# ltc_1_com_exp_prof_auto_100.csv

year_list <- c(1, 2, 3)

model_without_auto  = c(KU_ALL, XIN_FEAT)
model_with_auto     = c(KU_ALL_AUTO, XIN_FEAT_AUTO)


# Load sk rank data
sk_rank_file = sprintf("%ssk_rank_results_two.csv",sk_rank_dir)
rank_data = read.csv(file= sk_rank_file, header=TRUE,sep=",")
rank_data_withotu_auto = rank_data[rank_data$model_type == "no-auto",]
rank_data_with_auto = rank_data[rank_data$model_type == "auto",]


selected_columns = c("ltc_year","features", "classifier", "bootstrap", "auc")
full_data <- NULL
scaleFUN <- function(x) sprintf("%.2f", x)
for (year in year_list){
  for (m in model_without_auto){
    file_path = sprintf("%sltc_%d_%s_100.csv",model_result_dir, year, m )
    distr_data = read.csv(file= file_path, header=TRUE,sep=",")
    distr_data = distr_data[distr_data$classifier=="RandomForest",]
    distr_data = distr_data[selected_columns]
    distr_data$model_type = "no-auto"
    full_data <- rbind(full_data, distr_data)
  }
  for (m in model_with_auto){
    file_path = sprintf("%sltc_%d_%s_100.csv",model_result_dir, year, m )
    distr_data = read.csv(file= file_path, header=TRUE,sep=",")
    distr_data = distr_data[distr_data$classifier=="RandomForest",]
    distr_data = distr_data[selected_columns]
    distr_data$model_type = "auto"
    full_data <- rbind(full_data, distr_data)
  }
}

colnames(full_data) <- c("ltc", "model_name", "classifier", "bootstrap", "auc","model_type")

distribution_with_rank = merge(full_data, rank_data, by = c("model_name", "ltc", "model_type"))
distribution_with_rank$ltc <- sub("^", "LTC-", distribution_with_rank$ltc )
distribution_with_rank$rank <- sub("^", "Rank-", distribution_with_rank$rank )

distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "ku_all_dim_prof", "KULTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "xin_feature", "Bao et al.")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "ku_all_xin_all", "KULTC+Bao et al.")

distribution_with_rank$model_type =  str_replace(distribution_with_rank$model_type, "no-auto", "All-Features")
distribution_with_rank$model_type =  str_replace(distribution_with_rank$model_type, "auto", "Filtered-Features")




# LTC 1 result with and without autoSpearman

distribution_with_rank_LTC_1 = distribution_with_rank[distribution_with_rank$ltc == "LTC-1", ]
meds <- distribution_with_rank_LTC_1 %>% group_by(model_name, model_type, ltc ,rank) %>% summarise(med = median(auc))

distribution_with_rank_LTC_1 <- ggplot(distribution_with_rank_LTC_1, aes(model_name, auc)) +    # Create facet plot again
  geom_boxplot() +
  #facet_grid(.~ltc,scale="free", space="free_x") + 
  #scale_x_discrete(labels=setNames(as.character(boxData$Classification), boxData$temp)) +
  scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0,1)) + 
  ylab("AUC") +
  xlab("Models") +
  theme_bw() + theme(text = element_text(size=16))

gg_auc_model_ltc_1 = distribution_with_rank_LTC_1 + facet_nested(~model_type+rank, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5)


ggsave(gg_auc_model_ltc_1,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-ltc-1'), width = 12, height = 6)
  
##################################################################################################


selected_columns = c("ltc_year","features", "classifier", "bootstrap", "auc")
full_data <- NULL
scaleFUN <- function(x) sprintf("%.2f", x)
for (year in year_list){
  for (m in model_without_auto){
    file_path = sprintf("%sltc_%d_%s_100.csv",model_result_dir, year, m )
    distr_data = read.csv(file= file_path, header=TRUE,sep=",")
    distr_data = distr_data[distr_data$classifier=="RandomForest",]
    distr_data = distr_data[selected_columns]
    distr_data$model_type = "no-auto"
    full_data <- rbind(full_data, distr_data)
  }
}

colnames(full_data) <- c("ltc", "model_name", "classifier", "bootstrap", "auc","model_type")

distribution_with_rank_without_auto = merge(full_data, rank_data_withotu_auto, by = c("model_name", "ltc", "model_type"))
distribution_with_rank_without_auto$ltc <- sub("^", "LTC-", distribution_with_rank_without_auto$ltc )
distribution_with_rank_without_auto$rank <- sub("^", "Rank-", distribution_with_rank_without_auto$rank )

distribution_with_rank_without_auto$model_name =  str_replace(distribution_with_rank_without_auto$model_name, "ku_all_dim_prof", "KULTC")
distribution_with_rank_without_auto$model_name =  str_replace(distribution_with_rank_without_auto$model_name, "xin_feature", "Bao et al.")
distribution_with_rank_without_auto$model_name =  str_replace(distribution_with_rank_without_auto$model_name, "ku_all_xin_all", "KULTC+Bao et al.")


meds <- distribution_with_rank_without_auto %>% group_by(model_name, ltc, rank) %>% summarise(med = median(auc))

gg_auc_models_without_auto <- ggplot(distribution_with_rank_without_auto, aes(model_name, auc)) +    # Create facet plot again
  geom_boxplot() +
  #facet_grid(.~ltc,scale="free", space="free_x") + 
  #scale_x_discrete(labels=setNames(as.character(boxData$Classification), boxData$temp)) +
  scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0,1)) + 
  ylab("AUC") +
  xlab("Models") +
  theme_bw() + theme(text = element_text(size=16))

gg_auc_models_without_auto = gg_auc_models_without_auto + facet_nested(~ltc+rank, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5)


ggsave(gg_auc_models_without_auto,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-without_auto'), width = 16, height = 6)

#################################################
selected_columns = c("ltc_year","features", "classifier", "bootstrap", "auc")
full_data <- NULL
scaleFUN <- function(x) sprintf("%.2f", x)
for (year in year_list){
  for (m in model_with_auto){
    file_path = sprintf("%sltc_%d_%s_100.csv",model_result_dir, year, m )
    distr_data = read.csv(file= file_path, header=TRUE,sep=",")
    distr_data = distr_data[distr_data$classifier=="RandomForest",]
    distr_data = distr_data[selected_columns]
    distr_data$model_type = "auto"
    full_data <- rbind(full_data, distr_data)
  }
}
colnames(full_data) <- c("ltc", "model_name", "classifier", "bootstrap", "auc","model_type")

distribution_with_rank_with_auto = merge(full_data, rank_data_with_auto, by = c("model_name", "ltc", "model_type"))
distribution_with_rank_with_auto$ltc <- sub("^", "LTC-", distribution_with_rank_with_auto$ltc )
distribution_with_rank_with_auto$rank <- sub("^", "Rank-", distribution_with_rank_with_auto$rank )

distribution_with_rank_with_auto$model_name =  str_replace(distribution_with_rank_with_auto$model_name, "ku_all_dim_prof_auto", "KULTC")
distribution_with_rank_with_auto$model_name =  str_replace(distribution_with_rank_with_auto$model_name, "xin_feature_auto", "Bao et al.")
distribution_with_rank_with_auto$model_name =  str_replace(distribution_with_rank_with_auto$model_name, "ku_all_xin_all_auto", "KULTC+Bao et al.")

meds <- distribution_with_rank_with_auto %>% group_by(model_name, ltc, rank) %>% summarise(med = median(auc))

gg_auc_models_with_auto <- ggplot(distribution_with_rank_with_auto, aes(model_name, auc)) +    # Create facet plot again
  geom_boxplot() +
  #facet_grid(.~ltc,scale="free", space="free_x") + 
  #scale_x_discrete(labels=setNames(as.character(boxData$Classification), boxData$temp)) +
  scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0,1)) + 
  ylab("AUC") +
  xlab("Models") +
  theme_bw() + theme(text = element_text(size=16))

gg_auc_models_with_auto = gg_auc_models_with_auto + 
                        facet_nested(~ltc+rank, scales = "free_x", independent = "x") +
                        geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5)

ggsave(gg_auc_models_with_auto,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-with_auto'), width = 14, height = 6)

########### ################ ##
  ## KU Variations ## 
########### ################ ##

KU_ALL          = "ku_all_dim_prof"
KU_EXPERIENCE = "experience_ku"
KU_PROJECT = "project_ku"
CURRENT_PROJ_EXP_KU = "current_proj_exp_ku"
PREV_PROJ_EXP_KU = "prev_proj_exp_ku"

model_list_new = c(
  KU_ALL,
  KU_EXPERIENCE, 
  KU_PROJECT, 
  CURRENT_PROJ_EXP_KU, 
  PREV_PROJ_EXP_KU)

# Load sk rank data
sk_rank_file = sprintf("%ssk_rank_results_remain_ku_models.csv",sk_rank_dir)
rank_data = read.csv(file= sk_rank_file, header=TRUE,sep=",")
rank_data_withotu_auto = rank_data[rank_data$model_type == "no-auto",]
rank_data_with_auto = rank_data[rank_data$model_type == "auto",]


selected_columns = c("ltc_year","features", "classifier", "bootstrap", "auc")
full_data <- NULL
scaleFUN <- function(x) sprintf("%.2f", x)

for (year in year_list){
  for (m in model_list_new){
    file_path = sprintf("%sltc_%d_%s_100.csv",model_result_dir, year, m )
    distr_data = read.csv(file= file_path, header=TRUE,sep=",")
    distr_data = distr_data[distr_data$classifier=="RandomForest",]
    distr_data = distr_data[selected_columns]
    distr_data$model_type = "no-auto"
    full_data <- rbind(full_data, distr_data)
  }
}

colnames(full_data) <- c("ltc", "model_name", "classifier", "bootstrap", "auc","model_type")

distribution_with_rank_without_auto = merge(full_data, rank_data_withotu_auto, by = c("model_name", "ltc", "model_type"))
distribution_with_rank_without_auto$ltc <- sub("^", "LTC-", distribution_with_rank_without_auto$ltc )
distribution_with_rank_without_auto$rank <- sub("^", "Rank-", distribution_with_rank_without_auto$rank )

distribution_with_rank_without_auto$model_name =  str_replace(distribution_with_rank_without_auto$model_name, "ku_all_dim_prof", "KULTC")
distribution_with_rank_without_auto$model_name =  str_replace(distribution_with_rank_without_auto$model_name, "current_proj_exp_ku", "KULTC-CUR")
distribution_with_rank_without_auto$model_name =  str_replace(distribution_with_rank_without_auto$model_name, "prev_proj_exp_ku", "KULTC-PREV")
distribution_with_rank_without_auto$model_name =  str_replace(distribution_with_rank_without_auto$model_name, "experience_ku", "KULTC-EXP")
distribution_with_rank_without_auto$model_name =  str_replace(distribution_with_rank_without_auto$model_name, "project_ku", "KULTC-PROJ")

meds <- distribution_with_rank_without_auto %>% group_by(model_name, ltc, rank) %>% dplyr::summarise(med = median(auc))

gg_auc_models_without_auto <- ggplot(distribution_with_rank_without_auto, aes(model_name, auc)) +    # Create facet plot again
  geom_boxplot() +
  #facet_grid(.~ltc,scale="free", space="free_x") + 
  #scale_x_discrete(labels=setNames(as.character(boxData$Classification), boxData$temp)) +
  scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0,1)) + 
  ylab("AUC") +
  xlab("Models") +
  theme_bw() + theme(text = element_text(size=16)) + theme(axis.text.x = element_text(angle=90))

gg_auc_models_without_auto = gg_auc_models_without_auto + facet_nested(~ltc+rank, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5)


ggsave(gg_auc_models_without_auto,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-without_auto_ku_remaining'), width = 16, height = 6)



########### ################ ##
## KU Xin Combined Models ## 
########### ################ ##
fig_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/model-sk-analysis/rank-result/figures-sk-rank/"
sk_rank_dir       = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/model-sk-analysis/rank-result/"
model_result_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/ku-different-dimension-100-bootstrap-dec-15/"


KU_ALL          = "ku_all_dim_prof"
XIN_FEAT        = "xin_feature"
KU_ALL_XIN_ALL  = "ku_all_xin_all"

KU_XIN_DEV = "ku_xin_dev_dim"
KU_XIN_REPO = "ku_xin_repo_dim"
KU_XIN_DEV_ACT = "ku_xin_dev_act_dim"
KU_XIN_REPO_ACT = "ku_xin_repo_act_dim"
KU_XIN_COLLAB_NET = "ku_xin_collab_net_xim"
KU_XIN_ALL = "ku_xin_all_dim"

model_rename = c(
  "KULTC",
  "Bao et al.",
  "KULTC+DEV",
  "KULTC+REPO",
  "KULTC+DEV_ACT",
  "KULTC+REPO_ACT",
  "KULTC+COLLAB",
  "KULTC+Bao et al."
)

model_ku_xin_comb = c(KU_ALL,
                     XIN_FEAT,
                     KU_XIN_DEV,
                     KU_XIN_REPO,
                     KU_XIN_DEV_ACT,
                     KU_XIN_REPO_ACT,
                     KU_XIN_COLLAB_NET,
                     KU_XIN_ALL)


selected_columns = c("ltc_year","features", "classifier", "bootstrap", "auc")
full_data <- NULL
scaleFUN <- function(x) sprintf("%.2f", x)

year_list = c(1, 2, 3)
model_type = c('auto', 'no-auto')
load_full_distrib_data <- function(model_result_dir, year_list, model_list, model_type){
  selected_columns = c("ltc_year","features", "classifier", "bootstrap", "auc")
  full_data <- NULL
  auto_string = NULL
  if (model_type == "auto"){
    auto_string = 'auto_'
  }else if (model_type == "no-auto"){
    auto_string = ''
  }
  print(sprintf("AUTO STRING: %s", auto_string))
  for (year in year_list){
    for (m in model_list){
      file_path = sprintf("%sltc_%d_%s_%s100.csv",model_result_dir, year, m, auto_string )
      #print(file_path)
      distr_data = read.csv(file= file_path, header=TRUE,sep=",")
      distr_data = distr_data[distr_data$classifier=="RandomForest",]
      distr_data = distr_data[selected_columns]
      distr_data$model_type = model_type
      full_data <- rbind(full_data, distr_data)
    }
  }
  return (full_data)
}

figure_generate_sk_rank_result <- function(rank_data, model_type_list, model_list, model_rename, model_result_dir, file_analysis_name){
  for(type_name in model_type_list){
    full_data <- NULL
    rank_data_type <- rank_data[rank_data$model_type == type_name,]
    full_data <- load_full_distrib_data(model_result_dir, year_list, model_list, type_name)
    colnames(full_data) <- c("ltc", "model_name", "classifier", "bootstrap", "auc","model_type")
    
    distribution_with_rank = merge(full_data, rank_data_type, by = c("model_name", "ltc", "model_type"))
    distribution_with_rank$ltc <- sub("^", "LTC-", distribution_with_rank$ltc )
    distribution_with_rank$rank <- sub("^", "Rank-", distribution_with_rank$rank )
    
    auto_string = NULL
    if (type_name == "auto"){
      auto_string = '_auto'
    }else if (type_name == "no-auto"){
      auto_string = ''
    }
    
    for(i in 1:length(model_rename)){
      distribution_with_rank$model_name <- str_replace(distribution_with_rank$model_name, paste(model_list[i],auto_string,sep=''), model_rename[i])
    }
    meds <- distribution_with_rank %>% group_by(model_name, ltc, rank) %>% dplyr::summarise(med = median(auc))
    
    gg_auc_models <- ggplot(distribution_with_rank, aes(model_name, auc)) +    # Create facet plot again
      geom_boxplot() +
      #facet_grid(.~ltc,scale="free", space="free_x") + 
      #scale_x_discrete(labels=setNames(as.character(boxData$Classification), boxData$temp)) +
      scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0,1)) + 
      ylab("AUC") +
      xlab("Models") +
      theme_bw() + theme(text = element_text(size=16)) + scale_x_discrete(guide = guide_axis(angle = 90)) 
      #theme(axis.text.x = element_text(angle=90))
    
    gg_auc_models = gg_auc_models + facet_nested(~ltc+rank, scales = "free_x", independent = "x") +
      geom_text(data = meds, aes(y = med, label = round(med,2)),size = 3, vjust = -1.0, hjust = 0.5)
    
    ggsave(gg_auc_models,filename = sprintf("%ssk-rank-model-%s-%s.pdf",fig_path,file_analysis_name, type_name), width = 16, height = 6)
    
  }
}

# Load sk rank data
sk_rank_file = sprintf("%ssk_rank_results_ku_xin_combined_analysis.csv",sk_rank_dir)
rank_data = read.csv(file= sk_rank_file, header=TRUE,sep=",")
figure_generate_sk_rank_result(rank_data, model_type, model_ku_xin_comb, model_rename, model_result_dir, "ku-xin-combined")
