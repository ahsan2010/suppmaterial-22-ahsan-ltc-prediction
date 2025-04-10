library(ggplot2)
library(dplyr)
library(tidyr)
library(ggh4x)
library(stringr)


KU_DEV_EXP      = "dev_exp_prof"
KU_OTHER_EXP    = "prev_exp_prof"
KU_COMMUNITY    = "com_exp_prof"
PROJECT_PROF    = "proj_prof"
PREV_PROJ_PROF  = "prev_proj_prof"
KU_ALL          = "ku_all_dim_prof"
XIN_FEAT        = "xin_feature"
KU_ALL_XIN_ALL  = "ku_all_xin_all"

KU_ALL_MINUS_EXP = "ku_all_minus_dev_exp"
KU_ALL_MINUS_PREV_EXP = "ku_all_minus_prev_exp"
KU_ALL_MINUS_COMM_EXP = "ku_all_minus_comm_exp"
KU_ALL_MINUS_PROJ_PROF = "ku_all_minus_proj_prof"
KU_ALL_MINUS_PREV_PROJ = "ku_all_minus_prev_proj_prof"

KU_CUR = "ku_present_proj_exp"
KU_CUR_XIN = "ku_present_proj_exp_xin_all"


KU_DEV_EXP_AUTO     = "dev_exp_prof_auto"
KU_OTHER_EXP_AUTO    = "prev_exp_prof_auto"
KU_COMMUNITY_AUTO    = "com_exp_prof_auto"
PROJECT_PROF_AUTO    = "proj_prof_auto"
PREV_PROJ_PROF_AUTO  = "prev_proj_prof_auto"
KU_ALL_AUTO          = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

KU_ALL_MINUS_EXP_AUTO = "ku_all_minus_dev_exp_auto"
KU_ALL_MINUS_PREV_EXP_AUTO = "ku_all_minus_prev_exp_auto"
KU_ALL_MINUS_COMM_EXP_AUTO = "ku_all_minus_comm_exp_auto"
KU_ALL_MINUS_PROJ_PROF_AUTO = "ku_all_minus_proj_prof_auto"
KU_ALL_MINUS_PREV_PROJ_AUTO = "ku_all_minus_prev_proj_prof_auto"

KU_CUR_AUTO = "ku_present_proj_exp_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"
KU_DEV_EXP_XIN_AUTO = "ku_dev_exp_xin_all_auto"

all_feature_model = c(
  KU_DEV_EXP,
  KU_OTHER_EXP,
  KU_COMMUNITY,
  PROJECT_PROF,
  PREV_PROJ_PROF,
  KU_ALL ,
  XIN_FEAT,
  KU_ALL_XIN_ALL,
  KU_ALL_MINUS_EXP,
  KU_ALL_MINUS_PREV_EXP,
  KU_ALL_MINUS_COMM_EXP,
  KU_ALL_MINUS_PROJ_PROF,
  KU_ALL_MINUS_PREV_PROJ,
  KU_CUR,
  KU_CUR_XIN
)

auto_model = c(
  KU_DEV_EXP_AUTO,
  KU_OTHER_EXP_AUTO,
  KU_COMMUNITY_AUTO,
  PROJECT_PROF_AUTO,
  PREV_PROJ_PROF_AUTO,
  KU_ALL_AUTO,
  XIN_FEAT_AUTO,
  KU_ALL_XIN_ALL_AUTO,
  KU_ALL_MINUS_EXP_AUTO,
  KU_ALL_MINUS_PREV_EXP_AUTO,
  KU_ALL_MINUS_COMM_EXP_AUTO,
  KU_ALL_MINUS_PROJ_PROF_AUTO,
  KU_ALL_MINUS_PREV_PROJ_AUTO,
  KU_CUR_AUTO,
  KU_CUR_XIN_AUTO
)

KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"


hyper_parameter_model = c(
  KU_ALL_XIN_ALL_AUTO,
  KU_CUR_XIN_AUTO
)

fig_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/sk-fig/"
sk_rank_dir       = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/sk_rank/"
model_result_file  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/full_model_result.csv"

model_result_data = read.csv(file= model_result_file, header=TRUE,sep=",")

year_list <- c('LTC-1', 'LTC-2', 'LTC-3')
scaleFUN <- function(x) sprintf("%.2f", x)

###################
model_result_data


################## RQ1 FIGURE ##################

rq1_model_list = c(KU_ALL_AUTO, XIN_FEAT_AUTO)
data = model_result_data %>% filter(model_name %in% rq1_model_list)
sk_rank_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/sk_rank/sk_rank_result_rq1.csv'
rank_data = read.csv(file= sk_rank_file, header=TRUE,sep=",")
distribution_with_rank = merge(data, rank_data, by = c("model_name", "year"))

distribution_with_rank$year =  str_replace(distribution_with_rank$year, "1", "LTC-1")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "2", "LTC-2")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "3", "LTC-3")
distribution_with_rank$rank <- sub("^", "Rank-", distribution_with_rank$rank )
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "ku_all_dim_prof_auto", "KULTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "xin_feature_auto", "BAOLTC")

meds <- distribution_with_rank %>% group_by(model_name, year, rank) %>% dplyr::summarise(med = median(auc))

gg_fig <- ggplot(distribution_with_rank, aes(model_name, auc)) +    # Create facet plot again
  geom_boxplot() +
  scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0.5,1)) + 
  ylab("AUC") +
  xlab("Models") +
  theme_bw() + theme(text = element_text(size=16))

gg_fig = gg_fig + facet_nested(~year+rank, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5)

gg_fig_year = gg_fig + facet_nested(~year, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5)

ggsave(gg_fig,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-rq1'), width = 16, height = 6)
ggsave(gg_fig_year,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-rq1-year'), width = 12, height = 6)


d <- distribution_with_rank %>% dplyr::select(model_name, year, boot_id, auc)
data_wide <- spread(d, model_name, auc)

data_wide$change <- 100.0 * (data_wide$KULTC - 0.70)/(1-0.70)


data_wide$change <- 100.0 * (data_wide$KULTC - data_wide$BAOLTC)/(1 - data_wide$BAOLTC)

data_1_change <- mean(data_wide[data_wide$year=='LTC-1',]$change)
data_2_change <- mean(data_wide[data_wide$year=='LTC-2',]$change)
data_3_change <- mean(data_wide[data_wide$year=='LTC-3',]$change)




scaleFUN <- function(x) sprintf("%.2f", x)
fac <- with(distribution_with_rank, reorder(model_name, -auc, median, order = TRUE))
distribution_with_rank$model_name <- factor(distribution_with_rank$model_name, levels = levels(fac))


reorder_within <- function(x, by, within, fun = mean, sep = "___", ...) {
  new_x <- paste(x, within, sep = sep)
  stats::reorder(new_x, by, FUN = fun)
}

scale_x_reordered <- function(..., sep = "___") {
  reg <- paste0(sep, ".+$")
  ggplot2::scale_x_discrete(labels = function(x) gsub(reg, "", x), ...)
}

meds <- distribution_with_rank %>% group_by(model_name, year, rank) %>% dplyr::summarise(med = median(auc))


p <- ggplot(distribution_with_rank, aes(x = model_name, y = auc)) + 
  geom_boxplot(width = 5) + 
  #scale_x_reordered()+
  facet_wrap(.~year,  scales = "free_x") 
  #+ 
  #geom_text(data = meds, aes(y = med, label = round(med)),size = 5, vjust = -1.0, hjust = 0.5) 
  
ggsave(p,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-rq1-year-update'), width = 12, height = 6)

#ggsave(gg_fig,filename = sprintf("%s%s.png",fig_path,'sk-rank-model-rq1'), width = 16, height = 6)

################## RQ2 FIGURE ##################
rq2_model_list = c(
  KU_ALL_AUTO,
  KU_ALL_MINUS_EXP_AUTO,
  KU_ALL_MINUS_PREV_EXP_AUTO,
  KU_ALL_MINUS_COMM_EXP_AUTO,
  KU_ALL_MINUS_PROJ_PROF_AUTO,
  KU_ALL_MINUS_PREV_PROJ_AUTO
)
data = model_result_data %>% filter(model_name %in% rq2_model_list)
sk_rank_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/sk_rank/sk_rank_result_rq2.csv'
rank_data = read.csv(file= sk_rank_file, header=TRUE,sep=",")
distribution_with_rank = merge(data, rank_data, by = c("model_name", "year"))

distribution_with_rank$year =  str_replace(distribution_with_rank$year, "1", "LTC-1")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "2", "LTC-2")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "3", "LTC-3")
distribution_with_rank$rank <- sub("^", "Rank-", distribution_with_rank$rank )

distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_AUTO, "KULTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_MINUS_EXP_AUTO, "KULTC-DEV_PREV_EXP")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_MINUS_PREV_EXP_AUTO, "KULTC-DEV_EXP")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_MINUS_COMM_EXP_AUTO, "KULTC-COLLAB_EXP")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_MINUS_PROJ_PROF_AUTO, "KULTC-PROJ_PROF")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_MINUS_PREV_PROJ_AUTO, "KULTC-PREV_PROJ")

meds <- distribution_with_rank %>% group_by(model_name, year, rank) %>% dplyr::summarise(med = median(auc))

gg_fig <- ggplot(distribution_with_rank, aes(model_name, auc)) +    # Create facet plot again
  geom_boxplot() +
  scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0.5,1)) + 
  ylab("AUC") +
  xlab("Models") +
  theme_bw() + theme(text = element_text(size=16)) + scale_x_discrete(guide = guide_axis(angle = 90))

gg_fig = gg_fig + facet_nested(~year+rank, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5)

ggsave(gg_fig,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-rq2-updated'), width = 16, height = 6)
#ggsave(gg_fig,filename = sprintf("%s%s.png",fig_path,'sk-rank-model-rq2'), width = 16, height = 6)


d <- distribution_with_rank %>% dplyr::select(model_name, year, boot_id, auc)
data_wide <- spread(d, model_name, auc)
data_wide$change_dev_exp <- 100.0 * (data_wide$`KULTC-DEV_EXP` - data_wide$KULTC)/data_wide$`KULTC-DEV_EXP`

data_1_change_dev_exp <- mean(data_wide[data_wide$year=='LTC-1',]$change_dev_exp)
data_2_change_dev_exp <- mean(data_wide[data_wide$year=='LTC-2',]$change_dev_exp)
data_3_change_dev_exp <- mean(data_wide[data_wide$year=='LTC-3',]$change_dev_exp)


data_wide$change_proj <- 100.0 * (data_wide$`KULTC-PROJ_PROF` - data_wide$KULTC)/data_wide$`KULTC-PROJ_PROF`

data_1_change_proj <- mean(data_wide[data_wide$year=='LTC-1',]$change_proj)
data_2_change_proj <- mean(data_wide[data_wide$year=='LTC-2',]$change_proj)
data_3_change_proj <- mean(data_wide[data_wide$year=='LTC-3',]$change_proj)

################## RQ3 FIGURE ##################
rq3_model_list = c(KU_ALL_AUTO, XIN_FEAT_AUTO, KU_ALL_XIN_ALL_AUTO)
data = model_result_data %>% filter(model_name %in% rq3_model_list)
sk_rank_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/sk_rank/sk_rank_result_rq3.csv'
rank_data = read.csv(file= sk_rank_file, header=TRUE,sep=",")
distribution_with_rank = merge(data, rank_data, by = c("model_name", "year"))

distribution_with_rank$year =  str_replace(distribution_with_rank$year, "1", "LTC-1")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "2", "LTC-2")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "3", "LTC-3")
distribution_with_rank$rank <- sub("^", "Rank-", distribution_with_rank$rank )
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "ku_all_dim_prof_auto", "KULTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "xin_feature_auto", "BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_XIN_ALL_AUTO, "KULTC+BAOLTC")

meds <- distribution_with_rank %>% group_by(model_name, year, rank) %>% dplyr::summarise(med = median(auc))

gg_fig <- ggplot(distribution_with_rank, aes(model_name, auc)) +    # Create facet plot again
  geom_boxplot() +
  scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0.5,1)) + 
  ylab("AUC") +
  xlab("Models") +
  theme_bw() + theme(text = element_text(size=16))

gg_fig = gg_fig + facet_nested(~year+rank, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5)

ggsave(gg_fig,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-rq3'), width = 16, height = 6)
#ggsave(gg_fig,filename = sprintf("%s%s.png",fig_path,'sk-rank-model-rq3'), width = 16, height = 6)

d <- distribution_with_rank %>% dplyr::select(model_name, year, boot_id, auc)
data_wide <- spread(d, model_name, auc)
data_wide$change_baoltc <- 100.0 * (data_wide$`KULTC+BAOLTC` - data_wide$BAOLTC)/(1-data_wide$BAOLTC)

data_1_change <- mean(data_wide[data_wide$year=='LTC-1',]$change_baoltc)
data_2_change <- mean(data_wide[data_wide$year=='LTC-2',]$change_baoltc)
data_3_change <- mean(data_wide[data_wide$year=='LTC-3',]$change_baoltc)


data_wide$change_kultc <- 100.0 * (data_wide$`KULTC+BAOLTC` - data_wide$KULTC)/(1-data_wide$KULTC)

data_1_change_kultc <- mean(data_wide[data_wide$year=='LTC-1',]$change_kultc)
data_2_change_kultc <- mean(data_wide[data_wide$year=='LTC-2',]$change_kultc)
data_3_change_kultc <- mean(data_wide[data_wide$year=='LTC-3',]$change_kultc)

################## RQ4 FIGURE ##################
rq4_model_list = c(KU_ALL_XIN_ALL_AUTO,
                   'ku_all_xin_all_auto_hpt_rf',
                   'ku_all_xin_all_auto_hpt_dtree',
                   'ku_all_xin_all_auto_hpt_knn',
                   'ku_all_xin_all_auto_hpt_NB',
                   'ku_all_xin_all_auto_hpt_XGB',
                   'ku_all_xin_all_auto_hpt_LGBM'
                   )
data = model_result_data %>% filter(model_name %in% rq4_model_list)
sk_rank_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/sk_rank/sk_rank_result_rq4_f1.csv'
rank_data = read.csv(file= sk_rank_file, header=TRUE,sep=",")
distribution_with_rank = merge(data, rank_data, by = c("model_name", "year"))

distribution_with_rank$year =  str_replace(distribution_with_rank$year, "1", "LTC-1")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "2", "LTC-2")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "3", "LTC-3")
distribution_with_rank$rank <- sub("^", "Rank-", distribution_with_rank$rank )

distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "ku_all_xin_all_auto_hpt_rf", "HPT_RF_KULTC+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "ku_all_xin_all_auto_hpt_dtree", "HPT_DT_KULTC+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "ku_all_xin_all_auto_hpt_knn", "HPT_KNN_KULTC+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "ku_all_xin_all_auto_hpt_NB", "HPT_NB_KULTC+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "ku_all_xin_all_auto_hpt_XGB", "HPT_XGB_KULTC+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "ku_all_xin_all_auto_hpt_LGBM", "HPT_LGBM_KULTC+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_XIN_ALL_AUTO, "KULTC+BAOLTC")

meds <- distribution_with_rank %>% group_by(model_name, year, rank) %>% dplyr::summarise(med = median(auc))

gg_fig <- ggplot(distribution_with_rank, aes(model_name, auc)) +    # Create facet plot again
  geom_boxplot() +
  scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0.5,1)) + 
  ylab("AUC") +
  xlab("Models") +
  theme_bw() + theme(text = element_text(size=16)) + scale_x_discrete(guide = guide_axis(angle = 90))

gg_fig = gg_fig + facet_nested(~year+rank, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5)

ggsave(gg_fig,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-rq4-f1'), width = 20, height = 6)
#ggsave(gg_fig,filename = sprintf("%s%s.png",fig_path,'sk-rank-model-rq4-f1'), width = 20, height = 6)

################## RQ5 FIGURE ##################
library(grid)
model_result_data = read.csv(file= model_result_file, header=TRUE,sep=",")
rq5_model_list_old = c(KU_ALL_AUTO,
                   XIN_FEAT_AUTO,
                   KU_ALL_XIN_ALL_AUTO,
                   'ku_present_proj_exp_xin_all_auto',
                   'ku_present_proj_exp_xin_all_auto_hpt_rf',
                   'ku_present_proj_exp_xin_all_auto_hpt_dtree',
                   'ku_present_proj_exp_xin_all_auto_hpt_knn',
                   'ku_present_proj_exp_xin_all_auto_hpt_NB',
                   'ku_present_proj_exp_xin_all_auto_hpt_XGB',
                   'ku_present_proj_exp_xin_all_auto_hpt_LGBM'
)

rq5_model_list = c(KU_ALL_AUTO,
                   XIN_FEAT_AUTO,
                   KU_ALL_XIN_ALL_AUTO,
                   'ku_dev_exp_xin_all_auto'
)


data = model_result_data %>% filter(model_name %in% rq5_model_list)
sk_rank_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/sk_rank/sk_rank_result_rq5-f2.csv'
rank_data = read.csv(file= sk_rank_file, header=TRUE,sep=",")
distribution_with_rank = merge(data, rank_data, by = c("model_name", "year"))

distribution_with_rank$year =  str_replace(distribution_with_rank$year, "1", "LTC-1")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "2", "LTC-2")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "3", "LTC-3")
distribution_with_rank$rank <- sub("^", "Rank-", distribution_with_rank$rank )
  

#distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_rf', "HPT_RF::KULTC_DEV_EXP+BAOLTC")
#distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_dtree', "HPT_DT::KULTC_DEV_EXP+BAOLTC")
#distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_knn', "HPT_KNN::KULTC_DEV_EXP+BAOLTC")
#distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_NB' , "HPT_NB::KULTC_DEV_EXP+BAOLTC")
#distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_XGB', "HPT_XGB::KULTC_DEV_EXP+BAOLTC")
#distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_LGBM', "HPT_LGBM::KULTC_DEV_EXP+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto', "KULTC_DEV_EXP+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_AUTO, "KULTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, XIN_FEAT_AUTO, "BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_XIN_ALL_AUTO, "KULTC+BAOLTC")

distribution_with_rank <- distribution_with_rank %>%
  mutate(color = ifelse(model_name=="KULTC_DEV_EXP+BAOLTC", "blue", "black"))


d <- distribution_with_rank %>% dplyr::select(model_name, year, boot_id, auc)
data_wide <- spread(d, model_name, auc)
data_wide$change_baoltc <- 100.0 * (data_wide$`KULTC_DEV_EXP+BAOLTC` - data_wide$`BAOLTC`)/(1-data_wide$`BAOLTC`)

data_1_change <- mean(data_wide[data_wide$year=='LTC-1',]$change_baoltc)
data_2_change <- mean(data_wide[data_wide$year=='LTC-2',]$change_baoltc)
data_3_change <- mean(data_wide[data_wide$year=='LTC-3',]$change_baoltc)

meds <- distribution_with_rank %>% group_by(model_name, year, rank) %>% dplyr::summarise(med = median(auc))
a <- ifelse(distribution_with_rank$model_name=="KULTC_DEV_EXP+BAOLTC", "blue", "red")

gg_fig <- ggplot(distribution_with_rank, aes(model_name, auc)) +    # Create facet plot again
  geom_boxplot() +
  scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0.5,1)) + 
  ylab("AUC") +
  xlab("Models") +
  theme_bw() + theme(text = element_text(size=16)) + scale_x_discrete(guide = guide_axis(angle = 90)) 
  #+ theme(axis.text.x = element_text(color = a)) 

gg_fig = gg_fig + facet_nested(~year+rank, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5) 

ggsave(gg_fig,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-rq5-update-2'), width = 16, height = 8)
#ggsave(gg_fig,filename = sprintf("%s%s.png",fig_path,'sk-rank-model-rq5'), width = 20, height = 8)

################## REBUTTAL ##################
library(grid)
model_result_data = read.csv(file= model_result_file, header=TRUE,sep=",")
rq5_model_list_old = c(KU_ALL_AUTO,
                       XIN_FEAT_AUTO,
                       KU_ALL_XIN_ALL_AUTO,
                       'ku_present_proj_exp_xin_all_auto',
                       'ku_present_proj_exp_xin_all_auto_hpt_rf',
                       'ku_present_proj_exp_xin_all_auto_hpt_dtree',
                       'ku_present_proj_exp_xin_all_auto_hpt_knn',
                       'ku_present_proj_exp_xin_all_auto_hpt_NB',
                       'ku_present_proj_exp_xin_all_auto_hpt_XGB',
                       'ku_present_proj_exp_xin_all_auto_hpt_LGBM'
)

rq5_model_list = c(KU_ALL_AUTO,
                   XIN_FEAT_AUTO,
                   KU_ALL_XIN_ALL_AUTO,
                   'ku_dev_exp_xin_all_auto',
                   KU_DEV_EXP_AUTO
)


data = model_result_data %>% filter(model_name %in% rq5_model_list)
sk_rank_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/sk_rank/sk_rank_result_rq6-f2.csv'
rank_data = read.csv(file= sk_rank_file, header=TRUE,sep=",")
distribution_with_rank = merge(data, rank_data, by = c("model_name", "year"))

distribution_with_rank$year =  str_replace(distribution_with_rank$year, "1", "LTC-1")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "2", "LTC-2")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "3", "LTC-3")
distribution_with_rank$rank <- sub("^", "Rank-", distribution_with_rank$rank )


#distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_rf', "HPT_RF::KULTC_DEV_EXP+BAOLTC")
#distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_dtree', "HPT_DT::KULTC_DEV_EXP+BAOLTC")
#distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_knn', "HPT_KNN::KULTC_DEV_EXP+BAOLTC")
#distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_NB' , "HPT_NB::KULTC_DEV_EXP+BAOLTC")
#distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_XGB', "HPT_XGB::KULTC_DEV_EXP+BAOLTC")
#distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_LGBM', "HPT_LGBM::KULTC_DEV_EXP+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto', "KULTC_DEV_EXP+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_AUTO, "KULTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, XIN_FEAT_AUTO, "BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_XIN_ALL_AUTO, "KULTC+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_DEV_EXP_AUTO, "KULTC_DEV_EXP")

distribution_with_rank <- distribution_with_rank %>%
  mutate(color = ifelse(model_name=="KULTC_DEV_EXP+BAOLTC", "blue", "black"))


d <- distribution_with_rank %>% dplyr::select(model_name, year, boot_id, auc)

data_wide <- spread(d, model_name, auc)
data_wide$change_baoltc <- 100.0 * (data_wide$`KULTC_DEV_EXP+BAOLTC` - data_wide$`BAOLTC`)/(1-data_wide$`BAOLTC`)

data_1_change <- mean(data_wide[data_wide$year=='LTC-1',]$change_baoltc)
data_2_change <- mean(data_wide[data_wide$year=='LTC-2',]$change_baoltc)
data_3_change <- mean(data_wide[data_wide$year=='LTC-3',]$change_baoltc)

meds <- distribution_with_rank %>% group_by(model_name, year, rank) %>% dplyr::summarise(med = median(auc))
a <- ifelse(distribution_with_rank$model_name=="KULTC_DEV_EXP+BAOLTC", "blue", "red")

gg_fig <- ggplot(distribution_with_rank, aes(model_name, auc)) +    # Create facet plot again
  geom_boxplot() +
  scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0.5,1)) + 
  ylab("AUC") +
  xlab("Models") +
  theme_bw() + theme(text = element_text(size=16)) + scale_x_discrete(guide = guide_axis(angle = 90)) 
#+ theme(axis.text.x = element_text(color = a)) 

gg_fig = gg_fig + facet_nested(~year+rank, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5) 

ggsave(gg_fig,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-rq6-rebuttal'), width = 16, height = 8)
#ggsave(gg_fig,filename = sprintf("%s%s.png",fig_path,'sk-rank-model-rq5'), width = 20, height = 8)

###########################################################


p_grob <- ggplotGrob(gg_fig)

# Function to recolor specific labels
recolor_axis_text <- function(grob, label_color_map) {
  if (grob$name == "axis-b-1-1") {
    print(grob$name)
    # Assuming the first x-axis is the one we're interested in (adjust as needed)
    children <- grob$children[[1]]$children
    for (i in seq_along(children)) {
      if (children[[i]]$name == "axis") {
        text_grob <- children[[i]]$children[[1]]  # Text grob is the first child
        labels <- text_grob$label
        colors <- sapply(labels, function(label) label_color_map[[label]] %||% "black")
        text_grob$gp$col <- colors
        children[[i]]$children[[1]] <- text_grob
      }
    }
  }
  grob
}

distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_rf', "HPT_RF::KULTC_DEV_EXP+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_dtree', "HPT_DT::KULTC_DEV_EXP+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_knn', "HPT_KNN::KULTC_DEV_EXP+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_NB' , "HPT_NB::KULTC_DEV_EXP+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_XGB', "HPT_XGB::KULTC_DEV_EXP+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto_hpt_LGBM', "HPT_LGBM::KULTC_DEV_EXP+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_dev_exp_xin_all_auto', "KULTC_DEV_EXP+BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_AUTO, "KULTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, XIN_FEAT_AUTO, "BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_XIN_ALL_AUTO, "KULTC+BAOLTC")

# Define colors for specific labels
label_colors <- list("HPT_RF::KULTC_DEV_EXP+BAOLTC" = "black", 
                     "HPT_DT::KULTC_DEV_EXP+BAOLTC" = "black",
                     "HPT_KNN::KULTC_DEV_EXP+BAOLTC" = "black",
                     "HPT_NB::KULTC_DEV_EXP+BAOLTC" = "black",
                     "HPT_XGB::KULTC_DEV_EXP+BAOLTC" = "black",
                     "HPT_LGBM::KULTC_DEV_EXP+BAOLTC" = "black",
                     "KULTC_DEV_EXP+BAOLTC" = "blue",
                     "KULTC" = "black",
                     "BAOLTC" = "blue",
                     "KULTC+BAOLTC" = "black"
                     )

# Apply the recoloring function
p_grob <- grid::editGrob(p_grob, gp=gpar(col="black"), grep=TRUE, global=TRUE, 
                         grob_fun=recolor_axis_text, label_color_map=label_colors)

gg_fig = gg_fig + facet_nested(~year+rank, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5) +
  theme(axis.text.x = element_blank())

p <- gg_fig + geom_text(data = data.frame(x = unique(distribution_with_rank$model_name), y = -Inf), 
                   aes(label = x, x = x, y = y), 
                   vjust = -1, color = c("red", "blue"), inherit.aes = FALSE)

ggsave(p,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-rq5-update_grob'), width = 22, height = 8)

################################################################################ 
######## SMOTE ANALYSIS #######
sm_model_list = c(
  KU_ALL_AUTO,
  XIN_FEAT_AUTO,
  KU_ALL_XIN_ALL_AUTO,
  "ku_all_dim_prof_auto_hpt_rf_somte",
  "ku_all_dim_prof_auto_hpt_rf_somte_tomek",
  'xin_feature_auto_hpt_rf_somte',
  'xin_feature_auto_hpt_rf_somte_tomek',
  'ku_all_xin_all_auto_hpt_rf_somte',
  'ku_all_xin_all_auto_hpt_rf_somte_tomek'
)

data = model_result_data %>% filter(model_name %in% sm_model_list)
sk_rank_file = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/sk_rank/sk_rank_result_sm_analysis.csv'
rank_data = read.csv(file= sk_rank_file, header=TRUE,sep=",")
distribution_with_rank = merge(data, rank_data, by = c("model_name", "year"))

distribution_with_rank$year =  str_replace(distribution_with_rank$year, "1", "LTC-1")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "2", "LTC-2")
distribution_with_rank$year =  str_replace(distribution_with_rank$year, "3", "LTC-3")
distribution_with_rank$rank <- sub("^", "Rank-", distribution_with_rank$rank )


distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, "ku_all_dim_prof_auto_hpt_rf_somte", "KULTC_SM")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_all_dim_prof_auto_hpt_rf_somte_tomek', "KULTC_SMT")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'xin_feature_auto_hpt_rf_somte', "BAOLTC_SM")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'xin_feature_auto_hpt_rf_somte_tomek' , "BAOLTC_SMT")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_all_xin_all_auto_hpt_rf_somte', "KULTC+BAOLTC-SM")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, 'ku_all_xin_all_auto_hpt_rf_somte_tomek', "KULTC+BAOLTC-SMT")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name,  KU_ALL_AUTO, "KULTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, XIN_FEAT_AUTO, "BAOLTC")
distribution_with_rank$model_name =  str_replace(distribution_with_rank$model_name, KU_ALL_XIN_ALL_AUTO, "KULTC+BAOLTC")



meds <- distribution_with_rank %>% group_by(model_name, year, rank) %>% dplyr::summarise(med = median(auc))

gg_fig <- ggplot(distribution_with_rank, aes(model_name, auc)) +    # Create facet plot again
  geom_boxplot() +
  scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0.5,1)) + 
  ylab("AUC") +
  xlab("Models") +
  theme_bw() + theme(text = element_text(size=16)) + scale_x_discrete(guide = guide_axis(angle = 90))

gg_fig = gg_fig + facet_nested(~year+rank, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5)

ggsave(gg_fig,filename = sprintf("%s%s.pdf",fig_path,'sk-rank-model-sm-analysis'), width = 20, height = 8)
#ggsave(gg_fig,filename = sprintf("%s%s.png",fig_path,'sk-rank-model-rq5'), width = 20, height = 8)


########################################################################
### Feature Rank
KU_ALL_AUTO          = "ku_all_dim_prof_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"
KU_CUR_XIN_AUTO = "ku_present_proj_exp_xin_all_auto"

model_list = c(KU_ALL_AUTO,KU_ALL_XIN_ALL_AUTO,KU_CUR_XIN_AUTO)

sk_feature_fig_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/feature-analysis/sk-figure/"
sk_feature_rank_dir       = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/feature-analysis/sk-result/"
feature_shap_dir  = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/feature-analysis/sk-rank-input/"

year_list <- c('LTC-1', 'LTC-2', 'LTC-3')
scaleFUN <- function(x) sprintf("%.2f", x)

for (m in model_list){
  distribution_with_rank = NULL
  for (year_value in c(1,2,3)){
    file_name = sprintf("%s%s_sk_input_year_%d.txt",feature_shap_dir,m,year_value)
    data = read.csv(file_name, sep = " ",header=FALSE)
    end_ind = sprintf("V%d",ncol(data))
    data_long <- gather(data, temp, shap_value, V2:end_ind, factor_key=TRUE)
    colnames(data_long) <- c('feature_name', 'temp', 'shap_value')
    sk_rank_file = sprintf("%ssk_result_%s_%d.csv",sk_feature_rank_dir,m,year_value)
    rank_data = read.csv(file= sk_rank_file, header=TRUE,sep=",")

    d = merge(data_long, rank_data, by = c("feature_name"))
    d$year = year_value
    distribution_with_rank <- rbind(distribution_with_rank,d)
  }
  distribution_with_rank$year =  str_replace(distribution_with_rank$year, "1", "LTC-1")
  distribution_with_rank$year =  str_replace(distribution_with_rank$year, "2", "LTC-2")
  distribution_with_rank$year =  str_replace(distribution_with_rank$year, "3", "LTC-3")
  distribution_with_rank$rank <- sub("^", "Rank-", distribution_with_rank$rank )
  
  distribution_with_rank$feature_name =  str_replace(distribution_with_rank$feature_name, 'KU_CUR_PROJ', "KU_PROJ")
  
  distribution_with_rank$feature_name =  str_replace(distribution_with_rank$feature_name, 'KU', "KULTC")
  distribution_with_rank$feature_name =  str_replace(distribution_with_rank$feature_name, 'BAO', "BAOLTC")
  
  bx = boxplot(shap_value ~ feature_name, data = distribution_with_rank)
  distribution_with_rank <- distribution_with_rank[!(distribution_with_rank$shap_value %in% bx$out), ]
  
  
  meds <- distribution_with_rank %>% group_by(feature_name, year, rank) %>% dplyr::summarise(med = median(shap_value))
  
  gg_fig <- ggplot(distribution_with_rank, aes(feature_name, shap_value)) +    # Create facet plot again
    geom_boxplot() +
    scale_y_continuous(n.breaks = 5,labels = scaleFUN) + 
    ylab("Sum of Absolute Shap Values ") +
    xlab("Feature Dimension") +
    theme_bw() + theme(text = element_text(size=16)) + scale_x_discrete(guide = guide_axis(angle = 90))
  
  gg_fig = gg_fig + facet_nested(~year+rank, scales = "free_x", independent = "x") 
    #+
    #geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5)
  
  ggsave(gg_fig,filename = sprintf("%s%s_updated.pdf",sk_feature_fig_path,m), width = 18, height = 8)
}

########################################################################
KU_ALL          = "ku_all_dim_prof"
XIN_FEAT        = "xin_feature"
KU_ALL_XIN_ALL  = "ku_all_xin_all"

KU_ALL_AUTO          = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

# ltc_1_com_exp_prof_100.csv
# ltc_1_com_exp_prof_auto_100.csv


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

#######################

library(ggplot)

dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/feature_importance_reb/'
dir_data = NULL

for (year_value in c(1,2,3)){
  f = sprintf("%syear_%d.csv",dir,year_value)
  d = read.csv(file= f, header=TRUE,sep=",")
  dir_data = rbind(dir_data,d)
}
  

dir_data['ratio'] = dir_data['pos']/abs(dir_data['neg'])
dir_data['t'] = dir_data['pos'] > abs(dir_data['neg'])

meds <- dir_data %>% group_by(year, dim) %>% dplyr::summarise(med = median(ratio))

gg_fig <- ggplot(dir_data, aes(dim, ratio)) +    # Create facet plot again
  geom_boxplot() +
  scale_y_continuous(n.breaks = 5,labels = scaleFUN, limits = c(0.5,1)) + 
  ylab("AUC") +
  xlab("Models") +
  theme_bw() + theme(text = element_text(size=16)) + scale_x_discrete(guide = guide_axis(angle = 90))

gg_auc_models_without_auto = gg_auc_models_without_auto + facet_nested(~ltc+rank, scales = "free_x", independent = "x") +
  geom_text(data = meds, aes(y = med, label = round(med,2)),size = 5, vjust = -1.0, hjust = 0.5)