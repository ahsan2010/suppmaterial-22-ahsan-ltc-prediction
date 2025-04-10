library(dplyr)
library(stringr)

KU_DEV_EXP_AUTO     = "dev_exp_prof_auto"
KU_OTHER_EXP_AUTO    = "prev_exp_prof_auto"
KU_COMMUNITY_AUTO    = "com_exp_prof_auto"
PROJECT_PROF_AUTO    = "proj_prof_auto"
PREV_PROJ_PROF_AUTO  = "prev_proj_prof_auto"

KU_ALL_AUTO          = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

sk_rank_result = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values-updated/final_sk_results/"
sk_rank_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values-updated/second_sk_input/"
sk_feature_rank_figure <- '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values-updated/figures/'


#sk_rank_result = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/sk-result/"
#sk_rank_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/sk-input-data/"
#shap_value_output_dir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/xin-ku-all-year/"
#sk_feature_rank_figure <- '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/feature-importance-analysis/shap-values/figures/'

generate_rank_figure_updated <- function(model_name, year_value, top_n){
  sk_output_file = sprintf("%s%s/%s_sk_input_year_%s.txt",sk_rank_dir,model_name,model_name,year_value)
  
  df <- read.csv(file=sk_output_file, header=F,sep=" ")
  df_long <- gather(df, condition, rank, V2:V101, factor_key=TRUE)
  colnames(df_long) <- c("feature","cond","rank_distr")
  
  final_rank_file <- sprintf("%ssk_result_%s_%d.csv",sk_rank_result,model_name,year_value)
  sk_rank_result_data <- read.csv(final_rank_file,header=T,sep=",")
  colnames(sk_rank_result_data) <- c('feature', 'rank', 'sk_rank')
  
  med_features <- df_long %>% group_by(feature) %>% dplyr::summarise(med = median(rank_distr))
  df_median_rank <- merge(sk_rank_result_data, med_features, by.x = "feature", by.y='feature')
  df_median_rank <- df_median_rank %>% arrange(rank,med)
  df_median_rank$rank <- rank(df_median_rank$rank, ties.method = 'min')
  df_median_rank <- df_median_rank %>% arrange(rank)
  df_median_rank <- df_median_rank %>% filter(rank <= top_n)
  
  full_data <- merge(df_long, df_median_rank, by.x = "feature", by.y='feature')
  full_data_top_10 <- full_data %>% filter(rank <= top_n)
  
  df_median_rank <- df_median_rank %>% mutate(color = ifelse(grepl("KU", feature), "blue", "black"))
  
  seq_vec <- seq(1,max(full_data_top_10$rank),1)
  max_rank <- max(full_data_top_10$rank)
  
  gg <- ggplot(full_data_top_10,aes(factor(feature,levels = rev(df_median_rank$feature)),rank_distr) ) +  
    #geom_boxplot(outlier.shape = NA, color="ivory4", fill="ivory1", alpha=0.5)+ coord_flip() +
    geom_boxplot(outlier.shape = NA) + coord_flip() +
    scale_x_discrete(labels = rev(paste(df_median_rank$feature,paste('[',paste(df_median_rank$rank),']')))) + 
    xlab('Features') + ylab('Rank') + theme_bw() +
    #scale_y_continuous(breaks=seq_vec, limits = c(1,max(max_rank))) +
    theme(legend.position = 'none') + 
    theme(axis.text.y = element_text(color = rev(df_median_rank$color),size=15)) +
    theme(axis.text.x = element_text(size=15)) +
    theme(axis.title = element_text(size=20))
  
  gg <- gg + geom_jitter(shape=16, position=position_jitter(width=0.3, height = 0),size = 2, color="peru" , alpha=0.4)
  
  file_path = sprintf("%s%s-rank-figure-LTC-%s.pdf",sk_feature_rank_figure,model_name,year_value)
  ggsave(file=file_path, plot = gg,width = 14 , height = 11)
}

generate_rank_figure <- function(model_name, year_value, top_n){
  sk_output_file = sprintf("%s%s_sk_input_year_%s.txt",sk_rank_dir,model_name,year_value)
  
  df <- read.csv(file=sk_output_file, header=F,sep=" ")
  df_long <- gather(df, condition, rank, V2:V101, factor_key=TRUE)
  colnames(df_long) <- c("feature","cond","rank_distr")
  
  final_rank_file <- sprintf("%ssk_result_%s_%d.csv",sk_rank_result,model_name,year_value)
  sk_rank_result_data <- read.csv(final_rank_file,header=T,sep=",")
  
  med_features <- df_long %>% group_by(feature) %>% dplyr::summarise(med = median(rank_distr))
  df_median_rank <- merge(sk_rank_result_data, med_features, by.x = "feature", by.y='feature')
  df_median_rank <- df_median_rank %>% arrange(rank,med)
  df_median_rank$rank <- rank(df_median_rank$rank, ties.method = 'min')
  df_median_rank <- df_median_rank %>% arrange(rank)
  df_median_rank <- df_median_rank %>% filter(rank <= top_n)
  
  full_data <- merge(df_long, df_median_rank, by.x = "feature", by.y='feature')
  full_data_top_10 <- full_data %>% filter(rank <= top_n)
  
  df_median_rank <- df_median_rank %>% mutate(color = ifelse(grepl("KU", feature), "blue", "black"))
  
  seq_vec <- seq(1,max(full_data_top_10$rank),1)
  max_rank <- max(full_data_top_10$rank)
  
  gg <- ggplot(full_data_top_10,aes(factor(feature,levels = rev(df_median_rank$feature)),rank_distr) ) +  
    #geom_boxplot(outlier.shape = NA, color="ivory4", fill="ivory1", alpha=0.5)+ coord_flip() +
    geom_boxplot(outlier.shape = NA) + coord_flip() +
    scale_x_discrete(labels = rev(paste(df_median_rank$feature,paste('[',paste(df_median_rank$rank),']')))) + 
    xlab('Features') + ylab('Rank') + theme_bw() +
    #scale_y_continuous(breaks=seq_vec, limits = c(1,max(max_rank))) +
    theme(legend.position = 'none') + 
    theme(axis.text.y = element_text(color = rev(df_median_rank$color),size=15)) +
    theme(axis.text.x = element_text(size=15)) +
    theme(axis.title = element_text(size=20))
  
  gg <- gg + geom_jitter(shape=16, position=position_jitter(width=0.3, height = 0),size = 2, color="peru" , alpha=0.4)
  
  file_path = sprintf("%s%s-rank-figure-LTC-%s.pdf",sk_feature_rank_figure,model_name,year_value)
  ggsave(file=file_path, plot = gg,width = 14 , height = 11)
}

top_n = 15

for (year in 1:3){
  model_name = "ku_all_xin_all_auto"
  generate_rank_figure_updated (model_name, year, top_n)
}

for (year in 1:3){
  model_name = "ku_all_dim_prof_auto"
  generate_rank_figure_updated (model_name, year, top_n)
}
