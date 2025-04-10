library(ggplot2)
library(effsize)

KU_DEV_EXP      = "dev_exp_prof"
KU_OTHER_EXP    = "prev_exp_prof"
KU_COMMUNITY    = "com_exp_prof"
PROJECT_PROF    = "proj_prof"
PREV_PROJ_PROF  = "prev_proj_prof"
KU_ALL          = "ku_all_dim_prof"
XIN_FEAT        = "xin_feature"
KU_ALL_XIN_ALL  = "ku_all_xin_all"

KU_ALL_AUTO = "ku_all_dim_prof_auto"
XIN_FEAT_AUTO        = "xin_feature_auto"
KU_ALL_XIN_ALL_AUTO  = "ku_all_xin_all_auto"

#ku_model_suffix  = 'ku_full_boostrap'
#ku_model_suffix = 'ku_count_boostrap'
#ku_model_suffix = 'ku_norm_boostrap'
#xin_model_suffix = 'xin_features_boostrap'
#combined_ku_xin_model_suffix = 'ku_full_xin_features_boostrap'
#combined_ku_xin_model_suffix = 'ku_count_xin_features_boostrap'
#combined_ku_xin_model_suffix = 'ku_norm_xin_features_boostrap'

#ku_model_suffix = KU_ALL
#xin_model_suffix = XIN_FEAT
#combined_ku_xin_model_suffix = ku_model_suffix


ku_model_suffix = KU_ALL_AUTO
xin_model_suffix = XIN_FEAT_AUTO
combined_ku_xin_model_suffix = ku_model_suffix

find_cliff_magnitude <- function(cliff_value){
  if (abs(cliff_value) <= 0.147){
    return ("N")
  }else if (abs(cliff_value) <= 0.33){
    return ("S")
  }else if (abs(cliff_value) <= 0.474){
    return ("M")
  }else{
    return ("L")
  }
}


#file_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/June_23_2022/'
#file_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/August_05_2022/'

#file_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/ku-different-dimension-100-bootstrap-dec-15/"

file_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/March_15_2024/results/"


#file_path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/ku-different-dimension/"
#file_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/Result/'
#file_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Sep-2023/features/Result/'

#ltc_2_ku_full_xin_features_boostrap.csv

classifier_list = c('rf','LogisticRegression','LightGBM')
classifier_id = 1


data <- read.csv(sprintf("%sfull_model_result.csv",file_path))

wilcox_result = NULL
# Year 1 analysis
year_value = 1

data_ku_model_year_1 = data %>% dplyr::filter(model_name == ku_model_suffix & year==year_value)
data_xin_model_year_1 = data %>% dplyr::filter(model_name == xin_model_suffix & year==year_value)
data_combined_ku_xin_model_year_1 = data %>% dplyr::filter(model_name == combined_ku_xin_model_suffix & year==year_value)

data_ku_model_year_1 = data_ku_model_year_1 %>% filter(algo == classifier_list[classifier_id])
data_xin_model_year_1 = data_xin_model_year_1 %>% filter(algo == classifier_list[classifier_id])
data_combined_ku_xin_model_year_1 = data_combined_ku_xin_model_year_1 %>% filter(algo == classifier_list[classifier_id])


measure_wilcox_year_1 <- wilcox.test(data_combined_ku_xin_model_year_1$auc, data_xin_model_year_1$auc,paired = TRUE)
measure_cliff_value_year_1 <- cliff.delta(data_combined_ku_xin_model_year_1$auc, data_xin_model_year_1$auc,paired = TRUE)
measure_cliff_effect_size_year_1 <- find_cliff_magnitude(measure_cliff_value_year_1$estimate)

df = data.frame(year_value,median(data_combined_ku_xin_model_year_1$auc),median(data_xin_model_year_1$auc),measure_wilcox_year_1$p.value,measure_cliff_value_year_1$estimate,measure_cliff_effect_size_year_1)
names(df) = c('ltc_year','KU_XIN_MODEL_AUC','XIN_MODEL_AUC','p_value','cliff_delta','effect_size')
wilcox_result = rbind(wilcox_result,df)

# Year 2 analysis
year_value = 2
data_ku_model_year_2 = data %>% dplyr::filter(model_name == ku_model_suffix & year==year_value)
data_xin_model_year_2 = data %>% dplyr::filter(model_name == xin_model_suffix & year==year_value)
data_combined_ku_xin_model_year_2 = data %>% dplyr::filter(model_name == combined_ku_xin_model_suffix & year==year_value)

data_ku_model_year_2 = data_ku_model_year_2 %>% filter(algo == classifier_list[classifier_id])
data_xin_model_year_2 = data_xin_model_year_2 %>% filter(algo == classifier_list[classifier_id])
data_combined_ku_xin_model_year_2 = data_combined_ku_xin_model_year_2 %>% filter(algo == classifier_list[classifier_id])

measure_wilcox_year_2 <- wilcox.test(data_combined_ku_xin_model_year_2$auc, data_xin_model_year_2$auc,paired = TRUE)
measure_cliff_value_year_2 <- cliff.delta(data_combined_ku_xin_model_year_2$auc, data_xin_model_year_2$auc,paired = TRUE)
measure_cliff_effect_size_year_2 <- find_cliff_magnitude(measure_cliff_value_year_2$estimate)

df = data.frame(year_value,median(data_combined_ku_xin_model_year_2$auc),median(data_xin_model_year_2$auc),measure_wilcox_year_2$p.value,measure_cliff_value_year_2$estimate,measure_cliff_effect_size_year_2)
names(df) = c('ltc_year','KU_XIN_MODEL_AUC','XIN_MODEL_AUC','p_value','cliff_delta','effect_size')
wilcox_result = rbind(wilcox_result, df)

# Year 3 analysis
year_value = 3

data_ku_model_year_3 = data %>% dplyr::filter(model_name == ku_model_suffix & year==year_value)
data_xin_model_year_3 = data %>% dplyr::filter(model_name == xin_model_suffix & year==year_value)
data_combined_ku_xin_model_year_3 = data %>% dplyr::filter(model_name == combined_ku_xin_model_suffix & year==year_value)

data_ku_model_year_3 = data_ku_model_year_3 %>% filter(algo == classifier_list[classifier_id])
data_xin_model_year_3 = data_xin_model_year_3 %>% filter(algo == classifier_list[classifier_id])
data_combined_ku_xin_model_year_3 = data_combined_ku_xin_model_year_3 %>% filter(algo == classifier_list[classifier_id])

measure_wilcox_year_3 <- wilcox.test(data_combined_ku_xin_model_year_3$auc, data_xin_model_year_3$auc,paired = TRUE)
measure_cliff_value_year_3 <- cliff.delta(data_combined_ku_xin_model_year_3$auc, data_xin_model_year_3$auc,paired = TRUE)
measure_cliff_effect_size_year_3 <- find_cliff_magnitude(measure_cliff_value_year_3$estimate)

df = data.frame(year_value,median(data_combined_ku_xin_model_year_3$auc),median(data_xin_model_year_3$auc),measure_wilcox_year_3$p.value,measure_cliff_value_year_3$estimate,measure_cliff_effect_size_year_3)
names(df) = c('ltc_year','KU_XIN_MODEL_AUC','XIN_MODEL_AUC','p_value','cliff_delta','effect_size')
wilcox_result = rbind(wilcox_result, df)
wilcox_result$is_significant <- wilcox_result$p_value < 0.05

####
path_year_1 = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/June_23_2022/full_result_ltc_1.csv'
path_year_2 = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/June_23_2022/full_result_ltc_2.csv'
path_year_3 = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/June_23_2022/full_result_ltc_3.csv'

data_ltc_1 = read.csv(path_year_1,header=TRUE,sep=",")
data_ltc_2 = read.csv(path_year_2,header=TRUE,sep=",")
data_ltc_3 = read.csv(path_year_3,header=TRUE,sep=",")


least_class_1 = sum(data_ltc_1$LTC_Developer_Cat_Year_One == 1)
least_class_2 = sum(data_ltc_2$LTC_Developer_Cat_Year_Two == 1)
least_class_3 = sum(data_ltc_3$LTC_Developer_Cat_Year_Three == 1)

total_ltc_1 = nrow(data_ltc_1)
total_ltc_2 = nrow(data_ltc_2)
total_ltc_3 = nrow(data_ltc_3)


###########
library(dplyr)
path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/dev_label_sep_2023.csv"
data = read.csv(path, sep = ",", header = T)

uniq_dev_list = unique(data$dev_name)
total_uniq_dev = length(uniq_dev_list)

per_project_dev_count = data %>% group_by(Project_Name) %>% summarise (tot_dev = n())
per_project_dev_count %>% arrange(desc(tot_dev))
######################################################################################




