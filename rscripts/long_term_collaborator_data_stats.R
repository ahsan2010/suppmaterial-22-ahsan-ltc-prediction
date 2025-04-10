library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(ggpubr)
library(scales)
library(cluster)
library(scales)
library(factoextra)
library(plotly)
library(NbClust)
library(Rtsne)
library(clusterCrit)
library(DescTools)
set.seed(123)

knowledge_unit_list <- c("data_type",
"operator_decision",
"array",
"loop",
"method_encapsulation",	
"inheritance",
"exception_handling",
"advanced_class_design",
"generic_collection",
"functional_interface",
"stream_api",
"date_time_api",
"java_io",
"java_nio",
"concurrency",
"data_base",
"localization",
"string_processing",
"persistence_object", 
"enterprise_java_bean",
"message_service",
"soap_service",
"servlet_web",
"rest_api",
"websocket_api",
"jfs_web_application",
"cdi_bean",
"enterprise_concurrency",
"batch_processing")

projects = c('elastic_elasticsearch','apache_lucene','apache_wicket','apache_activemq','jruby_jruby',
             'caskdata_cdap','apache_hbase','apache_hive','apache_storm','apache_groovy')
dev_feature_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/FeasibilityStudy/ku_feature_dev_profile/'

combine_full_data = NULL
for (project in projects){
  file_path = sprintf("%s%s_dev_ku_LTC.csv", dev_feature_path, project)
  data = read.csv(file= file_path, header=TRUE,sep=",")
  combine_full_data = rbind(combine_full_data,data)
}

path = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/FeasibilityStudy/figures/ltc_non_ltc_compare/FullDataResults/"

first_year_data = combine_full_data[,c(4:32,33)]
second_year_data = combine_full_data[,c(4:32,34)]
third_year_data = combine_full_data[,c(4:32,35)]


# min max analysis
five_num_summary_ltc_non_ltc = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/FeasibilityStudy/stats_ltc_non_ltc_kus/"
p <- wilcox.test(rq2.sanitycheck[rq2.sanitycheck$model==heu & rq2.sanitycheck$measure==measure ,]$value, rq2.sanitycheck[rq2.sanitycheck$model==real & rq2.sanitycheck$measure==measure ,]$value, paired = TRUE)$p.value
estimate <- cliff.delta(rq2.sanitycheck[rq2.sanitycheck$model==heu & rq2.sanitycheck$measure==measure ,]$value, rq2.sanitycheck[rq2.sanitycheck$model==real & rq2.sanitycheck$measure==measure ,]$value, paired = TRUE)$estimate
magnitude <- as.character(cliff.delta(rq2.sanitycheck[rq2.sanitycheck$model==heu & rq2.sanitycheck$measure==measure ,]$value, rq2.sanitycheck[rq2.sanitycheck$model==real & rq2.sanitycheck$measure==measure ,]$value, paired = TRUE)$magnitude)

# First year LTC vs Non LTC

long_first_year_data <- gather(first_year_data, knowledge_unit, ku_value, data_type:batch_processing, factor_key=TRUE)
colnames(long_first_year_data)[1] = "target"
bx = boxplot(ku_value  ~ knowledge_unit, data = long_first_year_data)
data_without_outliers <- long_first_year_data[!(long_first_year_data$ku_value %in% bx$out), ]

five_num_summary_result_first_year <- long_first_year_data %>% group_by(knowledge_unit,target) %>%
  summarise(n = n(),
            min = fivenum(ku_value)[1],
            Q1 = fivenum(ku_value)[2],
            median = fivenum(ku_value)[3],
            Q3 = fivenum(ku_value)[4],
            max = fivenum(ku_value)[5])

write.csv(five_num_summary_result_first_year, file = sprintf("%sfirst_year_stat.csv",five_num_summary_ltc_non_ltc), row.names = FALSE)

result_first_year = NULL
#wilcox test
for (ku in knowledge_unit_list){
  ku_non_ltc = long_first_year_data %>% filter(knowledge_unit == ku & target == 0)
  ku_ltc = long_first_year_data %>% filter(knowledge_unit == ku & target == 1)
  p <- wilcox.test(ku_non_ltc$ku_value, ku_ltc$ku_value)$p.value
  estimate <- cliff.delta(ku_non_ltc$ku_value, ku_ltc$ku_value)$estimate
  magnitude <- as.character(cliff.delta(ku_non_ltc$ku_value, ku_ltc$ku_value)$magnitude)
  result_first_year = rbind(result_first_year, data.frame(ku,p,estimate,magnitude, median(ku_non_ltc$ku_value), median(ku_ltc$ku_value)))
}
colnames(result_first_year) = c('knowledge_unit', 'p_value', 'estimate', 'magnitude', 'median_non_ltc', 'median_ltc')
write.csv(result_first_year,file=sprintf("%swilcox_test_first_year_ltc.csv",five_num_summary_ltc_non_ltc), row.names = FALSE)


for (ku in knowledge_unit_list){
  data_without_outliers_ku = data_without_outliers %>% filter(knowledge_unit == ku)
  gg <- ggplot(data_without_outliers_ku, aes(x=knowledge_unit, y=ku_value, fill= factor(target))) + 
    geom_boxplot()  + theme_bw()
  ggsave(plot = gg,file = sprintf("%sLTC_One_Year/%s_ltc_One_Year.png",path,ku),width = 12 , height = 10)
  ggsave(plot = gg,file = sprintf("%sLTC_One_Year/%s_ltc_One_Year.pdf",path,ku),width = 12 , height = 10)
  
}


# Second year LTC vs Non LTC

long_year_data <- gather(second_year_data, knowledge_unit, ku_value, data_type:batch_processing, factor_key=TRUE)
colnames(long_year_data)[1] = "target"
bx = boxplot(ku_value  ~ knowledge_unit, data = long_year_data)
data_without_outliers <- long_year_data[!(long_year_data$ku_value %in% bx$out), ]

five_num_summary_result_second_year <- long_year_data %>% group_by(knowledge_unit,target) %>%
  summarise(n = n(),
            min = fivenum(ku_value)[1],
            Q1 = fivenum(ku_value)[2],
            median = fivenum(ku_value)[3],
            Q3 = fivenum(ku_value)[4],
            max = fivenum(ku_value)[5])


write.csv(five_num_summary_result_second_year, file = sprintf("%ssecond_year_stat.csv",five_num_summary_ltc_non_ltc), row.names = FALSE)

result_second_year = NULL
#wilcox test
for (ku in knowledge_unit_list){
  ku_non_ltc = long_year_data %>% filter(knowledge_unit == ku & target == 0)
  ku_ltc = long_year_data %>% filter(knowledge_unit == ku & target == 1)
  p <- wilcox.test(ku_non_ltc$ku_value, ku_ltc$ku_value)$p.value
  estimate <- cliff.delta(ku_non_ltc$ku_value, ku_ltc$ku_value)$estimate
  magnitude <- as.character(cliff.delta(ku_non_ltc$ku_value, ku_ltc$ku_value)$magnitude)
  result_second_year = rbind(result_second_year, data.frame(ku,p,estimate,magnitude, median(ku_non_ltc$ku_value), median(ku_ltc$ku_value)))
}
colnames(result_second_year) = c('knowledge_unit', 'p_value', 'estimate', 'magnitude', 'median_non_ltc', 'median_ltc')
write.csv(result_second_year,file=sprintf("%swilcox_test_second_year_ltc.csv",five_num_summary_ltc_non_ltc), row.names = FALSE)


for (ku in knowledge_unit_list){
  data_without_outliers_ku = data_without_outliers %>% filter(knowledge_unit == ku)
  gg <- ggplot(data_without_outliers_ku, aes(x=knowledge_unit, y=ku_value, fill= factor(target))) + 
    geom_boxplot()  + theme_bw()
  #ggsave(plot = gg,file = sprintf("%sLTC_Two_Year/%s_ltc_One_Year.png",path,ku),width = 12 , height = 10)
  ggsave(plot = gg,file = sprintf("%sLTC_Two_Year/%s_ltc_Two_Year.pdf",path,ku),width = 12 , height = 10)
  
}



# Third year LTC vs Non LTC

long_year_data <- gather(third_year_data, knowledge_unit, ku_value, data_type:batch_processing, factor_key=TRUE)
colnames(long_year_data)[1] = "target"
bx = boxplot(ku_value  ~ knowledge_unit, data = long_year_data)
data_without_outliers <- long_year_data[!(long_year_data$ku_value %in% bx$out), ]

five_num_summary_result_third_year <- long_year_data %>% group_by(knowledge_unit,target) %>%
  summarise(n = n(),
            min = fivenum(ku_value)[1],
            Q1 = fivenum(ku_value)[2],
            median = fivenum(ku_value)[3],
            Q3 = fivenum(ku_value)[4],
            max = fivenum(ku_value)[5])

write.csv(five_num_summary_result_third_year, file = sprintf("%sthird_year_stat.csv",five_num_summary_ltc_non_ltc), row.names = FALSE)

result_third_year = NULL
#wilcox test
for (ku in knowledge_unit_list){
  ku_non_ltc = long_year_data %>% filter(knowledge_unit == ku & target == 0)
  ku_ltc = long_year_data %>% filter(knowledge_unit == ku & target == 1)
  p <- wilcox.test(ku_non_ltc$ku_value, ku_ltc$ku_value)$p.value
  estimate <- cliff.delta(ku_non_ltc$ku_value, ku_ltc$ku_value)$estimate
  magnitude <- as.character(cliff.delta(ku_non_ltc$ku_value, ku_ltc$ku_value)$magnitude)
  result_third_year = rbind(result_third_year, data.frame(ku,p,estimate,magnitude, median(ku_non_ltc$ku_value), median(ku_ltc$ku_value)))
}
colnames(result_third_year) = c('knowledge_unit', 'p_value', 'estimate', 'magnitude', 'median_non_ltc', 'median_ltc')
write.csv(result_third_year,file=sprintf("%swilcox_test_third_year_ltc.csv",five_num_summary_ltc_non_ltc), row.names = FALSE)

for (ku in knowledge_unit_list){
  data_without_outliers_ku = data_without_outliers %>% filter(knowledge_unit == ku)
  gg <- ggplot(data_without_outliers_ku, aes(x=knowledge_unit, y=ku_value, fill= factor(target))) + 
    geom_boxplot()  + theme_bw()
  #ggsave(plot = gg,file = sprintf("%sLTC_Third_Year_Year/%s_ltc_One_Year.png",path,ku),width = 12 , height = 10)
  ggsave(plot = gg,file = sprintf("%sLTC_Three_Year/%s_ltc_Three_Year.pdf",path,ku),width = 12 , height = 10)
  
}

##################


for (project in projects){
  file_path = sprintf("%s%s_dev_ku_LTC.csv", dev_feature_path, project)
  data = read.csv(file= file_path, header=TRUE,sep=",")
  
  first_year_data = data[,c(4:32,33)]
  second_year_data = data[,c(4:32,34)]
  third_year_data = data[,c(4:32,35)]
  
  long_first_year_data <- gather(first_year_data, knowledge_unit, ku_value, data_type:batch_processing, factor_key=TRUE)
  colnames(long_first_year_data)[1] = "target"
  bx = boxplot(ku_value  ~ knowledge_unit, data = long_first_year_data)
  data_without_outliers <- long_first_year_data[!(long_first_year_data$ku_value %in% bx$out), ]
  
  for (ku in knowledge_unit_list){
    data_without_outliers_ku = data_without_outliers %>% filter(knowledge_unit == ku)
    gg <- ggplot(data_without_outliers_ku, aes(x=knowledge_unit, y=ku_value, fill= factor(target))) + 
      geom_boxplot()  + theme_bw()
    ggsave(plot = gg,file = sprintf("%s/RQ2_Total_Clusters.png",file_path),width = 12 , height = 10)
    ggsave(plot = gg,file = sprintf("%s/RQ2_Total_Clusters.pdf",file_path),width = 12 , height = 10)
    
  }
  
}
  




