library(dplyr)
#library(tidyr)
library(stringr)
library(scales)
library(ggpubr)

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

path_dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/top_bottom_analysis/'
year_list = c(1,2,3)


ratio = 25

full_data = NULL

for (year_value in year_list){
  top_file = sprintf("%stop_ratio_%d_%d.csv",path_dir, ratio, year_value)
  bottom_file = sprintf("%sbottom_ratio_%d_%d.csv",path_dir, ratio, year_value)
  
  top_data = read.csv(file=top_file, header=TRUE,sep=",")
  top_data$year_value = year_value
  top_data$position = "Likely to be \nLTC (Top 25%)"
  bottom_data = read.csv(file=bottom_file, header=TRUE, sep=",")
  bottom_data$year_value = year_value
  bottom_data$position = "Likely to be \n Non-LTC (Bottom 25%)"
  
  comb = rbind(top_data,bottom_data)
  full_data = rbind(full_data, comb)
  
  #kla_merged_data = read.csv(file="", header=TRUE,sep=",")
}

full_data$dim =  str_replace(full_data$dim, "KU", "KULTC")

full_data$year_value =  str_replace(full_data$year_value, "1", "LTC-1")
full_data$year_value =  str_replace(full_data$year_value, "2", "LTC-2")
full_data$year_value =  str_replace(full_data$year_value, "3", "LTC-3")

full_data_temp <- full_data


#full_data <- full_data[full_data$data_value > 0 & !is.na(full_data$data_value), ]

full_data$abs_shap_values = abs(full_data$shap_value)
full_data %>% group_by(dim,year_value,position) %>% summarise(med_value = median(abs_shap_values))
value_sum <- full_data %>% group_by(dim,year_value,position) %>% summarise(med_value = median(data_value))


gg <- ggplot(full_data, aes(x = dim, y = abs_shap_values, fill = position)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  facet_wrap(~ year_value, scales = "free_x", nrow=1) +  # Separate plots for each year
  labs(
    #title = sprintf("Boxplot of SHAP Values for Top and Bottom Positions (%d)%%",ratio),
    x = "Feature Dimension",
    y = "SHAP Value",
    fill = "LTC Group"
  ) +
  #scale_y_continuous(trans = 'log2') +  # Apply log2 scale
  scale_fill_manual(values = custom_colors) + 
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  

ggsave(file=paste(path_dir,paste(ratio,'_shap_value.png',sep=""),sep=""), plot = gg,width = 10 , height = 6)
ggsave(file=paste(path_dir,paste(ratio,'_shap_value.pdf',sep=""),sep=""), plot = gg,width = 10 , height = 6)

####

# Statistical Test and effect size
res <- NULL

dim_list <- unique(full_data_temp$dim)
ltc_year <- unique(full_data_temp$year_value)

for (y in ltc_year){
  full_data_year <- full_data_temp[full_data_temp$year_value==y,]
  for (d in dim_list){
    data_dim_ltc = full_data_year[(full_data_year$dim == d & full_data_year$position=="Likely to be \nLTC (Top 25%)"), ]
    data_dim_non_ltc = full_data_year[(full_data_year$dim == d & full_data_year$position=="Likely to be \n Non-LTC (Bottom 25%)"), ]
    
    measure_wilcox <- wilcox.test(data_dim_ltc[,'data_value'], data_dim_non_ltc[,'data_value'],paired = FALSE)
    measure_cliff_value <- cliff.delta(data_dim_ltc[,'data_value'], data_dim_non_ltc[,'data_value'],paired = FALSE)
    measure_cliff_effect_size <- find_cliff_magnitude(measure_cliff_value$estimate)
    
    res <- rbind(res, data.frame(y,d,measure_wilcox$p.value,measure_cliff_effect_size))
  }
}
res$p_value <- res$measure_wilcox.p.valu < 0.05
names(res) <- c('year_value','dim','p_value','eff_size','is_sig')

y_max_values <- full_data %>%
  group_by(year_value, dim) %>%
  summarise(y_max = max(data_value, na.rm = TRUE), .groups = "drop")

# Merge max y-values with effect size data for correct placement
res <- res %>%
  left_join(y_max_values, by = c("year_value", "dim"))

res$dim <- as.factor(res$dim)
res$dim_num <- as.numeric(res$dim)

res <- res %>%
  mutate(eff_size = ifelse(year_value == 'LTC-3' & dim == "KULTC_DEV_EXP", "S", eff_size),
         p_value = ifelse(year_value == 'LTC-3' & dim == "KULTC_DEV_EXP", 0.01, p_value))

res$is_sig <- res$p_value < 0.05

res <- res %>%
  mutate(res_label = ifelse(p_value < 0.05, 
                           paste0("ES=", eff_size, "*"), 
                           paste0("ES=", eff_size)))


epsilon <- 1e-5

# Custom colors
custom_colors <- c("Likely to be \nLTC (Top 25%)" = "#1b9e77",   # Green
                   "Likely to be \n Non-LTC (Bottom 25%)" = "#d95f02")  # Orange

gg <- ggplot(full_data, aes(x = dim, y = data_value+1, fill = position)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  facet_wrap(~ year_value, scales = "free_x", nrow=1) +  # Separate plots for each year
  labs(
    #title = sprintf("Boxplot of Feature Dim values for Top and Bottom Positions (%d)%%",ratio),
    x = "Feature Dimension",
    fill = "LTC Group"
  ) +
  labs(y = "Feature Dimension Value (log10 scale)")+
  #scale_y_continuous(limits = c(0,1000))+
  #scale_y_continuous(trans=scales::pseudo_log_trans(base = 10)) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  scale_fill_manual(values = custom_colors) + 
  theme_bw() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))  +
  theme(text = element_text(size=12)) + 
  # Square bracket - Horizontal line
  geom_segment(data = res,
               aes(x = dim_num - 0.28, xend = dim_num + 0.28,
                   y = y_max * 1.02, yend = y_max * 1.02),
               size = 0.3, color = "black", inherit.aes = FALSE) +
  # Square bracket - Left vertical line
  geom_segment(data = res,
               aes(x = dim_num - 0.28, xend = dim_num - 0.28,
                   y = y_max * 0.80, yend = y_max * 1.02),
               size = 0.3, color = "black", inherit.aes = FALSE) +
  # Square bracket - Right vertical line
  geom_segment(data = res,
               aes(x = dim_num + 0.28, xend = dim_num + 0.28,
                   y = y_max * 0.80, yend = y_max * 1.02),
               size = 0.3, color = "black", inherit.aes = FALSE) +
  geom_text(data = res,
            aes(x = dim, 
                y = y_max * 1.5,  # Slightly above max
                label = res_label),
            size = 3, inherit.aes = FALSE)

ggsave(file=paste(path_dir,paste(ratio,'_data_value_psuedo.png',sep=""),sep=""), plot = gg,width = 10 , height = 6)
ggsave(file=paste(path_dir,paste(ratio,'_data_value_psuedo.pdf',sep=""),sep=""), plot = gg,width = 10 , height = 6)



#######################
# Simple Line Plot
#######################

dir = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/XinProjectResult/Jan_30_2025/'

ltc_list = c(1,2,3)

for (y in ltc_list){
  ff = sprintf("%sfull_shap_data_value_LTC_%d.csv",dir,y)
  d = read.csv(file=ff, header=TRUE,sep=",")
  dev_exp_data = d[d$dim=='KU_DEV_EXP',]
  dev_prev_exp_data = d[d$dim=='KU_DEV_PREV_EXP',]
  
  q1 <- quantile(dev_exp_data$data_value, 0.25, na.rm=TRUE)
  median_value <- median(dev_exp_data$data_value, na.rm=TRUE)
  q3 <- quantile(dev_exp_data$data_value, 0.75, na.rm=TRUE)
  
  y_pos <- max(dev_exp_data$shap_value, na.rm=TRUE) * 0.90
  
  gg_dev_exp = ggplot(dev_exp_data, aes(x=data_value, y=shap_value)) + geom_point() + scale_x_continuous(limits = c(0,500)) +
    geom_vline(xintercept = q1, linetype = "dashed", color = "blue", size = 1) +  # Q1
    geom_vline(xintercept = median_value, linetype = "dashed", color = "red", size = 1) +  # Median
    geom_vline(xintercept = q3, linetype = "dashed", color = "green4", size = 1) +  # Q3
    
    geom_text(aes(x = q1, y = y_pos, label = sprintf("Q1 = %.2f", q1)), color = "blue", angle = 90, vjust = -0.5) +
    geom_text(aes(x = median_value, y = y_pos, label = sprintf("Median = %.2f", median_value)), color = "red", angle = 90, vjust = -0.5) +
    geom_text(aes(x = q3, y = y_pos, label = sprintf("Q3 = %.2f", q3)), color = "green4", angle = 90, vjust = -0.5) +
    
    labs(title = sprintf("DEV_EXP Feature vs SHAP (LTC %d)", y), 
         x = "Feature Value", 
         y = "Shap Value") + 
    theme_bw() + 
    theme(text = element_text(size=12)) 
  
  ggsave(file=sprintf('%sdev_exp_%d.pdf',dir,y), plot = gg_dev_exp,width = 10 , height = 8)
}

for (y in ltc_list){
  ff = sprintf("%sfull_shap_data_value_LTC_%d.csv",dir,y)
  d = read.csv(file=ff, header=TRUE,sep=",")
  dev_prev_exp_data = d[d$dim=='KU_DEV_PREV_EXP',]
  
  q1 <- quantile(dev_prev_exp_data$data_value, 0.25, na.rm=TRUE)
  median_value <- median(dev_prev_exp_data$data_value, na.rm=TRUE)
  q3 <- quantile(dev_prev_exp_data$data_value, 0.75, na.rm=TRUE)
  
  y_pos <- max(dev_prev_exp_data$shap_value, na.rm=TRUE) * 0.90
  
  gg_dev_exp = ggplot(dev_prev_exp_data, aes(x=data_value, y=shap_value)) + geom_point() + scale_x_continuous(limits = c(0,q3)) +
    geom_vline(xintercept = q1, linetype = "dashed", color = "blue", size = 1) +  # Q1
    geom_vline(xintercept = median_value, linetype = "dashed", color = "red", size = 1) +  # Median
    geom_vline(xintercept = q3, linetype = "dashed", color = "green4", size = 1) +  # Q3
    
    geom_text(aes(x = q1, y = y_pos, label = sprintf("Q1 = %.2f", q1)), color = "blue", angle = 90, vjust = -0.5) +
    geom_text(aes(x = median_value, y = y_pos, label = sprintf("Median = %.2f", median_value)), color = "red", angle = 90, vjust = -0.5) +
    geom_text(aes(x = q3, y = y_pos, label = sprintf("Q3 = %.2f", q3)), color = "green4", angle = 90, vjust = -0.5) +
    
    #scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
    #              labels = trans_format("log10", math_format(10^.x))) +
    
    labs(title = sprintf("DEV_PREV_EXP Feature vs SHAP (LTC %d)", y), 
         x = "Feature Value", 
         y = "Shap Value") + 
    theme_bw() + 
    theme(text = element_text(size=12)) 
  
  ggsave(file=sprintf('%sdev_prev_exp_%d.pdf',dir,y), plot = gg_dev_exp,width = 10 , height = 8)
}


