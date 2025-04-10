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
require(randomForest)
require(caret)
require(Hmisc)
require(rms)
require(pROC)
require(RWeka)
set.seed(123)

classifiers <- vector(mode="list", length=5)
names(classifiers) <- c("RF", "NB", "Tree", "KNN", "SVM")
classifiers[["RF"]] <- "weka.classifiers.trees.RandomForest"
classifiers[["NB"]] <- "weka.classifiers.bayes.NaiveBayes"
classifiers[["Tree"]] <- "weka.classifiers.trees.J48"
classifiers[["KNN"]] <- "weka.classifiers.lazy.IBk"
classifiers[["SVM"]] <- "weka.classifiers.functions.SMO"

calc_measures <- function(cmtable){
  tp <- cmtable[1]
  tn <- cmtable[4]
  fp <- cmtable[3]
  fn <- cmtable[2]
  
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  precision0 <- tp / (tp + fp)
  recall0 <- tp / (tp + fn)
  f1_score0 <- 2 * precision0 * recall0 / (precision0 + recall0)
  precision1 <- tn / (tn + fn)
  recall1 <- tn / (tn + fp)
  f1_score1 <- 2 * precision1 * recall1 / (precision1 + recall1)
  
  mcc <- (tp * tn - fp * fn) / (sqrt(tp + fp) * sqrt(tp + fn) * sqrt(tn + fp) * sqrt(tn + fn))
  # print(mcc)
  
  return(c(accuracy, precision0, recall0, f1_score0, precision1, recall1, f1_score1, mcc))
}


predict_newcomer <- function(records, selected_features=NULL, cls="RF", fold=10, imp=FALSE, imbalance_method=NULL, outfile="results.csv"){
  print(selected_features)
  selected_records <- records[-which(names(records) %in% c("repo_id", "user_id", "language"))]
  if(!is.null(selected_features)){
    selected_records <- selected_records[which(names(selected_records) %in% c(selected_features, c("is_ltc")))]
  }
  
  varNames <- names(selected_records)
  varNames <- varNames[!varNames %in% c("is_ltc")]
  varNames1 <- paste(varNames, collapse = "+")
  rf.form <- as.formula(paste("is_ltc", varNames1, sep = " ~ "))
  
  selected_records$is_ltc <- as.factor(selected_records$is_ltc)
  
  folds <- createFolds(selected_records$is_ltc, k=10, list=TRUE, returnTrain=TRUE)
  cmtable <- NULL
  target_ltc <- NULL
  predict_ltc <- NULL
  importance_frame <- data.frame(row.names = varNames)
  results_frame <- data.frame(stringsAsFactors = FALSE)
  
  for (k in names(folds)) {
    print(paste0("running fold ", k, "\n"))
    inTrain <- folds[[k]]
    training <- selected_records[inTrain,]
    testing <- selected_records[-inTrain,]
    
    # print(table(training$is_ltc))
    
    if(imp){
      fit <- randomForest(rf.form, training, ntree=100, importance=T)
    }
    else{
      mycontrol <- NULL
      if(cls == 'RF'){
        mycontrol <- Weka_control(I=100, K=0, S=1)
      }
      Classifier <- make_Weka_classifier(classifiers[[cls]])
      fit <- Classifier(rf.form, data=training, control=mycontrol)
    }
    
    testing$predicted <- predict(fit, testing, type="prob")
    if(imp){
      temp <- data.frame(importance(fit, type=2))
      importance_frame <- cbind(importance_frame, temp)
    }
    
    prediction <- as.factor(as.numeric(testing$predicted[, 1] < .5))
    cm = confusionMatrix(data=prediction, reference=testing$is_ltc, positive="1")
    
    if(is.null(cmtable)){
      cmtable <- cm$table
      target_ltc <- testing$is_ltc
      predict_ltc <- testing$predicted[, 1]
    }else{
      cmtable <- cmtable + cm$table
      target_ltc <- c(target_ltc, testing$is_ltc)
      predict_ltc <- c(predict_ltc, testing$predicted[, 1])
    }
    fold_metrics <- calc_measures(cm$table)
    folder_auc <- auc(testing$is_ltc, testing$predicted[, 1])
    
    results_frame <- rbind(results_frame, c(fold_metrics, folder_auc))
    gc(verbose = getOption("verbose"), reset=FALSE) 
    #aucs <- auc(testing$is_ltc, testing$predicted[, 1])
  }
  
  if(!is.null(outfile)){
    colnames(results_frame) <- c("accuracy", "precision0", "recall0", "f1_score0", "precision1", "recall1", "f1_score1", "MCC", "AUC")
    write.table(results_frame, file = outfile, append=FALSE, quote=TRUE, sep = ",",
                eol = "\n", na = "NA", dec = ".", row.names = FALSE)
  }
  
  auc_value <- auc(target_ltc, predict_ltc)
  
  metrics <- calc_measures(cmtable)
  
  if(imp){
    return(list("importance"=importance_frame, "metrics"=c(metrics, auc_value)))
  }
  else{
    return(list("metrics"=c(metrics, auc_value[1])))
  } 
}


file_path = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_paper_data/data/data_year_1.csv'
data = read.csv(file= file_path, header=TRUE,sep=",")
predict_newcomer(record = data, imp = TRUE,outfile = '/home/local/SAIL/ahsan/BACKUP/ahsan_project_2022/xin_xia_code/results/rf_results_1.csv')

records = data

selected_records <- records[-which(names(records) %in% c("repo_id", "user_id", "language"))]
if(!is.null(selected_features)){
  selected_records <- selected_records[which(names(selected_records) %in% c(selected_features, c("is_ltc")))]
}

varNames <- names(selected_records)
varNames <- varNames[!varNames %in% c("is_ltc")]
varNames1 <- paste(varNames, collapse = "+")
rf.form <- as.formula(paste("is_ltc", varNames1, sep = " ~ "))

selected_records$is_ltc <- as.factor(selected_records$is_ltc)

folds <- createFolds(selected_records$is_ltc, k=10, list=TRUE, returnTrain=TRUE)
cmtable <- NULL
target_ltc <- NULL
predict_ltc <- NULL
importance_frame <- data.frame(row.names = varNames)
results_frame <- data.frame(stringsAsFactors = FALSE)

for (k in names(folds)) {
  print(paste0("running fold ", k, "\n"))
  inTrain <- folds[[k]]
  print(length(inTrain))
  training <- selected_records[inTrain,]
  testing <- selected_records[-inTrain,]
  print(sprintf("testing %d",nrow(testing)))
  fit <- randomForest(rf.form, training, ntree=100, importance=T)
  testing$predicted <- predict(fit, testing, type="prob")
  prediction <- as.factor(as.numeric(testing$predicted[, 1] < .5))
  cm = confusionMatrix(data=prediction, reference=testing$is_ltc, positive="1")
  fold_metrics <- calc_measures(cm$table)
  prediction <- as.numeric(testing$predicted[, 1] < .5)
  folder_auc <- pROC::auc(testing$is_ltc, testing$predicted[, 1])
  break
}


actual <- c(1, 1, 1, 0, 0, 0)
predicted <- c(0.9, 0.8, 0.4, 0.5, 0.3, 0.2)
auc(actual, predicted)
