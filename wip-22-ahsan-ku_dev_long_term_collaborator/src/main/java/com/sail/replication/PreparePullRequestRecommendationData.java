package com.sail.replication;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.sail.model.ReviwerRecommendationDataLoader;
import com.sail.replication.model.ReviewerRankingModel;
import com.sail.replication.model.TrainTestDataModel;
import com.sail.util.ConstantUtil;
import com.sail.util.RecommendationEvaluation;
import com.sail.util.RecommendationUtil;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class PreparePullRequestRecommendationData {
 
    public PreparePullRequestRecommendationData() {

    }

    
    public RecommendationEvaluation baseModelReviewRecommendation(String projectName, String pullRequestPath, String pullReviewPath, String pullCommentPath,
            String pullFileChangPath) {

        ReviwerRecommendationDataLoader dataModel = new ReviwerRecommendationDataLoader(projectName, pullRequestPath, pullReviewPath,
                pullCommentPath, pullFileChangPath);
        Map<Integer, DescriptiveStatistics> meanAveragePrecision = new HashMap<Integer, DescriptiveStatistics>();
        Map<String, Set<String>> reviewerReviewExperience = new HashMap<String, Set<String>>();
        TrainTestDataModel trainTestModel = dataModel.getTrainTestSplits();

        for (String prNumber : trainTestModel.getTrainintPullRequestList()) {
            List<String> reviewerList = dataModel.getPrOnlyReviewerInfo().get(prNumber);
            for (String reviewString : reviewerList) {
                String reviewerName = reviewString.split("-")[0].trim();
                if (!reviewerReviewExperience.containsKey(reviewerName)) {
                    reviewerReviewExperience.put(reviewerName, new HashSet<String>());
                }
                reviewerReviewExperience.get(reviewerName).add(prNumber);
            }
        }
        Map<String, ArrayList<ReviewerRankingModel>> rankingResultList = new HashMap<String, ArrayList<ReviewerRankingModel>>();

        for(String prTest : trainTestModel.getTestingPullRequestList()){
            ArrayList<ReviewerRankingModel> rankReviewers = new ArrayList<ReviewerRankingModel>();
            for(String devName : reviewerReviewExperience.keySet()){
                ReviewerRankingModel rm = new ReviewerRankingModel();
                rm.setReviewerName(devName);
                rm.setScore((double)reviewerReviewExperience.get(devName).size());
                rankReviewers.add(rm);
            }
            rankReviewers.sort(new Comparator<ReviewerRankingModel>() {
                @Override
                public int compare(ReviewerRankingModel o1, ReviewerRankingModel o2) {
                    if(o1.getScore() < o2.getScore()){
                        return 1;
                    }else if (o1.getScore() > o2.getScore()){
                        return -1;
                    }
                    return 0;
                }
               });
               rankingResultList.put(prTest, rankReviewers);
               List<String> actualReviewer = dataModel.getPullRequestReviewerMap().get(prTest);
               for(String reviewerName : actualReviewer){
                if (!reviewerReviewExperience.containsKey(reviewerName)) {
                    reviewerReviewExperience.put(reviewerName, new HashSet<String>());
                }
                reviewerReviewExperience.get(reviewerName).add(prTest);

               }
        }

        RecommendationEvaluation evaluationRest = new RecommendationEvaluation(projectName);
        evaluationRest.calculateAccuracyAndMeanAveragePrecisionII(dataModel, 
        dataModel.getTrainTestSplits().getTestingPullRequestList(), 
        null, rankingResultList,
        ConstantUtil.REVIEW_REVIEW_EXP);

          
        System.out.println("Project: " + projectName);

        evaluationRest.printAccuracyResult();
        evaluationRest.printMAP();

        System.out.println(dataModel.getCandidateReviewerList().size()); 

        return evaluationRest;

    }

    public void startRecoommendation(){
        /*List<String> projectList = Arrays.asList("apache_lucene", "apache_wicket", "apache_activemq", "jruby_jruby",
                "caskdata_cdap", "apache_hbase", "apache_hive", "apache_storm", "apache_stratos", "apache_groovy",
                "elastic_elasticsearch");*/
        
        List<String> projectList = Arrays.asList("apache_activemq","apache_groovy","apache_lucene",
        "apache_hbase","apache_hive", "apache_storm","apache_wicket", "elastic_elasticsearch");
        List<RecommendationEvaluation> recommendationResultList = new ArrayList<RecommendationEvaluation>();
        for(String projectName : projectList){
            String prFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request/pr_reports_" + projectName +".csv";
            String prChagneFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_changed_files/pull_request_files_csv/" + projectName +"_files.csv";
            String prCommentFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_comments/comments_csv_files/" + projectName +"_comments_with_discussion.csv";
            String prReviewerPath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_reviewer/reviewer_csv_files/" + projectName + "_review.csv";
            RecommendationEvaluation recRes = baseModelReviewRecommendation(projectName, prFilePath, prReviewerPath, prCommentFilePath, prChagneFilePath);
            recommendationResultList.add(recRes);
        }
        String resultPath = RecommendationUtil.RESULT_DIR + "res_baseline_review_exp.csv";
        RecommendationUtil.writeRecResult(resultPath, recommendationResultList);
    }

    public static void main(String[] args) {

        String projectName = "apache_hbase";
        String prFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request/pr_reports_" + "apache_hbase" + ".csv";
        String prChagneFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_changed_files/pull_request_files_csv/apache_hbase_files.csv";
        String prCommentFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_comments/comments_csv_files/apache_hbase_comments.csv";
        String prReviewerPath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_reviewer/reviewer_csv_files/apache_hbase_review.csv";
        
        PreparePullRequestRecommendationData ob = new PreparePullRequestRecommendationData();
        ob.startRecoommendation();
        
        System.out.println("Program finishes successfully");
    }
}
