package com.sail.replication;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.sail.github.model.GitCommitModel;
import com.sail.model.ReviwerRecommendationDataLoader;
import com.sail.replication.model.PullRequestModel;
import com.sail.replication.model.ReviewerRankingModel;
import com.sail.replication.model.TrainTestDataModel;
import com.sail.util.ConstantUtil;
import com.sail.util.DateUtil;
import com.sail.util.RecommendationEvaluation;
import com.sail.util.RecommendationUtil;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.joda.time.DateTime;
import org.joda.time.Days;

public class ExpertiseExplorer {

    public List<String> filePathUpdater(List<String> reviewFileList){
        List<String> filePaths = new ArrayList<String>();
        for(String f : reviewFileList){
            String ff = f.substring(0, f.lastIndexOf(".java"));
            ff = ff.replace("/", ".");
            filePaths.add(ff);
        }
        return filePaths;
    }

    public RecommendationEvaluation baseModelReviewRecommendation(String projectName, String pullRequestPath, String pullReviewPath, String pullCommentPath,
            String pullFileChangPath) {

        ReviwerRecommendationDataLoader dataModel = new ReviwerRecommendationDataLoader(projectName, pullRequestPath, pullReviewPath,
                pullCommentPath, pullFileChangPath);
        Map<Integer, DescriptiveStatistics> meanAveragePrecision = new HashMap<Integer, DescriptiveStatistics>();
        Map<String, Set<String>> reviewerReviewExperience = new HashMap<String, Set<String>>();
        TrainTestDataModel trainTestModel = dataModel.getTrainTestSplits();

        Map<String, ArrayList<ReviewerRankingModel>> rankingResultList = new HashMap<String, ArrayList<ReviewerRankingModel>>();

        int count = 0;

        for (String prTest : trainTestModel.getTestingPullRequestList()) {
            PullRequestModel prModel = dataModel.getPullRequestList().get(prTest);
            DateTime prTime = prModel.getPrCreatedJodaTime();

            List<String> reviewFileList = filePathUpdater(dataModel.getPrChangedFileList().get(prTest));

            Map<String, ArrayList<String>> devHistory = dataModel.getAuthorCommitList();
            Map<String, Set<Integer>> devFreq = new HashMap<String, Set<Integer>>();
            Set<String> devListSet = new HashSet<String>();
            List<String> devList = new ArrayList<String>();
            ++count;
            Map<String, DateTime> developerFileUpdateScore = new HashMap<String, DateTime>();

            for (String devName : devHistory.keySet()) {
                for (String commitId : devHistory.get(devName)) {
                    GitCommitModel commit = dataModel.getGitCommitListMap().get(commitId);
                    
                    if (commit.getCommitJodaDate().isAfter(prTime.minusDays(1))) {
                        continue;
                    }
                    
                    List<String> commitChangeFileList = commit.getChangedJavaFileList();
                    for (String fileName : commitChangeFileList) {
                        if(reviewFileList.contains(fileName)){
                            if(!developerFileUpdateScore.containsKey(devName)){
                                developerFileUpdateScore.put(devName, commit.getCommitJodaDate());
                            }else{
                                DateTime currentRecentDate = developerFileUpdateScore.get(devName);
                                if(commit.getCommitJodaDate().isAfter(currentRecentDate)){
                                    developerFileUpdateScore.put(devName, commit.getCommitJodaDate());
                                }
                            }
                        }
                    }
                }
            }

            ArrayList<ReviewerRankingModel> rankReviewers = new ArrayList<ReviewerRankingModel>();
            for(String devName : developerFileUpdateScore.keySet()){
                ReviewerRankingModel rm = new ReviewerRankingModel();
                rm.setReviewerName(devName);
                rm.setRecentAccessDate(developerFileUpdateScore.get(devName));
                rankReviewers.add(rm);
            }
            if(rankReviewers.size() < 5){
                for(int k = 0 ; k < 5 ; k ++){
                    ReviewerRankingModel rm = new ReviewerRankingModel();
                    rm.setReviewerName("X"+k);
                    rm.setRecentAccessDate(DateUtil.formatterWithHyphen.parseDateTime("1998-01-01"));
                    rankReviewers.add(rm);
                }
            }
            rankReviewers.sort(new Comparator<ReviewerRankingModel>() {
                @Override
                public int compare(ReviewerRankingModel o1, ReviewerRankingModel o2) {
                   return (o2.getRecentAccessDate().compareTo(o1.getRecentAccessDate()));
                }
               });
               rankingResultList.put(prTest, rankReviewers);
               
            
            /*   for(ReviewerRankingModel rm : rankReviewers){
                System.out.println(rm.getReviewerName() + " Date: " + rm.getRecentAccessDate());
            }
            System.out.println("========================");

            if(count > 10){
                System.exit(0);
            }*/
        }

        
        RecommendationEvaluation evaluationRest = new RecommendationEvaluation(projectName);
        evaluationRest.calculateAccuracyAndMeanAveragePrecisionII(dataModel, 
        dataModel.getTrainTestSplits().getTestingPullRequestList(), 
        null, rankingResultList,
        ConstantUtil.REVIEW_EXPLORER);

          
        System.out.println("Project: " + projectName);

        evaluationRest.printAccuracyResult();
        evaluationRest.printMAP();

        System.out.println(dataModel.getCandidateReviewerList().size()); 

        return evaluationRest;

    }

    public void startRecoommendation(){
        List<String> projectList = Arrays.asList("apache_lucene", "apache_wicket", "apache_activemq",
               "apache_hbase", "apache_hive", "apache_storm", "apache_groovy",
                "elastic_elasticsearch");
        
        //List<String> projectList = Arrays.asList("apache_activemq","apache_groovy","apache_lucene",
        //"apache_hbase","apache_hive", "apache_storm","apache_wicket");

        //ist<String> projectList = Arrays.asList("apache_hive");

        List<RecommendationEvaluation> recommendationResultList = new ArrayList<RecommendationEvaluation>();
        for(String projectName : projectList){
            String prFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request/pr_reports_" + projectName +".csv";
            String prChagneFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_changed_files/pull_request_files_csv/" + projectName +"_files.csv";
            String prCommentFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_comments/comments_csv_files/" + projectName +"_comments_with_discussion.csv";
            String prReviewerPath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_reviewer/reviewer_csv_files/" + projectName + "_review.csv";
            RecommendationEvaluation recRes = baseModelReviewRecommendation(projectName, prFilePath, prReviewerPath, prCommentFilePath, prChagneFilePath);
            recommendationResultList.add(recRes);
        }
        String resultPath = RecommendationUtil.RESULT_DIR + "Expert_Recom_Baseline_15_May_2022.csv";
        RecommendationUtil.writeRecResult(resultPath, recommendationResultList);
    }

    public static void main(String[] args) {

        String projectName = "apache_hbase";
        String prFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request/pr_reports_" + "apache_hbase" + ".csv";
        String prChagneFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_changed_files/pull_request_files_csv/apache_hbase_files.csv";
        String prCommentFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_comments/comments_csv_files/apache_hbase_comments.csv";
        String prReviewerPath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_reviewer/reviewer_csv_files/apache_hbase_review.csv";
        
        ExpertiseExplorer ob = new ExpertiseExplorer();
        ob.startRecoommendation();
        
        System.out.println("Program finishes successfully");
    }
}
