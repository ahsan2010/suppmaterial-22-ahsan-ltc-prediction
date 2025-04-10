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

public class BaseLineDevComFreq {

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

        for(String prTest : trainTestModel.getTestingPullRequestList()){
            List<String> reviewFileList = filePathUpdater(dataModel.getPrChangedFileList().get(prTest));
            ArrayList<ReviewerRankingModel> rankReviewers = new ArrayList<ReviewerRankingModel>();

            PullRequestModel prModel = dataModel.getPullRequestList().get(prTest);
            DateTime prTime = prModel.getPrCreatedJodaTime();

            Map<String, ArrayList<String>> devHistory = dataModel.getAuthorCommitList();
            Map<String, Set<String>> devCommitFreq = new HashMap<String,Set<String>>();
            Set<String> devListChangeReviewFiles = new HashSet<String>();
            for(String devName : devHistory.keySet()){
                for(String commitId : devHistory.get(devName)){
                    GitCommitModel commit = dataModel.getGitCommitListMap().get(commitId);
                    if(commit.getCommitJodaDate().isAfter(prTime)){
                        continue;
                    }
                    if(commit.getChangedJavaFileList().size() <= 0){
                        continue;
                    }
                    List<String> commitChangeFileList = commit.getChangedJavaFileList();
                    for (String fileName : commitChangeFileList) {
                        if(reviewFileList.contains(fileName)){
                            devListChangeReviewFiles.add(devName);
                        }
                    }
                    // we need to consider these values
                    if(!devCommitFreq.containsKey(devName)){
                        devCommitFreq.put(devName, new HashSet<String>());
                    }
                    devCommitFreq.get(devName).add(commitId);
                }
            }
            for(String devName : devCommitFreq.keySet()){
                if(!devListChangeReviewFiles.contains(devName)){
                    continue;
                }
                ReviewerRankingModel rm = new ReviewerRankingModel();
                rm.setReviewerName(devName);
                rm.setScore((double)devCommitFreq.get(devName).size());
                rankReviewers.add(rm);
            }
            if(rankReviewers.size() < 5){
                for(int k = 0 ; k < 5 ; k ++){
                    ReviewerRankingModel rm = new ReviewerRankingModel();
                    rm.setReviewerName("X"+k);
                    rm.setScore((double)(-1));
                    rankReviewers.add(rm);
                }
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
        }

        

        RecommendationEvaluation evaluationRest = new RecommendationEvaluation(projectName);
        evaluationRest.calculateAccuracyAndMeanAveragePrecisionII(dataModel, 
        dataModel.getTrainTestSplits().getTestingPullRequestList(), 
        null, rankingResultList,
        ConstantUtil.REVIEW_COMMIT_EXP);
        // Now we are going to test
        
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
        String resultPath = RecommendationUtil.RESULT_DIR + "res_baseline_commit_exp_review_files.csv";
        RecommendationUtil.writeRecResult(resultPath, recommendationResultList);
    }

    public static void main(String[] args) {

        String projectName = "apache_hbase";
        String prFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request/pr_reports_" + "apache_hbase" + ".csv";
        String prChagneFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_changed_files/pull_request_files_csv/apache_hbase_files.csv";
        String prCommentFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_comments/comments_csv_files/apache_hbase_comments.csv";
        String prReviewerPath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_reviewer/reviewer_csv_files/apache_hbase_review.csv";
        
        BaseLineDevComFreq ob = new BaseLineDevComFreq();
        ob.startRecoommendation();
        
        System.out.println("Program finishes successfully");
    }
}
