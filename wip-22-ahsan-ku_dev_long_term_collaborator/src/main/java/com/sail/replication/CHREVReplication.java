package com.sail.replication;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.sail.model.ReviwerRecommendationDataLoader;
import com.sail.replication.model.CHREVRecommendScoreModel;
import com.sail.replication.model.PullRequestModel;
import com.sail.replication.model.ReviewerRankingModel;
import com.sail.util.ConstantUtil;
import com.sail.util.RecommendationEvaluation;
import com.sail.util.RecommendationUtil;

import org.joda.time.DateTime;
import org.joda.time.Days;

public class CHREVReplication {


    public CHREVRecommendScoreModel calculateCommentContributor(ReviwerRecommendationDataLoader dataModel,
    String prNumberTesting,
     List<String> changeFileListTesting,
    DateTime prDateTimeTesting){

        List<String> trainingPRList = dataModel.getTrainTestSplits().getFullPullRequestList();
        Set<String> reviewerListTesting = new HashSet<String>();
        Set<String> changeFileSetTesting = new HashSet<String>();

        Map<String,ArrayList<DateTime>> fullCommentsToFileList = new HashMap<String, ArrayList<DateTime>>();
        Map<String,Set<String>> fullWorkDay = new HashMap<String, Set<String>>();
        Map<String, Map<String, ArrayList<DateTime>>> reviewCommentFileList = new HashMap<String, Map<String, ArrayList<DateTime>>>();
        Map<String, Map<String, Set<String>>> reviewerWorkDayToFile = new HashMap<String, Map<String, Set<String>>>();

        Map<String, DateTime> fullCommentRecentWorkDay = new HashMap<String,DateTime>();
        Map<String, Map <String, DateTime>> reviewerRecentWorkDay = new HashMap<String, Map <String, DateTime>>();

        Map<String,Map<String, Double>> commentContributionRatio = new HashMap<String, Map<String, Double>>();
        Map<String,Map<String, Double>> workDayContributionRatio = new HashMap<String, Map<String, Double>>();
        Map<String,Map<String, Double>> recentWorkDayContributionRatio = new HashMap<String, Map<String, Double>>();
        Map<String, Map<String, Double>> workDayRatios = new HashMap<String, Map<String, Double>>();

        for(String revName : dataModel.getPullRequestReviewerMap().get(prNumberTesting)){
            reviewerListTesting.add(revName);
        }
        for(String changeFile: changeFileListTesting){
            changeFileSetTesting.add(changeFile);
        }


        for (String prNumber : trainingPRList) {
            PullRequestModel prModel = dataModel.getPullRequestList().get(prNumber);
            if (prModel.getPrCreatedJodaTime().isAfter(prDateTimeTesting)) {
                break;
            }
            for (String changeFile : dataModel.getPrChangedFileList().get(prNumber)) {
                if (changeFileSetTesting.contains(changeFile)) {
                    if (dataModel.getReviewerComments().containsKey(changeFile)) {
                        if (dataModel.getReviewerComments().get(changeFile).containsKey(prNumber)) {
                            Map<String, ArrayList<DateTime>> commenterDateTimeList = dataModel.getReviewerComments()
                                    .get(changeFile).get(prNumber);
                            //System.out
                              //      .println("PR Number: " + prNumber + " Commenter: " + commenterDateTimeList.size());

                            for (String commenterName : commenterDateTimeList.keySet()) {
                                for (DateTime commentDateTime : commenterDateTimeList.get(commenterName)) {
                                    if (commentDateTime.isAfter(prDateTimeTesting)) {
                                        break;
                                    }
                                    String commentTimeString = commentDateTime.getYear() + "-"
                                            + commentDateTime.getMonthOfYear() + "-" + commentDateTime.getDayOfMonth();
                                    // need to take this comment
                                    // Update Full Comment File map
                                    if (!fullCommentsToFileList.containsKey(changeFile)) {
                                        fullCommentsToFileList.put(changeFile, new ArrayList<DateTime>());
                                    }
                                    fullCommentsToFileList.get(changeFile).add(commentDateTime);

                                    // Update full work day
                                    if (!fullWorkDay.containsKey(changeFile)) {
                                        fullWorkDay.put(changeFile, new HashSet<String>());
                                    }
                                    fullWorkDay.get(changeFile).add(commentTimeString);

                                    // Recent work daay
                                    if (!fullCommentRecentWorkDay.containsKey(changeFile)) {
                                        fullCommentRecentWorkDay.put(changeFile, commentDateTime);
                                    } else {
                                        if (commentDateTime.isAfter(fullCommentRecentWorkDay.get(changeFile))) {
                                            fullCommentRecentWorkDay.put(changeFile, commentDateTime);
                                        }

                                        // Update Reviewer File Map

                                        if (!reviewCommentFileList.containsKey(changeFile)) {
                                            reviewCommentFileList.put(changeFile,
                                                    new HashMap<String, ArrayList<DateTime>>());
                                        }
                                        if (!reviewCommentFileList.get(changeFile).containsKey(commenterName)) {
                                            reviewCommentFileList.get(changeFile).put(commenterName,
                                                    new ArrayList<DateTime>());
                                        }
                                        reviewCommentFileList.get(changeFile).get(commenterName).add(commentDateTime);

                                        // Update review work day
                                        if (!reviewerWorkDayToFile.containsKey(changeFile)) {
                                            reviewerWorkDayToFile.put(changeFile, new HashMap<String, Set<String>>());
                                        }
                                        if (!reviewerWorkDayToFile.get(changeFile).containsKey(commenterName)) {
                                            reviewerWorkDayToFile.get(changeFile).put(commenterName,
                                                    new HashSet<String>());
                                        }
                                        reviewerWorkDayToFile.get(changeFile).get(commenterName).add(commentTimeString);

                                        // Update review recent day

                                        if (!reviewerRecentWorkDay.containsKey(changeFile)) {
                                            reviewerRecentWorkDay.put(changeFile, new HashMap<String, DateTime>());
                                        }
                                        if (!reviewerRecentWorkDay.get(changeFile).containsKey(commenterName)) {
                                            reviewerRecentWorkDay.get(changeFile).put(commenterName, commentDateTime);
                                        }
                                        if (commentDateTime
                                                .isAfter(reviewerRecentWorkDay.get(changeFile).get(commenterName))) {
                                            reviewerRecentWorkDay.get(changeFile).put(commenterName, commentDateTime);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Finish Training
        for(String changeFile : changeFileListTesting){
            if (!fullCommentsToFileList.containsKey(changeFile)){
                continue;
            }
            // Sum full Comment
            double fullCommentsValueForAllReviewersForThisFile = fullCommentsToFileList.get(changeFile).size();
            double fullWorkDayForThisFile = fullWorkDay.get(changeFile).size();
            DateTime fullRecentWorkDay = fullCommentRecentWorkDay.get(changeFile);
            if(!reviewCommentFileList.containsKey(changeFile)){
                continue;
            }
            // sum all reviewer
            for(String commenterName : reviewCommentFileList.get(changeFile).keySet()){
                double reviewerContribution = 0.0;
                double reviewerWorkDay = 0.0;
                

                if(reviewCommentFileList.get(changeFile).containsKey(commenterName)){
                    reviewerContribution = reviewCommentFileList.get(changeFile).get(commenterName).size();
                }
                double ratioCommentContrib = reviewerContribution/fullCommentsValueForAllReviewersForThisFile;

                if(reviewerWorkDayToFile.get(changeFile).containsKey(commenterName)){
                    reviewerWorkDay = reviewerWorkDayToFile.get(changeFile).get(commenterName).size();
                }
                double ratioWorkContrib = reviewerWorkDay/fullWorkDayForThisFile;

                double ratioRecentWorkContrib = 1.0;
                if(reviewerRecentWorkDay.get(changeFile).containsKey(commenterName)){
                    DateTime reviewerWorkDayRecent = reviewerRecentWorkDay.get(changeFile).get(commenterName);
                    double dayDiff = Math.abs(Days.daysBetween(reviewerWorkDayRecent, fullRecentWorkDay).getDays());
                    //System.out.println("Day Diff: " + dayDiff);
                    if(dayDiff > 0){
                        ratioRecentWorkContrib = 1 / dayDiff;
                    }
                }

                if(!commentContributionRatio.containsKey(changeFile)){
                    commentContributionRatio.put(changeFile, new HashMap<String, Double>());
                }
                if(!workDayContributionRatio.containsKey(changeFile)){
                    workDayContributionRatio.put(changeFile, new HashMap<String, Double>());
                }
                if(!recentWorkDayContributionRatio.containsKey(changeFile)){
                    recentWorkDayContributionRatio.put(changeFile, new HashMap<String, Double>());
                }
                commentContributionRatio.get(changeFile).put(commenterName, ratioCommentContrib);
                workDayContributionRatio.get(changeFile).put(commenterName, ratioWorkContrib);
                recentWorkDayContributionRatio.get(changeFile).put(commenterName, ratioRecentWorkContrib);
            }
        }
        CHREVRecommendScoreModel ob = new CHREVRecommendScoreModel();
        ob.setCommentContributionRatio(commentContributionRatio);
        ob.setWorkDayContributionRatio(workDayContributionRatio);
        ob.setRecentWorkDayContributionRatio(recentWorkDayContributionRatio);
        
        return ob;
    }
    

    public Map<String, ArrayList<ReviewerRankingModel>> testingReviewerRecommendation(){
        Map<String, ArrayList<ReviewerRankingModel>> rankingResultList = new HashMap<String, ArrayList<ReviewerRankingModel>>();
        return rankingResultList;
    }
    public RecommendationEvaluation recommendingReviewersWithCHREV(String projectName, String pullRequestPath,
    String pullReviewPath, String pullCommentPath, String pullFileChangPath){

        ReviwerRecommendationDataLoader dataModel = new ReviwerRecommendationDataLoader(projectName, pullRequestPath,
                pullReviewPath, pullCommentPath, pullFileChangPath);

        List<String> testingPRList = dataModel.getTrainTestSplits().getTestingPullRequestList();
        Map<String, ArrayList<ReviewerRankingModel>> rankingResultList = new HashMap<String, ArrayList<ReviewerRankingModel>>();
        int totalPRTesting = testingPRList.size();
        for(String prNumber : testingPRList){
            System.out.println("Working " + prNumber + " Remaining " + (totalPRTesting--));
            PullRequestModel prModel = dataModel.getPullRequestList().get(prNumber);
            DateTime prDateTime = prModel.getPrCreatedJodaTime();
            List<String> changeFileList = dataModel.getPrChangedFileList().get(prNumber);

            CHREVRecommendScoreModel chrevResult = calculateCommentContributor( dataModel, prNumber, changeFileList, prDateTime);
            rankingResultList.put(prNumber, new ArrayList<ReviewerRankingModel>());
            Map<String, Double> reviewerScoreList = new HashMap<String, Double>();
            for(String changeFile : chrevResult.getCommentContributionRatio().keySet()){
                for(String commenterName : chrevResult.getCommentContributionRatio().get(changeFile).keySet()){
                    double score = 
                    chrevResult.getCommentContributionRatio().get(changeFile).get(commenterName)
                    + chrevResult.getWorkDayContributionRatio().get(changeFile).get(commenterName) 
                    + chrevResult.getRecentWorkDayContributionRatio().get(changeFile).get(commenterName)
                    ;

                    if(!reviewerScoreList.containsKey(commenterName)){
                        reviewerScoreList.put(commenterName, 0.0);
                    }
                    reviewerScoreList.put(commenterName, reviewerScoreList.get(commenterName) + score);
                }
            }
            for(String revName : reviewerScoreList.keySet()){
                ReviewerRankingModel revScore = new ReviewerRankingModel();
                revScore.setReviewerName(revName);
                revScore.setScore(reviewerScoreList.get(revName));
                rankingResultList.get(prNumber).add(revScore);
            }
            for(String devName : dataModel.getDeveloperChangedJavaFilesCommits().keySet()){
                if(!reviewerScoreList.containsKey(devName)){
                    ReviewerRankingModel revScore = new ReviewerRankingModel();
                    revScore.setReviewerName(devName);
                    revScore.setScore(reviewerScoreList.get(0.0));
                    rankingResultList.get(prNumber).add(revScore);
                }
            }

            /*for(String changeFile : chrevResult.getCommentContributionRatio().keySet()){
                for(String commenterName : chrevResult.getCommentContributionRatio().get(changeFile).keySet()){
                    //System.out.println(changeFile + " " + commenterName + " Comment: " + chrevResult.getCommentContributionRatio().get(changeFile).get(commenterName) + " Work: " + chrevResult.getWorkDayContributionRatio().get(changeFile).get(commenterName) + " Recent Work: " + chrevResult.getRecentWorkDayContributionRatio().get(changeFile).get(commenterName)) ;
                }
            }*/
        }
        RecommendationEvaluation evaluationRest = new RecommendationEvaluation(projectName);
        String fileDir = "/home/local/SAIL/ahsan/BACKUP/ahsan_project_2021/ahsan_gustavo/dev_knowledge/Recommendation_Rank_Analysis/chrev_replication_recommendation/" + projectName + "/";

        evaluationRest. writeReviewerRankResultData(projectName, dataModel, rankingResultList, fileDir);

        evaluationRest.calculateAccuracyAndMeanAveragePrecisionII(dataModel, 
        dataModel.getTrainTestSplits().getTestingPullRequestList(), 
        null, rankingResultList,
                ConstantUtil.REVIEW_CHREV);
        System.out.println("Project Name: " + projectName);
        System.out.println("-----------------------");
        evaluationRest.printAccuracyResult();
        evaluationRest.printMAP();

        return evaluationRest;
    }

    public void startRecoommendation(){
        /*List<String> projectList = Arrays.asList("apache_lucene", "apache_wicket", "apache_activemq", "jruby_jruby",
                "caskdata_cdap", "apache_hbase", "apache_hive", "apache_storm", "apache_stratos", "apache_groovy",
                "elastic_elasticsearch");*/
        
        long startTime = System.currentTimeMillis();
       
        List<String> projectList = Arrays.asList("apache_activemq","apache_groovy","apache_lucene",
        "apache_hbase","apache_hive", "apache_storm","apache_wicket", "elastic_elasticsearch");
        List<RecommendationEvaluation> recommendationResultList = new ArrayList<RecommendationEvaluation>();

        for(String projectName : projectList){
            System.out.println("Working ["+ projectName +"]");
            String prFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request/pr_reports_" + projectName +".csv";
            String prChagneFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_changed_files/pull_request_files_csv/" + projectName +"_files.csv";
            String prCommentFilePath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_comments/comments_csv_files/" + projectName +"_comments_with_discussion.csv";
            String prReviewerPath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_reviewer/reviewer_csv_files/" + projectName + "_review.csv";
            RecommendationEvaluation recRes = recommendingReviewersWithCHREV(projectName, prFilePath, prReviewerPath, prCommentFilePath, prChagneFilePath);
            recommendationResultList.add(recRes);
        }

        String resultPath = RecommendationUtil.RESULT_DIR + "res_chrev_replication_reviewer_commenter.csv";
        RecommendationUtil.writeRecResult(resultPath, recommendationResultList);
        
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Executation Time ["+(totalTime/1000)+"]");
    }

    public static void main(String[] args) {
        CHREVReplication ob = new CHREVReplication();
        ob.startRecoommendation();
        System.out.println("Program finishes successfully");
    }
}
