package com.sail.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.csvreader.CsvWriter;
import com.sail.github.model.GitCommitModel;
import com.sail.replication.model.ReviewerRankingModel;

import org.joda.time.DateTime;

import reviewerrecommendation.KnonwledgeUnitsExpertiseFromDevelopment;

public class RecommendationUtil {

    public static List<String> evalMetrics = Arrays.asList("Accuracy", "Recall", "MAP");
    public static String RESULT_DIR = "/scratch/ahsan/Java_Exam_Work/dev_ku_data/results/Result_Reco_Models/";

    public static Map<String, Double> getNormalizedTrainKUFactorFromPullRequest(
            Map<String, ArrayList<Map<String, Double>>> reviewerExpertiseKUData) {
        Map<String, Double> normalizedKUFactor = new HashMap<String, Double>();
        for (int i = 0; i < ConstantUtil.majorTopicList.size(); i++) {
            String kuName = ConstantUtil.majorTopicList.get(i);
            normalizedKUFactor.put(kuName, 0.0);
        }
        for (String reviewerName : reviewerExpertiseKUData.keySet()) {
            ArrayList<Map<String, Double>> reviewerKUData = reviewerExpertiseKUData.get(reviewerName);
            for (int i = 0; i < reviewerKUData.size(); i++) {
                Map<String, Double> kuValues = reviewerKUData.get(i);
                for (String kuName : kuValues.keySet()) {
                    Double value = kuValues.get(kuName);
                    normalizedKUFactor.put(kuName, normalizedKUFactor.get(kuName) + value);
                }
            }
        }
        return normalizedKUFactor;
    }

    public static Map<String, Map<String, Double>> calculateReviewExpertiseOld(
            Map<String, ArrayList<Map<String, Double>>> reviewerExpertiseKUData) {
        Map<String, Map<String, Double>> reviewerEpertise = new HashMap<String, Map<String, Double>>();
        for (String reviewerName : reviewerExpertiseKUData.keySet()) {
            double totalKUElements = 0.0;
            Map<String, Double> reviewerKU = new HashMap<String, Double>();
            for (int i = 0; i < ConstantUtil.majorTopicList.size(); i++) {
                String kuName = ConstantUtil.majorTopicList.get(i);
                reviewerKU.put(kuName, 0.0);
            }
            for (int i = 0; i < reviewerExpertiseKUData.get(reviewerName).size(); i++) {
                Map<String, Double> kuValues = reviewerExpertiseKUData.get(reviewerName).get(i);
                for (int k = 0; k < ConstantUtil.majorTopicList.size(); k++) {
                    String kuName = ConstantUtil.majorTopicList.get(k);
                    Double value = 0.0;
                    if (kuValues.containsKey(kuName)) {
                        value = kuValues.get(kuName);
                    }
                    reviewerKU.put(kuName, reviewerKU.get(kuName) + value);
                    totalKUElements += value;
                }
            }
            reviewerEpertise.put(reviewerName, new HashMap<String, Double>());
            for (int k = 0; k < ConstantUtil.majorTopicList.size(); k++) {
                String kuName = ConstantUtil.majorTopicList.get(k);
                double normalizedValue = reviewerKU.get(kuName) / Math.max(1, totalKUElements);
                reviewerEpertise.get(reviewerName).put(kuName, normalizedValue);
            }

        }
        return reviewerEpertise;
    }

    public static Map<String, Map<String, Double>> calculateReviewExpertise(
            Map<String, ArrayList<Map<String, Double>>> reviewerExpertiseKUData,
            Map<String, Double> normalizedKUFactor) {
        Map<String, Map<String, Double>> reviewerEpertise = new HashMap<String, Map<String, Double>>();

        for (String reviewerName : reviewerExpertiseKUData.keySet()) {
            Map<String, Double> reviewerKU = new HashMap<String, Double>();
            for (int i = 0; i < ConstantUtil.majorTopicList.size(); i++) {
                String kuName = ConstantUtil.majorTopicList.get(i);
                reviewerKU.put(kuName, 0.0);
            }
            for (int i = 0; i < reviewerExpertiseKUData.get(reviewerName).size(); i++) {
                Map<String, Double> kuValues = reviewerExpertiseKUData.get(reviewerName).get(i);
                for (int k = 0; k < ConstantUtil.majorTopicList.size(); k++) {
                    String kuName = ConstantUtil.majorTopicList.get(k);
                    Double value = 0.0;
                    if (kuValues.containsKey(kuName)) {
                        value = kuValues.get(kuName);
                    }
                    reviewerKU.put(kuName, reviewerKU.get(kuName) + value);
                }
            }
            reviewerEpertise.put(reviewerName, new HashMap<String, Double>());
            for (int k = 0; k < ConstantUtil.majorTopicList.size(); k++) {
                String kuName = ConstantUtil.majorTopicList.get(k);
                double normalizedValue = reviewerKU.get(kuName) / Math.max(1, normalizedKUFactor.get(kuName));
                reviewerEpertise.get(reviewerName).put(kuName, normalizedValue);
            }
        }
        return reviewerEpertise;
    }

    public static Map<String, Double> getNormalizedKUFactorFromDevelopementHistory(
            KnonwledgeUnitsExpertiseFromDevelopment developmentKnowledgeUnits, DateTime boundaryDate) {
        Map<String, Double> normalizedKUFactor = new HashMap<String, Double>();
        for (int k = 0; k < ConstantUtil.majorTopicList.size(); k++) {
            String kuName = ConstantUtil.majorTopicList.get(k);
            normalizedKUFactor.put(kuName, 0.0);
        }
        for (String commitId : developmentKnowledgeUnits.getFileKnowledgeUnitOccuranceMatrix().keySet()) {
            GitCommitModel commitModel = developmentKnowledgeUnits.getGitCommitsMapWithCommit().get(commitId);
            if (commitModel.getCommitAuthorJodaDate().isAfter(boundaryDate)) {
                continue;
            }
            for (String fileName : developmentKnowledgeUnits.getFileKnowledgeUnitOccuranceMatrix().get(commitId)
                    .keySet()) {
                for (int k = 0; k < ConstantUtil.majorTopicList.size(); k++) {
                    String kuName = ConstantUtil.majorTopicList.get(k);
                    double kuValue = 0;
                    if (developmentKnowledgeUnits.getFileKnowledgeUnitOccuranceMatrix().get(commitId).get(fileName)
                            .containsKey(kuName)) {
                        kuValue = developmentKnowledgeUnits.getFileKnowledgeUnitOccuranceMatrix().get(commitId)
                                .get(fileName).get(kuName);
                    }
                    normalizedKUFactor.put(kuName, normalizedKUFactor.get(kuName) + kuValue);
                }
            }
        }

        return normalizedKUFactor;
    }

    public static void writeRecResult(String path, List<RecommendationEvaluation> recommendationResultList) {
        try {
            CsvWriter writer = new CsvWriter(path);
            writer.write("Project_Name");
            writer.write("Eval_Metric");
            for (int i = 0; i < recommendationResultList.get(0).getStudiedRankedList().length; i++) {
                writer.write("Rank-" + recommendationResultList.get(0).getStudiedRankedList()[i]);
            }
            writer.endRecord();

            for (RecommendationEvaluation recRes : recommendationResultList) {
                for (String evalMetric : evalMetrics) {
                    writer.write(recRes.getProjectName());
                    writer.write(evalMetric);
                    for (int i = 0; i < recRes.getStudiedRankedList().length; i++) {
                        int rank = recRes.getStudiedRankedList()[i];
                        if (evalMetric.compareTo("Accuracy") == 0) {
                            writer.write(String.format("%.4f", recRes.getAccuracyValue()[rank]));
                        } else if (evalMetric.compareTo("Recall") == 0) {
                            writer.write(String.format("%.4f", recRes.getRecallValue()[rank]));
                        } else if (evalMetric.compareTo("MAP") == 0) {
                            writer.write(String.format("%.4f", recRes.getMeanAveragePrecision().get(rank).getMean()));
                        }
                    }
                    writer.endRecord();
                }
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static ReviewerRankingModel scoreFunction(String devName,
            Map<String, Double> kuPresentInChangeFiles,
            Map<String, Map<String, Double>> reviewerDevelopementEpertiseHistory,
            Map<String, Map<String, Double>> reviewerReviewEpertiseHistory,
            List<Double> weights) {

        ReviewerRankingModel revScore = new ReviewerRankingModel();
        Map<String, Double> devExpertise = reviewerDevelopementEpertiseHistory.get(devName);

        double devScore = 0.0;
        for (int i = 0; i < ConstantUtil.majorTopicList.size(); i++) {
            String kuName = ConstantUtil.majorTopicList.get(i);
            if (devExpertise.containsKey(kuName)) {
                devScore = devScore + devExpertise.get(kuName) * kuPresentInChangeFiles.get(kuName) * weights.get(i);
            }
        }
        revScore.setDevScore(devScore);
        revScore.setReviewScore(0.0);
        revScore.setScore(devScore);
        revScore.setReviewerName(devName);

        return revScore;
    }

    // Add developement + reviewer review experience
    public static ReviewerRankingModel scoreFunctionIII(String devName,
            Map<String, Double> kuPresentInChangeFiles,
            Map<String, Map<String, Double>> reviewerDevelopementEpertiseHistory,
            Map<String, Map<String, Double>> reviewerReviewEpertiseHistory,
            List<Double> weights) {

        ReviewerRankingModel revScore = new ReviewerRankingModel();

        Map<String, Double> reviewExpertise = null;
        double hasReviewExpereience = 0.0;
        if (reviewerReviewEpertiseHistory.containsKey(devName)) {
            hasReviewExpereience = 1.0;
        }
        double REVIEW_EXPERIENCE = 1.0;
        double devScore = 0.0;
        double reviewingScore = 0.0;

        Map<String, Double> devExpertise = reviewerDevelopementEpertiseHistory.get(devName);
        if (hasReviewExpereience > 0) {
            reviewExpertise = reviewerReviewEpertiseHistory.get(devName);
        }

        for (int i = 0; i < ConstantUtil.majorTopicList.size(); i++) {
            String kuName = ConstantUtil.majorTopicList.get(i);
            if (devExpertise.containsKey(kuName)) {
                devScore = devScore + devExpertise.get(kuName) * kuPresentInChangeFiles.get(kuName) * weights.get(i);
            }
            if (reviewExpertise != null) {
                if (reviewExpertise.containsKey(kuName)) {
                    reviewingScore = reviewingScore
                            + reviewExpertise.get(kuName) * kuPresentInChangeFiles.get(kuName) * weights.get(i);
                }
            }
        }
        double score = devScore + reviewingScore + REVIEW_EXPERIENCE * hasReviewExpereience;
        revScore.setDevScore(devScore);
        revScore.setReviewScore(reviewingScore);
        revScore.setScore(score);
        revScore.setReviewerName(devName);

        return revScore;
    }

    // Include Time Factor
    public static ReviewerRankingModel scoreFunctionIV(String devName,
            Map<String, Double> kuPresentInChangeFiles,
            Map<String, Map<String, Double>> reviewerDevelopementEpertiseHistory,
            Map<String, Map<String, Double>> reviewerReviewEpertiseHistory,
            List<Double> weights, Double lastAccessTimeScore) {

        ReviewerRankingModel revScore = new ReviewerRankingModel();

        Map<String, Double> reviewExpertise = null;
        double hasReviewExpereience = 0.0;
        if (reviewerReviewEpertiseHistory.containsKey(devName)) {
            hasReviewExpereience = 1.0;
        }
        double REVIEW_EXPERIENCE = 1.0;
        double devScore = 0.0;
        double reviewingScore = 0.0;

        Map<String, Double> devExpertise = reviewerDevelopementEpertiseHistory.get(devName);
        if (hasReviewExpereience > 0) {
            reviewExpertise = reviewerReviewEpertiseHistory.get(devName);
        }

        for (int i = 0; i < ConstantUtil.majorTopicList.size(); i++) {
            String kuName = ConstantUtil.majorTopicList.get(i);
            if (devExpertise.containsKey(kuName)) {
                devScore = devScore + devExpertise.get(kuName) * kuPresentInChangeFiles.get(kuName) * weights.get(i);
            }
            if (reviewExpertise != null) {
                if (reviewExpertise.containsKey(kuName)) {
                    reviewingScore = reviewingScore
                            + reviewExpertise.get(kuName) * kuPresentInChangeFiles.get(kuName) * weights.get(i);
                }
            }
        }
        double score = devScore + reviewingScore + REVIEW_EXPERIENCE * hasReviewExpereience + lastAccessTimeScore;
        revScore.setDevScore(devScore);
        revScore.setReviewScore(reviewingScore);
        revScore.setScore(score);
        revScore.setReviewerName(devName);
        revScore.setLastTimeAccessFileScore(lastAccessTimeScore);

        return revScore;
    }

    // Include Time Factor
    public static ReviewerRankingModel scoreFunctionV(String devName,
            Map<String, Double> kuPresentInChangeFiles,
            Map<String, Map<String, Double>> reviewerDevelopementEpertiseHistory,
            Map<String, Map<String, Double>> reviewerReviewEpertiseHistory,
            List<Double> weights, Map<String,Double> lastAccessKUTimeScore,
            Map<String,Double> lastAccessKUTimeScorePR) {

        ReviewerRankingModel revScore = new ReviewerRankingModel();

        Map<String, Double> reviewExpertise = null;
        double hasReviewExpereience = 0.0;
        if (reviewerReviewEpertiseHistory.containsKey(devName)) {
            hasReviewExpereience = 1.0;
        }
        double REVIEW_EXPERIENCE = 1.0;
        double devScore = 0.0;
        double reviewingScore = 0.0;

        Map<String, Double> devExpertise = reviewerDevelopementEpertiseHistory.get(devName);
        if (hasReviewExpereience > 0) {
            reviewExpertise = reviewerReviewEpertiseHistory.get(devName);
        }

        for (int i = 0; i < ConstantUtil.majorTopicList.size(); i++) {
            String kuName = ConstantUtil.majorTopicList.get(i);
            if (devExpertise.containsKey(kuName)) {
                devScore = devScore + devExpertise.get(kuName) * kuPresentInChangeFiles.get(kuName) * weights.get(i) + lastAccessKUTimeScore.get(kuName);
            }
            if (reviewExpertise != null) {
                if (reviewExpertise.containsKey(kuName)) {
                    reviewingScore = reviewingScore
                            + reviewExpertise.get(kuName) * kuPresentInChangeFiles.get(kuName) * weights.get(i) + lastAccessKUTimeScorePR.get(kuName);
                }
            }
        }
        double score = devScore + reviewingScore + REVIEW_EXPERIENCE * hasReviewExpereience;
        revScore.setDevScore(devScore);
        revScore.setReviewScore(reviewingScore);
        revScore.setScore(score);
        revScore.setReviewerName(devName);
        //revScore.setLastTimeAccessFileScore(lastAccessTimeScore);

        return revScore;
    }

    // With reviewer expertise
    public static ReviewerRankingModel scoreFunctionII(String devName,
            Map<String, Double> kuPresentInChangeFiles,
            Map<String, Map<String, Double>> reviewerDevelopementEpertiseHistory,
            Map<String, Map<String, Double>> reviewerReviewEpertiseHistory,
            List<Double> weights) {

        ReviewerRankingModel revScore = new ReviewerRankingModel();

        double hasReviewExpereience = 0.0;
        if (reviewerReviewEpertiseHistory.containsKey(devName)) {
            hasReviewExpereience = 1.0;
        }
        double REVIEW_EXPERIENCE = 1.0;
        double reviewScore = 0.0;

        if (reviewerReviewEpertiseHistory.containsKey(devName)) {
            hasReviewExpereience = 1.0;
        }

        if (reviewerReviewEpertiseHistory.containsKey(devName)) {
            Map<String, Double> revExpertise = reviewerReviewEpertiseHistory.get(devName);
            for (int i = 0; i < ConstantUtil.majorTopicList.size(); i++) {
                String kuName = ConstantUtil.majorTopicList.get(i);
                if (revExpertise.containsKey(kuName)) {
                    reviewScore = reviewScore
                            + revExpertise.get(kuName) * kuPresentInChangeFiles.get(kuName) * weights.get(i);
                }
            }
        }
        double score = reviewScore + REVIEW_EXPERIENCE * hasReviewExpereience;
        revScore.setDevScore(0.0);
        revScore.setReviewScore(reviewScore);
        revScore.setScore(score);
        revScore.setReviewerName(devName);
        return revScore;
    }

    public List<String> filePathUpdater(List<String> reviewFileList) {
        List<String> filePaths = new ArrayList<String>();
        for (String f : reviewFileList) {
            String ff = f.substring(0, f.lastIndexOf(".java"));
            ff = ff.replace("/", ".");
            filePaths.add(ff);
        }
        return filePaths;
    }

    // only developmennt history with equal weight
    public static ArrayList<ReviewerRankingModel> rankingProcess1(
            Map<String, Map<String, Double>> reviewerDevelopementEpertiseHistory,
            Map<String, Map<String, Double>> reviewerReviewEpertiseHistory,
            Map<String, Double> kuPresentInChangeFiles,
            List<Double> weights) {

        ArrayList<ReviewerRankingModel> rankingResults = new ArrayList<ReviewerRankingModel>();

        Set<String> developerList = reviewerDevelopementEpertiseHistory.keySet();
        Set<String> reviewerList = reviewerReviewEpertiseHistory.keySet();

        for (String devName : developerList) {
            Map<String, Double> devExpertise = reviewerDevelopementEpertiseHistory.get(devName);
            // double score = scoreFunctionII(devName, kuPresentInChangeFiles,
            // reviewerDevelopementEpertiseHistory, reviewerReviewEpertiseHistory);
            ReviewerRankingModel rankedScore = scoreFunctionIII(devName, kuPresentInChangeFiles,
                    reviewerDevelopementEpertiseHistory,
                    reviewerReviewEpertiseHistory, weights);

            rankingResults.add(rankedScore);
        }
        rankingResults.sort(new Comparator<ReviewerRankingModel>() {
            @Override
            public int compare(ReviewerRankingModel o1, ReviewerRankingModel o2) {
                if (o1.getScore() < o2.getScore()) {
                    return 1;
                } else if (o1.getScore() > o2.getScore()) {
                    return -1;
                }
                return 0;
            }
        });

        return rankingResults;
    }

    // only developmennt history with equal weight
    public static ArrayList<ReviewerRankingModel> rankingProcess2(
            Map<String, Map<String, Double>> reviewerDevelopementEpertiseHistory,
            Map<String, Map<String, Double>> reviewerReviewEpertiseHistory,
            Map<String, Double> kuPresentInChangeFiles,
            List<Double> weights, Map<String, Double> devLastFileUpdateTimeScore) {

        ArrayList<ReviewerRankingModel> rankingResults = new ArrayList<ReviewerRankingModel>();

        Set<String> developerList = reviewerDevelopementEpertiseHistory.keySet();
        Set<String> reviewerList = reviewerReviewEpertiseHistory.keySet();

        for (String devName : developerList) {
            Map<String, Double> devExpertise = reviewerDevelopementEpertiseHistory.get(devName);
            // double score = scoreFunctionII(devName, kuPresentInChangeFiles,
            // reviewerDevelopementEpertiseHistory, reviewerReviewEpertiseHistory);
            // ReviewerRankingModel rankedScore = scoreFunctionIII(devName,
            // kuPresentInChangeFiles, reviewerDevelopementEpertiseHistory,
            // reviewerReviewEpertiseHistory, weights);

            ReviewerRankingModel rankedScore = scoreFunctionIV(devName, kuPresentInChangeFiles,
                    reviewerDevelopementEpertiseHistory,
                    reviewerReviewEpertiseHistory, weights, devLastFileUpdateTimeScore.get(devName));

            rankingResults.add(rankedScore);
        }
        rankingResults.sort(new Comparator<ReviewerRankingModel>() {
            @Override
            public int compare(ReviewerRankingModel o1, ReviewerRankingModel o2) {
                if (o1.getScore() < o2.getScore()) {
                    return 1;
                } else if (o1.getScore() > o2.getScore()) {
                    return -1;
                }
                return 0;
            }
        });

        return rankingResults;
    }

    // only developmennt history with equal weight
    public static ArrayList<ReviewerRankingModel> rankingProcess3(
            Map<String, Map<String, Double>> reviewerDevelopementEpertiseHistory,
            Map<String, Map<String, Double>> reviewerReviewEpertiseHistory,
            Map<String, Double> kuPresentInChangeFiles,
            List<Double> weights,Map<String, Map<String, Double>> devKULastAccess,
            Map<String, Map<String, Double>> devKULastAccessPR) {

        ArrayList<ReviewerRankingModel> rankingResults = new ArrayList<ReviewerRankingModel>();

        Set<String> developerList = reviewerDevelopementEpertiseHistory.keySet();
        Set<String> reviewerList = reviewerReviewEpertiseHistory.keySet();

        for (String devName : developerList) {
            Map<String, Double> devExpertise = reviewerDevelopementEpertiseHistory.get(devName);
            // double score = scoreFunctionII(devName, kuPresentInChangeFiles,
            // reviewerDevelopementEpertiseHistory, reviewerReviewEpertiseHistory);
            // ReviewerRankingModel rankedScore = scoreFunctionIII(devName,
            // kuPresentInChangeFiles, reviewerDevelopementEpertiseHistory,
            // reviewerReviewEpertiseHistory, weights);

            ReviewerRankingModel rankedScore = scoreFunctionV(devName, kuPresentInChangeFiles,
                    reviewerDevelopementEpertiseHistory,
                    reviewerReviewEpertiseHistory, weights, devKULastAccess.get(devName), devKULastAccessPR.get(devName));

            rankingResults.add(rankedScore);
        }
        rankingResults.sort(new Comparator<ReviewerRankingModel>() {
            @Override
            public int compare(ReviewerRankingModel o1, ReviewerRankingModel o2) {
                if (o1.getScore() < o2.getScore()) {
                    return 1;
                } else if (o1.getScore() > o2.getScore()) {
                    return -1;
                }
                return 0;
            }
        });

        return rankingResults;
    }

}
