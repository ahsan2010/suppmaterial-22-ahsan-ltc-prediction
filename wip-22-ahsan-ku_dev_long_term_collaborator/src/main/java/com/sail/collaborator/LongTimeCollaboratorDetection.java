package main.java.com.sail.collaborator;

import java.util.Arrays;
import java.util.List;

import com.sail.model.ReviwerRecommendationDataLoader;

public class LongTimeCollaboratorDetection {


    public ReviwerRecommendationDataLoader getDataModel(String projectName, 
    String pullRequestPath, String pullReviewPath, String pullCommentPath, String pullFileChangPath ){
        ReviwerRecommendationDataLoader dataModel = new ReviwerRecommendationDataLoader(projectName, pullRequestPath,
                pullReviewPath, pullCommentPath, pullFileChangPath);
        return dataModel;
    }

    public void detectLongTimeCollaborator(){

        long startTime = System.currentTimeMillis();
       
        //List<String> projectList = Arrays.asList("apache_activemq","apache_groovy","apache_lucene",
        //"apache_hbase","apache_hive", "apache_storm","apache_wicket", "elastic_elasticsearch");
        
        List<String> projectList = Arrays.asList("apache_activemq");
        

        for(String projectName : projectList){
            System.out.println("Working ["+ projectName +"]");
            String pullRequestPath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request/pr_reports_" + projectName +".csv";
            String pullReviewPath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_reviewer/reviewer_csv_files/" + projectName + "_review.csv";
            String pullFileChangPath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_changed_files/pull_request_files_csv/" + projectName +"_files.csv";
            String pullCommentPath = "/scratch/ahsan/Java_Exam_Work/Result/pull_request_comments/comments_csv_files/" + projectName +"_comments_with_discussion.csv";
           
            getDataModel(projectName, pullRequestPath, pullReviewPath, pullCommentPath, pullFileChangPath);
        }


        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Executation Time ["+(totalTime/1000)+"]");
    }

    public static void main(String[] args) {
        LongTimeCollaboratorDetection ob = new LongTimeCollaboratorDetection();
        ob.detectLongTimeCollaborator();
        System.out.println("Program finishes successfully");
    }

}
