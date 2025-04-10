# Predicting Long-time Contributors with Knowledge Units - Supplemental Materials

This repository contains the replication package for our manuscript:

**"Predicting Long-time Contributors with Knowledge Units of Programming Languages: An Empirical Study"**

---

## Table of Contents

1. [Extracting KUs from Java Source Code](#1-extracting-kus-from-java-source-code)  
2. [Generate KU-based Features for Profile Dimensions](#2-generate-ku-based-features-for-profile-dimensions)  
3. [Construct Studied Models (KULTC and Baseline)](#3-construct-studied-models-kultc-and-baseline)  
4. [Hyper-parameter Tuning Analysis](#4-hyper-parameter-tuning-analysis)  
5. [Rank Models Based on Performance (AUC)](#5-rank-models-based-on-performance-auc)  
6. [Model Feature Importance Analysis](#6-model-feature-importance-analysis)  

---

## 1. Extracting KUs from Java Source Code

- **Configure Paths:**  
  Update all paths in: util/ConstantUtil.java
- **Run Main Script:**  
To extract KUs from selected project commits, execute: MasterRepositoryReleaseLevelCommitAnalyzerOtherProject.java

- **Details:**
  - Threaded implementation accelerates KU extraction.
  - Set `numberOfThreads` (default: 30) for concurrent processing.
  - Internally, this invokes:
    ```
    ChildRepositoryReleaseLevelCommitAnalyzerOtherProject.java
    ```
  - Extracted KUs for each file in every commit are saved in the designated location.

---

## 2. Generate KU-based Features for Profile Dimensions

- **Configure Paths:**  
Set paths in `util/ConstantUtil.java`.

- **Run Feature Extraction Scripts:**

| Dimension | Script |
|-----------|--------|
| Studied Project Developer Expertise | `studiedProjects/KUFeatureExtractorMaster.java` |
| Other Project Developer Expertise   | `otherProjects/KUFeatureExtractorMaster.java` |
| Collaborator Expertise              | `collaborator/KUFeatureExtractorMaster.java` |
| Studied Project Characteristics     | `projectdim/ProjectDimKUAnalyzer.java` |
| Other Project Characteristics       | `projectdim/OtherProjDimKUAnalyzer.java` |

- **Output:**  
Each script will generate a `.csv` file containing the KU feature vector for the respective dimension.

---

## 3. Construct Studied Models (KULTC and Baseline)

- **Language:** Python  
- **Required Packages:**  
`sklearn`, `numpy`, `pandas`, `xgboost`, `imblearn`, `lightgbm`, `pickle`, `multiprocessing`

- **Run Script:**
- hyper-parameter-analysis/hyper-parameter-tuning-analyzer.py
- **Purpose:**  
  Builds models using varying parameters and classification algorithms to analyze performance.

---

## 5. Rank Models Based on Performance (AUC)

- **Method Used:** Scott-Knott ESD

- **Scripts:**
  - Generate SK-Rank Input:
    ```
    ltc-model-analysis/model-sk-rank-analysis.py
    ```
  - Apply SK-Rank:
    ```
    ku-model-variation/sk-rank-models.py
    ```
---

## 6. Model Feature Importance Analysis

### SHAP Analysis

- **Purpose:**  
  Analyze feature importance using SHAP — a widely used local interpretation technique.

- **Required Package:**  
  `shap`

- **Scripts:**
  - Generate SHAP values:
    ```
    ltc-model-analysis/shap_feature_importance_analysis_trained_all_data_single_model.py
    ```
  - Analyze SHAP results:
    ```
    ku-model-variation/full_model_feature_shap.py
    ```

### Rank Features Using Scott-Knott

- **Method Call:**  
  Run the following function inside the SHAP script:
  ```python
  generate_sk_rank_input_data()
