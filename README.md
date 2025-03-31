# WiDS Datathon 2025 - UCLA Team 2
## Impact Narrative

The WiDS Datathon challenges us to uncover the mysteries of the female brain, particularly the unique sex patterns observed in ADHD. By diving deep into functional connectivity and neural interactions, our research aims to reveal how the female brain differs in its response to ADHD. These insights could pave the way for more personalized, sex-specific approaches in ADHD diagnosis and treatment, ultimately demystifying the complex neural underpinnings that contribute to behavioral differences.


## Table of Contents

- [Team Members](#team-members)
- [Project Highlights](#project-highlights)
- [Setup and Execution](#setup-and-execution)
- [Project Overview](#project-overview)
- [Data Exploration](#data-exploration)
- [Model Development](#model-development)
- [Results and Key Findings](#results-and-key-findings)
- [Impact Narrative](#impact-narrative)
- [Next Steps & Future Improvements](#next-steps--future-improvements)
- [References & Additional Resources](#references--additional-resources)


### 🏁 Completed Projects | 📚 Project Archive


**The Mysteries of the Female Brain: Sex Patterns in ADHD**  
[WiDS Datathon 2025 Kaggle Competition](https://www.kaggle.com/competitions/widsdatathon2025)

---

### 👯 Team Members

| Name               | GitHub Handle                | Contribution                                   |
| ------------------ | ---------------------------- | ---------------------------------------------- |
| **Mariia Nikitash**   | [@MariiaNikitash](https://github.com/Itz-creator07) | Data Understanding & Preprocessing, Feature Engineering, Model Development & Implementation      |
| **Shubhangi Waldiya** | [@shubhangibw](https://github.com/subhangibw) | Data Understanding & Preprocessing, Model Development, Implementation & Evaluation |
| **Ami Rajesh**      | [@Arajesh03](https://github.com/Arajesh03) | Data Preprocessing, Feature Engineering, Model Development & Evaluation       |
| **Itzalen Lopez** | [@Itz-creator07](https://github.com/MariiaNikitash) | Exploratory Data Analysis (EDA), Visualization & Model Development        |


### Teaching Assistant 👩‍🏫

- **Babu Swagath**


---


## **🎯 Project Highlights**

- First place: **UCLA_WiDS_Team_2** out of **87 teams** at Break Through Tech program.
- Ranked **64** on the final official Kaggle leaderboard.
- Developed a **multi-outcome machine learning model** to predict ADHD diagnosis and sex differentiation in females.
- Used **functional MRI connectome matrices** and **socio-demographic data** to improve ADHD diagnostic processes.
- Aimed to better understand **brain activity patterns** for personalized treatment of ADHD.

---

## **Business Understanding**
- Researched various models with a focus on binary classification.

## **Data Understanding**
- Collaboratively analyzed the datasets for key insights.

## **Data Preparation**
- Executed similar preprocessing steps tailored to individual workflows.

---

## **Modeling & Evaluation**

### Models Explored & Their Metrics

- **Decision Trees:**  
  - *Max Depth=3:* Accuracy = 0.5247  
  - *Max Depth=4:* Accuracy = 0.5714  
  - *Max Depth=5:* Accuracy = 0.5522  
  - *Max Depth=6:* Accuracy = 0.5302  

- **Random Forest:**  
  - Accuracy = 0.66  
  - Cross-validation scores: [0.67489712, 0.68312757, 0.62139918, 0.66528926, 0.6322314]  
  - Mean accuracy = 0.6554  

- **Logistic Regression:**  
  - Accuracy = 0.74  
  - Detailed Metrics:  
    ```
    precision    recall  f1-score   support
    False       0.64      0.11      0.18        65
    True        0.75      0.98      0.85       178

    accuracy                           0.74       243
    macro avg       0.69      0.54      0.52       243
    weighted avg    0.72      0.74      0.67       243
    ```  
  - Cross-validation scores: [0.74485597, 0.70781893, 0.66666667, 0.69421488, 0.65289256]  
  - Mean accuracy = 0.6933  

- **XGBoost:**  
  - Accuracy = 0.7582  
  - F1 Score = 0.8174  

- **K-Nearest Neighbors (KNN):**  
  - **k=10:** Accuracy = 0.7434  
  - **k=100:** Accuracy = 0.7434  
  - **k=500:** Accuracy = 0.7434  

---

### **Final Model: Decision Trees & Random Forest Combined**
- **Best Performing Model:** Highest Accuracy = **76.031%**

- **Key Details:**
  - **COMBINED.ipynb:** Central notebook for final experiments.
  - Best Decision Tree model predicted the **ADHD** column.
  - Optimized Random Forest model predicted the **SEX** column.
  - Extensive parameter tuning was performed.
  - Graphical representations of model performance were included.


<div align="left">
  <table>
    <tr>
      <!-- Kaggle Badge -->
      <td>
        <a href="https://www.kaggle.com">
          <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle Badge" width="110">
        </a>
      </td>
      <td style="vertical-align: middle; padding-left: 10px;">
        Kaggle is a collaborative platform where data scientist share datasets, build models and compete in machine learning challenges.
      </td>
    </tr>
    <tr>
      <!-- WiDS Image and Link -->
      <td>
        <img src="https://media1-production-mightynetworks.imgix.net/asset/9e6b6f30-7e7d-47ad-9429-bd7e494c5959/Logo_Icon.png?ixlib=rails-4.2.0&auto=format&w=192&h=192&fit=crop&impolicy=Avatar" alt="WiDS Logo" width="50">
      </td>
      <td style="vertical-align: middle; padding-left: 10px;">
        🔗 <a href="https://www.kaggle.com/competitions/widsdatathon2025/overview">WiDS Datathon 2025 Kaggle Competition Page</a>
      </td>
    </tr>
  </table>
</div>

---
## ⚙️ Setup & Execution  

Follow these step-by-step instructions to run this repository on **Google Colab** and **GitHub**.  

![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)  
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)  
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)  


### 📋 Steps to Run  

### 🏷️ 1. Clone the Repository from GitHub  
   - Open **Google Colab**, start a new notebook, and run the following command to **download the repository**:  

     ```python
     !git clone https://github.com/UCLA-WiDS-Team/adhd-brain-prediction.git
     %cd adhd-brain-prediction
     ```

   - This will copy all code files from **GitHub** into your **Colab environment**.  


### 📦 2. Install Dependencies  
   - Install the required libraries manually using the following command. This includes libraries such as **pandas, numpy, matplotlib, seaborn, scikit-learn**, and others necessary for running the code:  

     ```python
     !pip install pandas numpy matplotlib seaborn scikit-learn
     ```

✅ Automate installation of all required libraries, ensure version consistency across different setups, and save time by avoiding missing dependencies.
    
### 📥 3. Download Datasets from Kaggle  
   The datasets are hosted on **Kaggle**,  and it must be downloaded from the competition page:  
   🔗 [WiDS Datathon 2025 Dataset](https://www.kaggle.com/competitions/widsdatathon2025/data)  

   #### 🔑 Setting Up Kaggle API  
   - First, install **Kaggle API**:  

     ```python
     !pip install kaggle
     ```

   - Upload your **Kaggle API key** (`kaggle.json`). You can generate this key from your Kaggle account under **Account Settings**:  

     ```python
     from google.colab import files
     files.upload()  # Upload kaggle.json
     ```

   - Move the API key to the correct directory and set permissions:  

     ```python
     !mkdir -p ~/.kaggle
     !mv kaggle.json ~/.kaggle/
     !chmod 600 ~/.kaggle/kaggle.json
     ```

   #### 📂 Download and Extract the Dataset  
   - Run the following command to **download the dataset**:  

     ```python
     !kaggle competitions download -c widsdatathon2025
     ```

   - Extract the dataset:  

     ```python
     !unzip widsdatathon2025.zip
     ```

   - Load the dataset into a **pandas DataFrame**:  

     ```python
     import pandas as pd
     df = pd.read_csv("train.csv")  # Adjust filename if needed
     df.head()
     ```

#### 🔑 **Why Do You Need a Kaggle API Key?**  
   The **Kaggle API key** is required to **programmatically download datasets** from Kaggle directly into **Google Colab**. Without it, you would need to manually download the dataset from Kaggle and upload it to Colab. The key helps with:  
   - **Authentication** – Kaggle restricts dataset access to registered users, so an API key verifies your identity.  
   - **Automated Downloads** – Fetch datasets directly without manual intervention.  
   - **Colab Integration** – Since Colab runs in the cloud, it doesn’t store your Kaggle login credentials.  
   - **Reproducibility** – Ensures others running your code can fetch the exact dataset without extra steps.  

### 🚀 4. Execute the Code  

#### 📌 Running the Notebook  
   - Open the **Google Colab notebook** from the cloned repository.  
   - Run **all cells** sequentially to reproduce results.  

#### 🏃 Running a Python Script  
   - If executing a standalone Python script, use the following command:  

     ```python
     !python your_script.py
     ```

---


This guide ensures a smooth setup process by covering:

- ✅ Cloning the GitHub repository  
- ✅ Installing dependencies (e.g., pandas, numpy, etc.)  
- ✅ Downloading the WiDS Datathon 2025 dataset from Kaggle  
- ✅ Explaining the need for the Kaggle API key  
- ✅ Running the notebook or script for execution  

This structured approach ensures users:

- ✅ Properly clone the repository before proceeding  
- ✅ Understand the code's source and execution process  
- ✅ Follow a clear flow: cloning → installing dependencies → downloading data → executing the code 🚀




## 🏗️ Project Overview

### About the Competition

The **Women in Data Science (WiDS) Datathon 2025** is a Kaggle competition aimed at encouraging women in AI and data science. This initiative is part of the **Break Through Tech AI Program** that seeks to bridge the gender gap in AI by offering real-world machine learning challenges.

### Objective

Our team is analyzing **The Mysteries of the Female Brain: Sex Patterns in ADHD**, a project designed to identify differences in ADHD diagnosis and manifestations in females using AI-driven predictions. The model predicts:

1. **ADHD diagnosis** (1 = ADHD, 0 = No ADHD)
2. **Sex** (1 = Female, 0 = Male)

### Real-World Significance

ADHD diagnosis can be challenging, especially in females, due to symptom presentations. Our model aims to:

- Improve **diagnostic accuracy** for ADHD in females
- Investigate **brain activity patterns** and their correlation with ADHD across genders
- Contribute to **personalized treatment** options for ADHD

---

## 📊 Data Exploration

### Dataset Description

Our project leverages data from the **Healthy Brain Network (HBN)**, which includes:

- Functional MRI connectome matrices
- Socio-demographic data
- Behavioral and parenting assessments
- ADHD diagnostic labels

### Exploratory Data Analysis (EDA) & Visualization 

 
  - Analyze distributions and relationships within the data (e.g., socio-demographic variables and MRI metrics).
  - Process categorical data and combine it with the functional connectome matrices.
  - Identify outliers, missing values, and noise to inform feature engineering and preprocessing strategies.
  
**Visualizations**
    - Visualized **brain activity patterns** and their link to ADHD
    - Examined **demographic distributions** and ADHD correlations
    - Assessed **feature importance** in model development

- **Feature Correlation Matrix Heatmap for Multiple Brain Regions**  
![Feature Correlation Matrix Heatmap](https://i.postimg.cc/wxLfVp0n/Feature-Correlation-Matrix-Heatmap.png)

- **Distribution of ADHD vs. Non-ADHD Participants by Demographics**  
![Distribution of ADHD vs. Non-ADHD Participants by Sex](https://i.postimg.cc/GpKgvJKt/adhd-distribuition.png)

**MRI Scan Outcome** 
![undefined-Imgur](https://i.postimg.cc/W1TfsnXN/undefined-Imgur.png)

**Social Demographic Emotions** 
![undefined-Imgur](https://i.postimg.cc/QNgywwJ4/SDQ-SDQ-Emotional-Problems.png)

---


## 🧠 Model Development

- **Model(s) Used:**
    - Regression models and classification algorithms to predict ADHD diagnosis and sex
    - Hyperparameter tuning for optimal model performance

- **Training Setup:**
    - Data split for training and validation
    - Evaluation metric: **Accuracy** and **F1 score**

---

## 📈 Results & Key Findings

- **Kaggle Leaderboard Score:** [Insert score here]
- **Model performance:** High prediction accuracy across both ADHD diagnosis and sex classification tasks
- **Key Insights:**
    - Strong correlation between brain activity patterns and ADHD diagnosis in females
    - **Differentiated patterns** between male and female ADHD participants

---
## **🖼️ Impact Narrative**

**WiDS Challenge:**

1. **What brain activity patterns are associated with ADHD; are they different between males and females, and, if so, how?**

   Our analysis of functional connectivity suggests that individuals with ADHD exhibit distinct neural activity patterns compared to those without ADHD. In particular, disruptions in the communication between regions responsible for attention, executive control, and emotional regulation were observed. Furthermore, preliminary results indicate potential sex-specific differences, where females may show alternative connectivity patterns or varying levels of activation in certain brain networks compared to males. These differences could be influenced by developmental, hormonal, or genetic factors, emphasizing the need for tailored approaches when studying ADHD.

2. **How could your work help contribute to ADHD research and/or clinical care?**

   By applying advanced machine learning techniques to functional MRI connectome data alongside socio-demographic variables, our work enhances the understanding of ADHD’s neural underpinnings. This improved insight can aid in:
   
   - **Early Diagnosis:** Providing biomarkers based on brain connectivity that may support earlier and more accurate identification of ADHD.
   - **Personalized Treatment:** Informing the development of targeted therapies that account for individual differences, including sex-specific neural profiles.
   - **Research Advancement:** Offering a framework for integrating complex neuroimaging data with clinical metrics, thereby contributing to a more comprehensive model of ADHD that can be validated in larger studies.

Overall, our findings aim to bridge the gap between neuroscience research and clinical practice, ultimately improving the diagnostic and therapeutic landscape for ADHD.


As you answer the questions below, consider using not only text, but also illustrations, annotated visualizations, poetry, or other creative techniques to make your work accessible to a wider audience.

## 🚀 Next Steps & Future Improvements

- Address model limitations by incorporating additional datasets (e.g., more brain scans)
- Explore **neural network models** for deeper insights into complex patterns
- Further evaluate **model fairness** and **bias** across different demographics

---
**Address the following:**

  

* What are some of the limitations of your model?  
* What would you do differently with more time/resources?  
* What additional datasets or techniques would you explore?

  ---


## 📄 References & Additional Resources

### Datasets and Competitions
- **Healthy Brain Network (HBN) Data:** [HBN Official Website](https://www.healthbrainnetwork.org)
- **Kaggle Competition:** [WiDS Datathon 2025](https://www.kaggle.com/competitions/widsdatathon2025)

### Organization & Task Management

<div align="left">
  <a href="https://www.notion.so">
    <img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white" alt="Notion Logo" width="110" />
  </a>
  <span style="margin-left: 10px; vertical-align: middle;">
   Notion is an all-in-one productivity platform with AI-powered features that enhance teamwork and collaboration. It helps manage tasks, due dates, and organization effectively.
  </span>
</div>


