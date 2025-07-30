# Model-Comparison
**Final Year Project as an undegraduate of a Student of University of Ilorin Studying Statistics**

# Project Topic: Comparative Performance Analysis of Some Selected Classification Models
## Introduction
In the dynamic landscape of data classification, the choice of an appropraite ML model plays a crucial in determining prediction accuracy. This study presents a comprehensive comparison of several widely used models: Logistic Regression, K-Nearest Neighbors(KNN), Naive Bayes, Random Forest, and Decision Tree. These models have attracted considerable attention due to their diverse applications and unique characteristics.
 **Logistic Regression**,a fundamental algorithm in supervised learning, is valued for its simplicity and interpretability [Bui et al., 2020]. It is particularly effective for binary classification tasks, utilizing a linear decision boundary to separate classes within a dataset.
 **K-Nearest Neighbors (KNN)** is a non-parametric algorithm known for its intuitive approach to classification. It predicts the class of a data point based on the majority class among its closest neighbors. KNN is highly effective when sufficient data is available, and it performs particularly well with larger sample sizes due to its sensitivity to local data structure [Alsabhan et al., 2022].
**Naïve Bayes**, a probabilistic classifier based on Bayes’ Theorem, assumes feature independence. While this assumption may not always hold in real-world scenarios, the model is computationally efficient and performs surprisingly well with small datasets [Bui et al., 2020]. However, its performance tends to decline as dataset complexity and volume increase.
**Random Forest**, an ensemble learning method introduced by Breiman [Breiman, 2001], aggregates the results of multiple decision trees to enhance accuracy and reduce overfitting. It is known for its robustness, especially when dealing with high-dimensional data or complex interactions among features. Nevertheless, its performance may vary depending on the size and nature of the dataset.
**Decision Tree**, although simple and interpretable, often suffers from high variance and overfitting, particularly with small datasets. However, as sample size increases, its predictive performance tends to improve significantly, making it a viable option when adequate data is available [Li et al., 2020].
This study evaluates these five models using performance metrics such as Precision, Recall, F1 Score, and AUC across varying sample sizes. By examining their behavior under different data conditions, this research aims to guide practitioners in selecting suitable models for effective and scalable classification tasks.
## Statement Problem
Choosing an appropriate classification model is a critical challenge in statistical modeeling and machine learning. With the vast array of algorithms available- each designed with different assumptions, computational complexities, and data requirements, selecting the most suitable model for a given dataset is far from striaghtforward. Practitioners frequently default to well-known models or rely on past experience and software tools, often without a thorough understanding of how model performance can vary under different data conditions.  Moreover, the performance of ML models is often evaluated using accuracy, which may provide a misleading picture, especially in imbalanced datasets where one class dominates. In such cases, metrics like precision, Recall, F1 Score and AUC (Area Under the Curve) provide a more holistic view of performance (Saito & Rehmsmeier, 2015; Powers, 2011). However, Many comparative studies continue to overlook these comprehensive metrics. Despite the growing application of ML in sensitive and high-stakes domains such as healthcare, finance, law enforcement, and socail science, there is alack of systematic studies comparing standard classification algorithms under controlled and replicable conditions. Most existing comparisons are conducted on real-world datasets that come with inherent noise, missing data, and unbalanced distiributions--factors that can obscure meaningful insights about model behavior.

 To address these gaps, this study  adopts a simulation-based approach that allows full control over key data generation parameters, such as a feature independence, outcome distributions, and sample size. This design enables a focused comparison of five commonly used classification models; Logistic Regression, Naive Bayes, Random Forest, K-Nearest Neighbours, and Decision Tree, across sample sizes ofv 100, 500, 1000, and 2000 observations. The use of multiple performance metrics ensure a comprehensive evaluation of each models's strengths and limitation.

 This study focuses on a simulation-based comparative analysis of five aforementioned classification models. Through the controlled generation of synthetic datasets with varying sample sizes and known properties, the study aims to evaluate and compare model performance using standard evaluation metrics. This approach allows fot the isolation of key vaariables and provide insights into how different models scale aand generalize across differnt data conditions

 ## Aim and Objectives of the Study
 ### Aim
 To conduct a simulation-based evaluation of five selected machine learning classification models and compare their performance across different sample sizes using standard performance metrics.
 ### Objectives
 To achieve the stated aim, the study will:
1.	To develop and apply five supervised classification models: logistics regression, naive bayes, k-nearest neighbors, decision tree, and random forest, using synthetically generated datasets that serve controlled statistical conditions in ensuring consistency and comparability in model evaluation.
2.	Evaluate the classification performance of each model using Precision, Recall, F1 Score, and AUC (Area Under the Curve) and investigate how each model’s performance changes as the sample size increase) n= 100, 500, 1000,2000)
## Scope of the Study
This study is strictly limited to binary classification problems under simulated data conditions. The scope includes:
1.	Data Generation: Ten predictor variables (X_1  to X_10) will be generated from a multivariate normal distribution with mean of 0 and variance of 1, assuming independence among features
2. Target Variable: The binary outcome (Y) is generated using a logistic function applied to a weighted sum of the predictors, with β coefficients randomly selected between -0.5 and 0.5.
3. Sample Sizes: Datasets will be generated for four different sample sizes; 100, 500, 1000, and 2000 observations
4. Classification Models: Five Models will be studied; Logistics Regression, Naïve Bayes, K-Nearest Neighbors, Decision Tree, and Random Forest.
5. Performance Metrics: Each model will be evaluated using Precision, Recall, F1 Score, and Area Under the Curve (AUC)

- Precision: This is the proportion of true positive predictions out of all positive predictions made by the model. It simply measures the accuracy of positive predictions. To correctly identify any false positive in the prediction.
- Recall: This is also known as sensitivity/true positive rate is the proportion of true positive predictions from all actual positive samples in the dataset. It measures the model’s ability to identify all positive  instances and is critical when the cost of false negative is high. A higher recall means the model is capturing more actual positive, which means fewer false negative.
- F1 Score: This is a measure of model’s accuracy that takes into account both precision and recall, where the goal is to classify instances correctly as positive or negative. A high --F1 score indicates a balanced performance across precision and recall for a given binary classification problem
- AUC: Area under the ROC curve, measuring overall performance of a classification model by calculating the area under the ROC curve (Receiver Operating Characteristic curve). AUC summarizes this True Positive Rate(TPR) and False Negative Rate (FPR) into a single number that tells us how well the model separates the positive class from the negative class. An AUC close to 1 indicates excellent separation, while 0.5 indicates random guessing, less than 0.5 indicates model perform worse than random.
## Data Analysis and Interpretation
This presents the results obtained from the simulation study described in Chapter Three. The performance of five selected classification models—Logistic Regression, Naïve Bayes, K-Nearest Neighbours (KNN), Decision Tree, and Random Forest—was assessed across four different sample sizes (n = 100, 500, 1000, 2000). The models were evaluated using four key performance metrics: Precision, Recall, F1 Score, and Area Under the Curve (AUC). The results are presented in graphical formats, followed by interpretations and comparisons.

**Precision**

<img width="547" height="358" alt="mEnoch" src="https://github.com/user-attachments/assets/7528fd1c-2e26-4768-8516-a66c47f272aa" />

1. Naïve Bayes: The Top Performer
Naïve Bayes (blue line) consistently achieved the highest precision across all sample sizes. While it showed minor fluctuations at smaller sample sizes, its precision stabilized as the sample size increased. This suggests that Naïve Bayes is the most reliable model for maintaining high precision, especially with larger datasets.
2. Logistic Regression: A Strong Alternative
Logistic Regression (green line) performed closely to Naïve Bayes, particularly at smaller sample sizes. Although Naïve Bayes pulled ahead as sample size increased, Logistic Regression remained a solid second choice, showing steady improvement and maintaining high precision throughout.
3. KNN: Competitive but Slightly Less Stable
KNN (yellow line) started with low precision at smaller sample sizes but showed significant improvement as sample size increased. It eventually reached a similar performance level to Logistic Regression. However, its slight instability at lower sample sizes suggests it benefits from larger datasets for optimal performance.
4. Random Forest: Strong Growth but Not the Best
Random Forest (pink line) started with low precision but rapidly improved as sample size increased. While it showed steady growth, it could not surpass Naïve Bayes or Logistic Regression, positioning it as a moderately effective option.
5. Decision Tree: The Weakest Performer
Decision Tree (red line) exhibited the most variability and the lowest precision, particularly for smaller datasets. It gradually improved but remained the least precise model overall, suggesting it is highly sensitive to sample size and may not be the best choice for precision-focused tasks.


Conclusion

•	Naïve Bayes is the most precise and reliable model, especially as sample size increases.
•	Logistic Regression follows closely, maintaining competitive precision.
•	KNN and Random Forest improve with larger datasets but are slightly less stable.
•	Decision Tree struggles the most, making it the weakest model for precision.

**Recall**

<img width="547" height="358" alt="MEnoch_Recall" src="https://github.com/user-attachments/assets/d6b952f3-5baa-4059-ba91-14a0c6e97e12" />

Interpretation of Recall vs. Sample Size
Recall measures a model’s ability to correctly identify positive instances, making it crucial for applications where missing positive cases is costly. The graph illustrates how recall changes with different sample sizes for five models: Decision Tree, KNN, Logistic Regression, Naïve Bayes, and Random Forest.
1. Top Performers: Naïve Bayes & Logistic Regression
•	Naïve Bayes and Logistic Regression consistently achieved the highest recall across all sample sizes.
•	At smaller sample sizes, both models peaked around 0.7, indicating their strong ability to detect positive cases early.
•	As the dataset size increased, their recall remained stable, showing that they generalize well with larger datasets.
2. Moderate Performers: KNN & Random Forest
•	KNN performed well initially but slightly declined as sample size increased, suggesting that its sensitivity to positive cases may decrease with larger datasets.
•	Random Forest started with a low recall but improved steadily, showing that it benefits from more data to enhance positive case detection.
3. Weak Performer: Decision Tree
•	Decision Tree had a good recall at smaller sample sizes, but as the dataset grew, its recall dropped significantly, making it the least reliable model for recall.
•	This indicates that Decision Tree struggles to maintain sensitivity as complexity increases, likely due to overfitting on smaller datasets and poor generalization to larger ones.

Conclusion

•	Naïve Bayes and Logistic Regression are the best models for recall, ensuring a high proportion of true positives are captured.
•	KNN and Random Forest offer moderate recall performance, with Random Forest improving at larger sample sizes.
•	Decision Tree is the weakest model in terms of recall, making it unsuitable for applications where missing positive cases is critical.

**F1 Score**

<img width="547" height="358" alt="MEnoch F1" src="https://github.com/user-attachments/assets/28424abe-aeb2-41bf-aa14-1ce269b9a2f3" />

Interpretation of F1 Score vs. Sample Size Analysis
The F1 score is a crucial metric that balances precision and recall, ensuring a model effectively captures positive cases while minimizing false positives. The comparison of Decision Tree, KNN, Logistic Regression, Naïve Bayes, and Random Forest reveals key performance patterns across different sample sizes.
1. Best Performer: Logistic Regression
Logistic Regression achieved the highest F1 score for most sample sizes, particularly excelling in small datasets. It quickly reached its peak (~0.7) at 500 observations and maintained a stable performance throughout. Even at larger sample sizes (1000+), it remained highly competitive, briefly surpassed by Naïve Bayes but still performing consistently well. Its ability to generalize effectively makes it the most reliable and balanced model in this analysis.
2. Strong Contender: Naïve Bayes
Naïve Bayes closely followed Logistic Regression, performing similarly at smaller sample sizes but slightly outperforming it around 1000 observations. However, Logistic Regression regained the lead at higher sample sizes, showing greater overall stability. Naïve Bayes still demonstrated strong generalization capabilities and was a close second, making it a strong alternative for classification tasks.
3. Moderate Performers: Random Forest & KNN
-	Random Forest started with a lower F1 score at small sample sizes but gradually improved as data increased. This indicates it benefits significantly from larger datasets but struggles with smaller ones. It remained stable but never outperformed Logistic Regression or Naïve Bayes.
- KNN excelled at small sample sizes but slightly declined as the dataset grew. This suggests KNN is effective for smaller datasets but may struggle with larger, more complex data due to noise sensitivity.
4. Weak Performer: Decision Tree
Decision Tree exhibited high variability, performing well at small sample sizes but experiencing a sharp decline as data increased. This suggests it overfits small datasets but fails to generalize to larger ones. Among all models, Decision Tree was the least reliable, making it the weakest in this comparison
Conclusion

Logistic Regression emerged as the best model, followed closely by Naïve Bayes. Random Forest and KNN had moderate performance, with Random Forest benefiting from larger datasets while KNN was better suited for smaller ones. Decision Tree struggled with generalization, making it the least stable model.

**AUC**

<img width="547" height="358" alt="MEnoch AUC" src="https://github.com/user-attachments/assets/e4aa7c06-abc5-4fbd-b7c0-01f78ab54c81" />

Interpretation of AUC vs Sample Sizes Analysis
AUC (Area Under the Curve) evaluates a model’s ability to distinguish between classes. Higher AUC indicates better classification performance. The AUC trends for the models across different sample sizes reveal key insights:
1.	Best Performers: Naïve Bayes & Logistic Regression
 - At 500 observations, both models peaked, with Naïve Bayes slightly higher (~0.75) than Logistic Regression.
 - 	At 1000 observations, their performance dropped slightly but remained competitive.
 - 	At 2000 observations, both models improved again, with Naïve Bayes slightly surpassing Logistic Regression.
 - 	This indicates that both models generalize well, with Naïve Bayes having a slight edge in larger datasets.
2.	Moderate Performers: Random Forest & KNN
 - Random Forest improved gradually, stabilizing with increasing sample size.
 - 	KNN fluctuated, initially performing well but showing instability at different sample sizes.
 - Their inconsistent trends suggest that these models require careful tuning to maintain robust AUC scores.
3.	Weakest Performer: Decision Tree
   -	It showed early improvement, then dropped sharply at 1000 observations, indicating poor generalization.
   -	AUC improved at 2000 observations, but it remained lower than the other models.
This suggests that Decision Trees struggle with scalability and tend to overfit smaller datasets.
Conclusion
- Naïve Bayes and Logistic Regression are the strongest models for AUC, excelling in distinguishing between classes.
- 	Random Forest and KNN perform moderately, but KNN is more unstable.
- Decision Tree is the weakest, struggling to maintain a consistent AUC score.
Thus, for optimal classification performance, Naïve Bayes or Logistic Regression is recommended.

## Summary
5.2 Summary of the Study
The primary aim of this study was to conduct a simulation-based comparative analysis of five widely used machine learning classification models—Logistic Regression, Naïve Bayes, K-Nearest Neighbours (KNN), Decision Tree, and Random Forest—under varying sample sizes. The data generation process was strictly controlled, employing a multivariate normal distribution to create ten independent predictor variables, with a binary outcome derived through a logistic transformation.
Four sample sizes (n = 100, 500, 1000, and 2000) were considered to simulate different data availability scenarios. The performance of each model was assessed using four evaluation metrics: Precision, Recall, F1 Score, and Area Under the Curve (AUC).
The methodology emphasized fairness and replicability by using consistent data generation processes, uniform model training-validation splits (80/20), and repeated simulations to reduce the impact of randomness. R programming language was used to implement the simulations and performance evaluations.
### Key Findings and Interpretation
#### Performance of Models by Metric
##### Precision:
- Naïve Bayes consistently achieved the highest precision across all sample sizes.
- Logistic Regression followed closely, particularly outperforming others at lower sample sizes.
- Random Forest showed significant improvement with larger sample sizes.
- KNN displayed inconsistent performance, improving with sample size but less stable overall.
- Decision Tree exhibited the lowest and most volatile precision, especially at small sample sizes.
  
##### Recall:
-	Naïve Bayes and Logistic Regression demonstrated strong and stable recall across all sample sizes.
-	Random Forest gradually improved in recall, benefiting from more data.
-	KNN had relatively stable recall but did not excel.
-	Decision Tree performed poorly, particularly with larger sample sizes, indicating overfitting and poor generalization.

##### F1 Score:
- Logistic Regression led in F1 Score, indicating strong balance between precision and recall.
- Naïve Bayes was a close second and showed robustness across all sample sizes.
- Random Forest showed gradual improvement.
- KNN had modest performance and a tendency to degrade with larger sample sizes.
- Decision Tree's performance declined with sample size, highlighting instability.
  
##### AUC (Area Under the Curve):
- Naïve Bayes and Logistic Regression had the highest AUCs, demonstrating excellent class separation.
- 	Random Forest performed well at higher sample sizes.
- 		KNN was inconsistent.
- 		Decision Tree had the lowest AUC and generalization ability.

#### Performance Trends Across Sample Sizes
- Small Sample Sizes (n = 100):
 - Naïve Bayes and Logistic Regression were the most effective.
 - 	Random Forest and Decision Tree underperformed due to overfitting or variance.
 - 	KNN struggled due to the sparsity of neighbours in low data volumes.
•	Medium Sample Sizes (n = 500–1000):
o	All models showed improvement.
o	Random Forest became more competitive.
o	Logistic Regression and Naïve Bayes maintained consistent dominance.
•	Large Sample Sizes (n = 2000):
o	Model performances stabilized.
o	Naïve Bayes and Logistic Regression remained the top performers.
o	Random Forest approached similar performance but still trailed slightly.
o	KNN improved modestly.
o	Decision Tree continued to lag behind in every metric.

5.4 Conclusions
Based on the results, the following conclusions can be drawn:
1.	Naïve Bayes and Logistic Regression are the most robust and consistent models across different sample sizes and evaluation metrics. Their simplicity and low variance make them highly suitable for both small and large datasets.
2.	Random Forest, while more complex, shows promise with larger datasets but may require more computational resources and tuning to match the performance of simpler models.
3.	K-Nearest Neighbours is highly sensitive to data characteristics and sample size. It performs adequately but lacks the robustness seen in Naïve Bayes and Logistic Regression.
4.	Decision Tree, although interpretable, struggles with generalization, especially as dataset size increases. It is highly prone to overfitting and should be used cautiously or in combination with ensemble methods like Random Forest.
5.	Sample size significantly influences model performance. Models with high variance benefit greatly from increased data, while simpler models like Logistic Regression and Naïve Bayes maintain performance even with limited data.

5.5 Recommendations
5.5.1 Practical Recommendations
•	For practitioners dealing with small datasets, Naïve Bayes or Logistic Regression is highly recommended due to their robustness, simplicity, and interpretability.
•	For larger datasets with complex interactions, Random Forest can be a competitive choice, provided there is enough computational capacity and careful tuning is applied.
•	Avoid using Decision Tree as a standalone model, especially with larger datasets. If interpretability is required, consider ensemble methods that retain tree-based structure but reduce overfitting.
•	KNN should be used cautiously in high-dimensional data or with large sample sizes due to scalability issues. It can be effective in well-clustered and low-dimensional settings.
•	Always evaluate models using multiple metrics (Precision, Recall, F1, AUC) to gain a comprehensive understanding of performance—especially when dealing with class imbalance.

5.5.2 Academic Recommendations
•	Simulation-based studies should be integrated into ML education to help students understand model behaviours under controlled conditions.
•	Emphasis should be placed on evaluating models beyond accuracy, particularly in sensitive domains like healthcare and finance.
Benchmarking practices must become standard in applied research for reliable model comparisons.
