# Predicting Obesity based on Lifestyle Choices

### Final Project, CS 9637 Introduction to Data Science, Western University

> #### **Note:**
>
> In order to learn and challenge myself,  I coded this project from scratch as much as I could, rather than use existing libraries. The code is in the **Master** branch. Now that I finished my project, I re-wrote my code to use existing libraries so that it follows industry-standard libraries and coding practices. It is much easier to read and follow compared to the version found in the **Master** branch. The re-written code is in the **Re-Factored** branch.
>
> ------
>
> #### What I Coded From Scratch:
>
> - GridSearch without Cross-Validation
> - Exhaustive search of optimal combinations of features, instead of using Recursive Feature Elimination (RFE)
> - Made my own pre-processing pipelines + pipes



### Abstract

**Background:** Obesity is an ever rising concern across numerous counties and, in the US, contributes to an estimated 112,000 deaths per year, all of which were preventable had they taken the necessary steps to reduce obesity onset. These increasing rates of obesity are also reported on a global scale by the World Health Organization, leaving many health professionals to view obesity as a very serious epidemic. Being able to predict the probability of succumbing to obesity based on one’s current lifestyle choices may allow individuals to take action and correct their habits in order to avoid obesity and all of its potential health ailments that may come with it. **Objective:** To develop a predictive model that can show the probability of someone falling into a class of obesity given that they maintain their current lifestyle choices and habits. Our motivation behind building this model is to hopefully provide incentives for someone using this model to correct their lifestyle choices if the model predicts that they may fall into a class of obesity given their inputed lifestyle. **Methods:** Obesity data used in training our models came from data containing estimations of obesity levels in people, ages from 14 to 61, from the countries of Mexico, Peru and Colombia, with diverse eating habits and physical condition. The data consisted of 2111 rows and 16 features. The target variable had 7 possible classes: Obesity I, Obesity II, Obesity III, Overweight I, Overweight II, Normal Weight, and Insufficient Weight. Data was split into training and testing data. Three supervised machine learning models were trained and evaluated (Logistic Regression, Random Forest, XGBoosting). Automated hyper-parameter tuning and feature-space tuning were employed to improve accuracy of our prediction models. **Results:** Selected model was a Random Forest-based Multi-Classifier that was hyper-parameter tuned and feature-tuned. Selected model achieved an accuracy of 86.55% on the test set. Averaged Precision score — averaged across the 7 classes — was 0.8; Recall score was 0.87, and F1-score was 0.86. Averaged Sensitivity and Specificity scores were 0.87 and 0.98. Averaged Balanced Accuracy was 0.92. Macro-Averaged AUROC was 0.98. **Conclusion:** We built a moderately accurate prediction model that classifies people’s risk of falling into a specific class of obesity (along with a normal weight class and underweight class) given that they maintain their existing lifestyle choices. Tuning strategies provided a decent improvement in performance. We achieved decent sensitivity and high specificity. Performance of our model is believed to be held back by the relatively small size of our dataset (especially when considering we have 7 classes to learn from) and, for any future work, acquisition of a larger dataset would be utmost priority to improve our model. Some limitations of our model are also discussed.

------

### Final Grade: 98/100

