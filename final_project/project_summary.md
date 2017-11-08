# Identify Fraud from Enron Email
## Enron Project Details
### How do I Complete this Project?
This project is connected to the Intro to Machine Learning course, but depending on your background knowledge of machine learning, you may not need to take the whole thing to complete this project.

A note before you begin: the mini-projects in the Intro to Machine Learning class were mostly designed to have lots of data points, give intuitive results, and otherwise behave nicely. This project is significantly tougher in that we're now using the real data, which can be messy and does not have as many data points as we usually hope for when doing machine learning. Don't get discouraged--imperfect data is something you need to be used to as a data analyst! If you encounter something you haven't seen before, take a step back and think about a smart way around. You can do it!

### Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

### Resources Needed
You should have python and sklearn running on your computer, as well as the starter code (both python scripts and the Enron dataset) that you downloaded as part of the first mini-project in the Intro to Machine Learning course. You can get the starter code on git: git clone https://github.com/udacity/ud120-projects.git

The starter code can be found in the final_project directory of the codebase that you downloaded for use with the mini-projects. Some relevant files:

poi_id.py : Starter code for the POI identifier, you will write your analysis here. You will also submit a version of this file for your evaluator to verify your algorithm and results.

final_project_dataset.pkl : The dataset for the project, more details below.

tester.py : When you turn in your analysis for evaluation by Udacity, you will submit the algorithm, dataset and list of features that you use (these are created automatically in poi_id.py). The evaluator will then use this code to test your result, to make sure we see performance that’s similar to what you report. You don’t need to do anything with this code, but we provide it for transparency and for your reference.

emails_by_address : this directory contains many text files, each of which contains all the messages to or from a particular email address. It is for your reference, if you want to create more advanced features based on the details of the emails dataset. You do not need to process the e-mail corpus in order to complete the project.

### Steps to Success
We will provide you with starter code that reads in the data, takes your features of choice, then puts them into a numpy array, which is the input form that most sklearn functions assume. Your job is to engineer the features, pick and tune an algorithm, and to test and evaluate your identifier. Several of the mini-projects were designed with this final project in mind, so be on the lookout for ways to use the work you’ve already done.

As preprocessing to this project, we've combined the Enron email and financial data into a dictionary, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI label: [‘poi’] (boolean, represented as integer)

You are encouraged to make, transform or rescale new features from the starter features. If you do this, you should store the new feature to my_dataset, and if you use the new feature in the final algorithm, you should also add the feature name to my_feature_list, so your evaluator can access it during testing. For a concrete example of a new feature that you could add to the dataset, refer to the lesson on Feature Selection.

In addition, we advise that you keep notes as you work through the project. As part of your project submission, you will compose answers to a series of questions to understand your approach towards different aspects of the analysis. Your thought process is, in many ways, more important than your final project and we will by trying to probe your thought process in these questions.

## General Submission and Evaluation Overview
### General Submission and Evaluation Overview
Your submission will contain several files: the code/classifier you create and some written documentation of your work. We will evaluate your project according to the rubric here; only projects that satisfy all "meets expectations" items will pass. Please self-evaluate before you submit! If you don't think your project meets all the criteria, the project evaluator likely won't either.

### Submission
Ready to submit your project? Go back to your Udacity Home, click on the project, and follow the instructions to submit!

You can either send us a GitHub link of the files or upload a compressed directory (zip file).
Inside the zip folder include a text file with a list of Web sites, books, forums, blog posts, GitHub repositories etc that you referred to or used in this submission (Add N/A if you did not use such resources).
It can take us up to a week to grade the project, but in most cases it is much faster. You will receive an email when your submission has been reviewed.

If you are having any problems submitting your project or wish to check on the status of your submission, please email us at dataanalyst-project@udacity.com.

### Items to include in submission:
#### Code/Classifier
When making your classifier, you will create three pickle files (my_dataset.pkl, my_classifier.pkl, my_feature_list.pkl). The project evaluator will test these using the tester.py script. You are encouraged to use this script before submitting to gauge if your performance is good enough. You should also include your modified poi_id.py file in case of any issues with running your code or to verify what is reported in your question responses (see next paragraph). Notably, we should be able to run poi_id.py to generate the three pickle files that reflect your final algorithm, without needing to modify the script in any way.

If you have intermediate code that you would like to provide as supplemental materials, it is encouraged for you to save them in files separate from poi_id.py. If you do so, be sure to provide a readme file that explains what each file is for. If you used a Jupyter notebook to work on the project, make sure that your finished code is transferred to the poi_id.py script to generate your final work.

#### Documentation of Your Work
Document the work you've done by answering (in about one or two paragraphs each) the questions found here. You can write your answers in a PDF, text/markdown file, HTML, or similar format. The responses in your documentation should allow a reviewer to understand and follow the steps you took in your project and to verify your understanding of the methods you have performed.

#### Text File Listing Your References
A list of Web sites, books, forums, blog posts, github repositories etc. that you referred to or used in this submission (add N/A if you did not use such resources). Please carefully read the following statement and include it in your document “I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.

## Enron Submission Questions
A critical part of machine learning is making sense of your analysis process and communicating it to others. The questions below will help us understand your decision-making process and allow us to give feedback on your project. Please answer each question; your answers should be about 1-2 paragraphs per question. If you find yourself writing much more than that, take a step back and see if you can simplify your response!

When your evaluator looks at your responses, he or she will use a specific list of rubric items to assess your answers. Here is the link to that rubric: [Link](https://review.udacity.com/#!/projects/3174288624/rubric) Each question has one or more specific rubric items associated with it, so before you submit an answer, take a look at that part of the rubric. If your response does not meet expectations for all rubric points, you will be asked to revise and resubmit your project. Make sure that your responses are detailed enough that the evaluator will be able to understand the steps you took and your thought processes as you went through the data analysis.

Once you’ve submitted your responses, your coach will take a look and may ask a few more focused follow-up questions on one or more of your answers.  
We can’t wait to see what you’ve put together for this project!

## Questions & Answers
### Question No. 1
*Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?*

This project is about using machine learning techniques to indetify persons of interest (POIs) in the Enron dataset. "In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives." This data will be used in this project to build a person of interest indentifier.
There were a four outliers of which two are still included in this dataset. The ones removed were TOTAL and TRAVEL AGENCY IN THE PARK, which are both no natural persons. The ones still inclueded are SKILLING JEFFREY K as well as LAY KENNETH L which have very high payments but were in the center of the fraud and should be included. Also there were two parsing mistakes that needed to be fixed.

### Question No. 2
*What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.*

I ended up using all features exept the ones I did in the identifier. While I started the selection process using KBest which effectively selected all features with a p-value below 5% and identified SVC as the best fit algorithm.

| feature                 | score        | pvalue  |
| ----------------------- |------------- | -----   |
| sent_to_poi_ratio	      | 13.433	     | 0.0003  |
| restricted_stock	      | 8.769	     | 0.0036  |
| exercised_stock_options | 6.756	     | 0.0103  |
| recieved_from_poi_ratio | 6.005	     | 0.0155  |
| total_stock_value	      | 5.866        | 0.0167  |
| shared_receipt_with_poi | 5.678        | 0.0185  |
| total_payments	      | 4.872        | 0.0289  |

I played around with a lot of combinations of the top features in the list but ended up using all of the features included in the original dataset to achive precision and recall values above 0.3.
There are three measurable interactions with POIs in the dataset: sending, recieving emails as well as sharing a reciept with a POI. I translated those interactions ratios to be get a metric that is not biased by the total amount of interactions but based on the share of those with POIs as kind of a connectedness metric.


##### Question No. 3
*What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?*

I ended up using a Support Vector Machine Classifier after trying K-means Clustering and Random Forest as well. I chose those algorithms as they come from three different approches to identify patterns which I wanted to try on the dataset.

| Algorithm     | precision  | recall |
| ------------- | ---------- | ------ |
| KMeans        | 0.025      | 0.147  |
| SVC           | 0.265      | 0.609  |
| Random Forest | 0.312      | 0.160  |


##### Question No. 4
*What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).*

Parameter tuning is about choosing a set of optimal parameters for a learning algorithm. If the parameters are not tuned well the algortihm risks overfitting.
After chosing SVC as the algorithm to further tune I first played around with the features selected leading to include all original features in the model as those had the highest natural precision & recall. Then I moved on to optimise  parameters gamma (Kernel coefficient) and C (penalty). After reading about them and moving them around to find the optimum I eneded up achiving a precision & recall of over 0.3.

##### Question No. 5
*What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?*

Validation is about quantifying the performance of an algorithm based on confision matrixs metrics such as precision and recall. The enron dataset includes way more non-POI's than POI's, which needs to be considered when evaluating the classification algos. If an algo such as POI = False would be deployed the accuracy would already be at 86%, but precision at 0%.
This is why I chose to only optimise on precision and recall as mentioned above.


##### Question No. 6
*Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.*

The algorithms had the following mean precision and recall values of 1000 iterations:

| Algorithm     | precision  | recall |
| ------------- | ---------- | ------ |
| KMeans        | 0.025      | 0.147  |
| SVC           | 0.265      | 0.609  |
| Random Forest | 0.312      | 0.160  |

When optimising the performance of the SVC algorithm I chose `C=2000` and `gamma=0.0001` which lead to the following:

| Algorithm     | precision  | recall |
| ------------- | ---------- | ------ |
| SVC           | 0.314      | 0.630  |


Precision: How many selected items are relevant?  
Of all indentified POIs there are 31% which actually are POIs while 69% are non-pois.

Recall: How many relevant items are selected?
The model identified 63% of all POIs in the dataset.

##### Outlook

Given the bad precision of only 31% this identifier is mostly wrong in identifying POIs from the Enron dataset. Still with a recall of 69% it is better than guessing. It turnes out that identifying Fraud or Spamm which is based on highly skewed data is very hard. To increase the performance of identification I'd suggest to look for patterns in the text of the emails. Maybe there are certain word combinations that are mostly used by POIs.
The documentation on Spamm classification which is available could be of great help there as it handles a similar problem: identifying behavioural patterns of a small group of people.
