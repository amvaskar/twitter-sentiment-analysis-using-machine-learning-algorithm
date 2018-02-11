# Twitter-sentiment-Analysis-Using-Machine-Learning-Algorithm
### Twitter Sentiment Analysis

This repo contains the code that will classify tweets as either _positive_ or _negative_ using various machine learning models such as Naive baeyes, Random Forest and others. 
Rather than relying on older algorithms such as VADER and Textblob, this method models a classifier from scratch which also takes into account the presence of features such as emoticons, punctuations, exclamations, hashtags and other characters to determine the sentiment of the tweet.

#### **Usage:**
###  Getting the data:

A training set of data from Stanford was used to train the model.It can be found [here.](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip) 
A different training set can also be used to train the model as well. The path for the training data is **data/training.csv**
Any other data set can be used and placed in the above path to obtain the prediciton.
The format for the data is as follows:

| tweet  | Sentiment |
| ------------- | ------------- |
| eeeks i like some1 :X :S. gossh, i promised myself not again! but just cant help it this time   | 1  |
| @euniqueflair You would be very happy with a Razer Mamba  http://is.gd/13mMG recommended!  | 1  |
| Freakin' crap! I just bit my tongue on accident.  | 0  |

### Algorithm And Tools Used

#### Naive Bayes classifiers: 
It is a probabilistic classifier with strong conditional independence
assumption that is optimal for classifying classes with highly dependent features.Naive Bayes
is a very simple classifier with acceptable results but not as good as other classifiers.
3

#### Random forests or random decision forests:
It are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision
trees at training time and outputting the class that is the mode of the classes (classification) or
mean prediction (regression) of the individual trees. Random decision forests correct for
decision trees' habit of overfitting to their training set.

#### k-nearest neighbors algorithm (k-NN):
It is a non-parametric method used for classification and
regression. In both cases, the input consists of the k closest training examples in the feature space.
The output depends on whether k-NN is used for classification or regression.

#### Scikit-learn:
It is a free software machine learning library for the Python programming
language. It features various classification, regression and clustering algorithms including
support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is
designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

#### Natural Language Toolkit : 
NLTK is a platform used for building Python programs that
work with human language data for applying in statistical natural language processing (NLP).
It contains text processing libraries for tokenization, parsing, classification, stemming,
tagging and semantic reasoning.


### Extracting features:

The following features were added to the existing dataset.
1. No. of positive emoticons(/data/positive.txt)
2. No. of negative emoticons(/data/negative.txt)
3. No. of exclamations
4. No. of hashtags
5. No. of question marks
6. No. of hyperlinks

Prior to fitting the model and using machine learning algorithms, we need to represent it in a bag-of-words model. 
For each unique tokenized word in a tweet, a unigram feature is created for the classifier. 
Plus to add the above features to the data set, the following script should be run on the data.
```
python add_features.py -f training.csv
```
_Note: Pass the name of the data file as the parameter for -f_

### Prediction:

After the Analytical-Base-Table is ready for sentiment classification, various machine-learning algorithms can be used to classify the tweets as positive or negative.
Run the script to check the results of the prediction.
```
python predict.py

```
Team Members:
Vaskar Maharjan 
Kritarth Acharya 
Suraj Duwal

