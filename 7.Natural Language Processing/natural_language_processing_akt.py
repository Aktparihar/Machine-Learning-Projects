# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk

#importing tools to help, these tools are lists of words(which are not relevant it called stop words)
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #thats the class for getting the root word
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    
    # Now splitting the review first into words and then if it is present in stopwords then we will remove it
    review = review.split() # Now review is a list
    ps = PorterStemmer() # Object of the stem class
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #shortcut for complete visit the every word in first review
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the bag of words of model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
# Now we need to include the dependent variable
y =  dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting BOW to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) 