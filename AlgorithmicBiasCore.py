
# coding: utf-8

# # Algorithmic Bias - Core Code
# Some code to get started on the Algorithmic Bias assignment. 

# # Rahul Praveen
# ## 16203022

# In[ ]:


import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
bcDB = datasets.load_breast_cancer()


# In[ ]:


bcDF = pd.DataFrame(bcDB.data, columns= list(bcDB['feature_names']))
bcDF['target'] = pd.Series(bcDB.target)
bcDF = bcDF.sort_values(by = ['target'])
bcDF = bcDF.reset_index(drop=True)
bcDF.head(5)


# In[ ]:


vc = bcDF['target'].value_counts()
for i,j in enumerate(bcDB.target_names):
    print (vc[i],j)


# In[ ]:


y = bcDF.pop('target').values
X = bcDF.values
X.shape, y.shape


# ## $k$-NN
# Malignant is the minority class at ~40%.  
# $k$-NN classifier picks up this under-representation and accentuates it,  
# predicting just 36% malignant. 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

kNN = KNeighborsClassifier(n_neighbors=3)


# Hold Out validation for KNN

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
y_pred = kNN.fit(X_train, y_train).predict(X_test)
print(X_train.shape,X_test.shape)

y_test.sum()/len(y_test)
print("KNN Hold-out validation :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# In[ ]:


from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mnb = GaussianNB()
y_pred = mnb.fit(X_train, y_train).predict(X_test)
breastCancer_NB = mnb.fit(X_train, y_train)
y_dash = breastCancer_NB.predict(X_test)

y_test.sum()/len(y_test)
print("Naive - Bayes Hold-out validation :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
y_pred = dtree.fit(X_train, y_train).predict(X_test)
y_test.sum()/len(y_test)
print("Decision Trees Hold-out validation :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# In[ ]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state = 0, solver = 'lbfgs', multi_class = 'multinomial')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
y_pred = clf.fit(X_train, y_train).predict(X_test)
y_test.sum()/len(y_test)
print("Logistic Regression Hold-out validation :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))

models = [mnb,kNN,dtree,clf]


# ## We can see that the none of the methods are giving us an exact estimate. This proves that the classifiers are biased towards the majority in the imbalanced data that we have. 
# 
# ## We can learn more about the bias in the 4-fold cross validation done below :

# In[ ]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
scoring = {'tp' : make_scorer(tp), 'tn' : make_scorer(tn),
           'fp' : make_scorer(fp), 'fn' : make_scorer(fn)}

folds = 4
v = 0


# In[ ]:


from sklearn.metrics import confusion_matrix
for m in models:
    cv_results = cross_validate(m, X, y, cv= folds,scoring=scoring, return_train_score=False, 
                                    verbose = v, n_jobs = -1)
    fp_rate = cv_results['test_fp'].sum()/(cv_results['test_fp'].sum()+cv_results['test_tn'].sum())
    tp_rate = cv_results['test_tp'].sum()/(cv_results['test_tp'].sum()+cv_results['test_fn'].sum())
  
    print("{} x CV {:22} FP: {:.2f}  TP: {:.2f}".format(folds, type(m).__name__, fp_rate, tp_rate)) 


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_dash, classes=['Malignant','Benign'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_dash, classes=['Manignant','Benign'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ## Assessment :
# ### We can see that the false positives (Benign) are much higher than the false negatives (Malignant), which means that the classifier is being positive biased (Since Benign is a majority value.) 
# ### To fix this bias, we need to balance out the data so there is no clear minority or majority within the sample data used. 
# ### From my understanding and research done, I beleive that downsampling or upsampling the given dataset can remove the bias up to a certain extent by adding or removing values from the minority or majority class respectively.

# In[ ]:


bcDF2 = pd.DataFrame(bcDB.data, columns= list(bcDB['feature_names']))
bcDF2['target'] = pd.Series(bcDB.target)
bcDF2 = bcDF2.sort_values(by = ['target'])
bcDF2.head(5)


# In[ ]:


benign = bcDF2[bcDF2.target == 1]
manignant = bcDF2[bcDF2.target == 0]


# ## Upsampling the majority Benign data

# In[ ]:


from sklearn.utils import resample
manignant_upsampled = resample(manignant, replace = True, n_samples= len(benign), random_state =2)


# In[ ]:


upsampled = pd.concat([benign, manignant_upsampled])
upsampled.target.value_counts()


# ## We can see that the dataset is balanced 
# ## Hold - out validations -

# In[ ]:


y_train = upsampled.target
X_train = upsampled.drop('target', axis = 1)


# ## KNN Hold - out validation on downsampled data

# In[ ]:


y_pred = kNN.fit(X_train, y_train).predict(X_test)
print(X_train.shape,X_test.shape)

y_test.sum()/len(y_test)
print("KNN Hold-out validation :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# Clear Improvement can be seen in the KNN results after the upsampling

# ## Decision Tree Hold - out validation on upsampled data

# In[ ]:


y_pred = dtree.fit(X_train, y_train).predict(X_test)
y_test.sum()/len(y_test)
print("Decision Trees Hold-out validation :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# Decision tree gives a worse result after upsampling

# ## Logistic Regression algorithm Hold - out validation on upsampled data

# In[ ]:


y_pred = clf.fit(X_train, y_train).predict(X_test)
y_test.sum()/len(y_test)
print("Logistic Regression Hold-out validation :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# ## Naive - Bayes

# In[ ]:


y_pred = mnb.fit(X_train, y_train).predict(X_test)
y_test.sum()/len(y_test)
print("Naive - Bayes Hold-out validation :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# ## Ruuning tests on a different data set 

# ## I have used a banknote dataset that I found online. The dataset gives information pn the findings of fake notes. 0 denotes authentic notes and 1 denotes Inauthentic notes.

# In[ ]:


banknoteData = pd.read_csv('data_banknote_authentication.csv')


# In[ ]:


banknoteData.head()


# In[ ]:


banknoteData2 = banknoteData.copy()
banknoteData2.head(5)


# 0 - Authentic note, 1 - inauthentic note

# In[ ]:


n = banknoteData.shape[0]
np = banknoteData['Class'].sum()
nn = n - np
print('{} Samples'.format(n))
print('{} inauthentic \n{} Authentic'.format(np,nn))


# In[ ]:


y = banknoteData.pop('Class').values
X = banknoteData.values


# ## I will carry out hold - out validations on this data set

# ## KNN

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)


# In[ ]:


y_pred = kNN.fit(X_train, y_train).predict(X_test)
print(X_train.shape,X_test.shape)

y_test.sum()/len(y_test)
print("KNN Hold-out validation on New dataset :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# ## Logistical Regression

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)


# In[ ]:


y_pred = clf.fit(X_train, y_train).predict(X_test)
y_test.sum()/len(y_test)
print("Logistic Regression Hold-out validation on new dataset :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# ## Decision Tree

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)


# In[ ]:


y_pred = dtree.fit(X_train, y_train).predict(X_test)
y_test.sum()/len(y_test)
print("Decision Trees Hold-out validation on new Dataset :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# ## Naive - Bayes

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)


# In[ ]:


y_pred = mnb.fit(X_train, y_train).predict(X_test)
y_test.sum()/len(y_test)
print("Naive - Bayes Hold-out validation on new Dataset :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))

bankData_NB = mnb.fit(X_train, y_train)
y_dash = bankData_NB.predict(X_test)


# In[ ]:


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
scoring = {'tp' : make_scorer(tp), 'tn' : make_scorer(tn),
           'fp' : make_scorer(fp), 'fn' : make_scorer(fn)}

folds = 4
v = 0


# In[ ]:


for m in models:
    cv_results = cross_validate(m, X, y, cv= folds,scoring=scoring, return_train_score=False, 
                                    verbose = v, n_jobs = -1)
    fp_rate = cv_results['test_fp'].sum()/(cv_results['test_fp'].sum()+cv_results['test_tn'].sum())
    tp_rate = cv_results['test_tp'].sum()/(cv_results['test_tp'].sum()+cv_results['test_fn'].sum())
  
    print("{} x CV {:22} FP: {:.2f}  TP: {:.2f}".format(folds, type(m).__name__, fp_rate, tp_rate)) 


# In[ ]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2, suppress =2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_dash, classes=['Malignant','Benign'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_dash, classes=['Manignant','Benign'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ## We will upsample this data as well

# In[ ]:


authentic = banknoteData2[banknoteData2.Class == 0]
inauthentic = banknoteData2[banknoteData2.Class == 1]


# In[ ]:


inauthentic_upsampled = resample(inauthentic, replace = True, n_samples= len(authentic), random_state =2)


# In[ ]:


upsampled2 = pd.concat([authentic, inauthentic_upsampled])
upsampled2.Class.value_counts()


# ## The data is now balanced

# In[ ]:


y_train = upsampled2.Class
X_train = upsampled2.drop('Class', axis = 1)


# ## KNN Hold - out validation on upsampled data

# In[ ]:


y_pred = kNN.fit(X_train, y_train).predict(X_test)
print(X_train.shape,X_test.shape)

y_test.sum()/len(y_test)
print("KNN Hold-out validation on second dataset after downsampling :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# ## Confusion matrix for upsampled KNN data

# In[ ]:


plot_confusion_matrix(y_test, y_pred, classes=['authentic','inauthentic'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=['authentic','inauthentic'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ## Logistical Regression

# In[ ]:


y_pred = clf.fit(X_train, y_train).predict(X_test)
y_test.sum()/len(y_test)
print("Logistic Regression Hold-out validation on new dataset :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# ## Confusion matrix for upsampled Logistical Regression data

# In[ ]:


plot_confusion_matrix(y_test, y_pred, classes=['authentic','inauthentic'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=['authentic','inauthentic'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ## Naive Bayes

# In[ ]:


y_pred = mnb.fit(X_train, y_train).predict(X_test)
y_test.sum()/len(y_test)
print("Naive - Bayes Hold-out validation on new Dataset :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# ## Confusion matrix for upsampled Naive Bayes data

# In[ ]:


plot_confusion_matrix(y_test, y_pred, classes=['authentic','inauthentic'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=['authentic','inauthentic'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ## Decision Tree

# In[ ]:


y_pred = dtree.fit(X_train, y_train).predict(X_test)
y_test.sum()/len(y_test)
print("Decision Trees Hold-out validation on new Dataset :")
print("Malignant in test set : %0.2f" % (1- (y_test.sum()/len(y_test))))
print("Predicted malignant : %0.2f" % (1- (y_pred.sum()/len(y_pred))))


# ## Confusion matrix for upsampled Decision tree data

# In[ ]:


plot_confusion_matrix(y_test, y_pred, classes=['authentic','inauthentic'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=['authentic','inauthentic'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ## From the above validations, it can be seen that upsampling gives 100% accuaracy in KNN validation and Decision tree validation. However, the method is less accurate for Naive bayes and Logical regression classifiers. 
# ## There is no logical conclusion here so as to determine a method to completely negate the bias.
# 
# ## However, upsampling has improved the KNN and Decision tree accuracies to a 100%. This is however not a conclusive finding, as the other classifiers did not show an improvement in accuracy. 
