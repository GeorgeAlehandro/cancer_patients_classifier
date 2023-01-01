#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn import linear_model


# In[2]:


gene_expression_matrix = pd.read_csv("input/data.csv")


# In[3]:


gene_expression_matrix.head()


# In[4]:


labels = pd.read_csv("input/labels.csv")


# In[5]:


labels.head()


# In[6]:


Y = labels.iloc[:,1]


# In[7]:


X = gene_expression_matrix.iloc[:,1:]


# Understanding our data

# In[8]:


X.shape
print("The shape of the data:",X.shape)

# UMAP

# In[9]:


import umap.umap_ as umap
import seaborn as sns


# In[10]:


# Define UMAP
brain_umap = umap.UMAP(random_state=999, n_neighbors=30, min_dist=.25)

# Fit UMAP and extract latent vars 1-2
embedding = pd.DataFrame(brain_umap.fit_transform(X), columns = ['UMAP1','UMAP2'])


# In[11]:


# Produce sns.scatterplot and pass metadata.subclasses as color
sns_plot = sns.scatterplot(x='UMAP1', y='UMAP2', data=embedding,
                hue=Y.to_list(),
                alpha=1, linewidth=0, s=1)
# Adjust legend
sns_plot.legend(loc='center left', bbox_to_anchor=(1, .5))
# Save PNG
sns_plot.figure.savefig('umap_scatter.png', bbox_inches='tight', dpi=500)


# Splitting data (+ make sure acceptable label presentation)

# In[12]:


Y.value_counts().plot(kind="bar")


# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 10, stratify=Y)


# Let's test linear models

# In[14]:


Y_test.value_counts().plot(kind="bar")


# In[15]:


Y_train.value_counts().plot(kind="bar")


# PCA APPROACH

# In[16]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pca = PCA(n_components=2)
#x_standard = StandardScaler().fit_transform(X)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2',])


# In[17]:


finalDf = pd.concat([principalDf, Y], axis = 1)


# In[18]:


finalDf


# In[19]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['PRAD', 'LUAD', 'BRCA', 'COAD', 'KIRC']
colors = ['r', 'g', 'b','y','pink']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[20]:


pca.explained_variance_ratio_


# In[25]:


X_pca = finalDf.iloc[:,0:2]


# In[26]:


X_pca


# In[73]:


X_train, X_test, Y_train, Y_test = train_test_split(X_pca,Y, test_size = 0.3, random_state = 10, stratify=Y)
logreg_pca = linear_model.LogisticRegression(max_iter = 2500, C=1000000, random_state=0, multi_class='multinomial')
logreg_pca.fit(X_train, Y_train)
linearmodel_predictions_pca = logreg_pca.predict(X_test)
sns.heatmap(pd.crosstab(Y_test,linearmodel_predictions_pca), annot = True)


# In[30]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(Y_test, linearmodel_predictions_pca)

cm_display = ConfusionMatrixDisplay(cm).plot()


# In[31]:


accuracy_score(Y_test, linearmodel_predictions_pca)


# In[32]:


recall_score(Y_test, linearmodel_predictions_pca, average = "macro")


# In[92]:


print("Using 2 PCA dimensions, the multi linear regression model has an accuracy of:", accuracy_score(Y_test, linearmodel_predictions_pca))


# END OF FIRST APPROACH

# Initating second approach (feature selection, chi2...)

# In[33]:


from sklearn.feature_selection import chi2, SelectPercentile


# In[34]:


X_5_percent = SelectPercentile(chi2, percentile=5).fit_transform(X, Y)
X_5_percent.shape


# In[35]:


X_train, X_test, Y_train, Y_test = train_test_split(X_5_percent,Y, test_size = 0.3, random_state = 10, stratify=Y)


# In[36]:


logreg = linear_model.LogisticRegression(max_iter = 2500, C=1e40, random_state=0, multi_class='multinomial')


# In[37]:


logreg.fit(X_train, Y_train)


# In[38]:


linearmodel_predictions = logreg.predict(X_test)


# In[75]:


sns.heatmap(pd.crosstab(Y_test,linearmodel_predictions), annot = True)


# In[40]:




accuracy_score(Y_test, linearmodel_predictions)


# In[41]:


recall_score(Y_test, linearmodel_predictions, average = "macro")


# In[42]:


f1_score(Y_test, linearmodel_predictions, average = "macro")


# In[91]:


print("Using the top 5% genes selected by chi2, the multi linear regression model has an accuracy of:", accuracy_score(Y_test, linearmodel_predictions))


# Only using the first 10 genes

# In[84]:


X_only_10 = gene_expression_matrix.iloc[:,1:10]


# In[85]:


X_train_10, X_test_10, Y_train, Y_test = train_test_split(X_only_10,Y, test_size = 0.3, random_state = 10, stratify=Y)


# In[86]:


logreg_only_10 = linear_model.LogisticRegression(max_iter = 2500, multi_class='multinomial')


# In[87]:


logreg_only_10.fit(X_train_10,Y_train)


# In[88]:


linearmodel_10_predictions = logreg_only_10.predict(X_test_10)
sns.heatmap(pd.crosstab(Y_test,linearmodel_10_predictions), annot = True)


# In[89]:


accuracy_score(Y_test, linearmodel_10_predictions)


# In[90]:


print("Using only 10 genes, the multi linear regression model an accuracy score of:", accuracy_score(Y_test, linearmodel_10_predictions))


# In[49]:


N_TRAIN_SAMPLES = X_train.shape[0]
N_EPOCHS = 25
N_BATCH = 128
N_CLASSES = np.unique(Y_train)

scores_train = []
scores_test = []
mini_batch_index = 0
# EPOCH
epoch = 0
while epoch < N_EPOCHS:
    logreg = linear_model.LogisticRegression(max_iter = epoch, C=500, random_state=0, multi_class='multinomial')
    print('epoch: ', epoch)
    # SHUFFLING
    random_perm = np.random.permutation(X_train.shape[0])
    while True:
        logreg.fit(X_train, Y_train)
        mini_batch_index += N_BATCH

        if mini_batch_index >= N_TRAIN_SAMPLES:
            break

    # SCORE TRAIN
    scores_train.append(accuracy_score(logreg.predict(X_train), Y_train))

    # SCORE TEST
    scores_test.append(accuracy_score(logreg.predict(X_test), Y_test))

    epoch += 1
    print(epoch)
""" Plot """
fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].plot(scores_train)
ax[0].set_title('Train')
ax[1].plot(scores_test)
ax[1].set_title('Test')
fig.suptitle("Accuracy over epochs", fontsize=14)
plt.show()


# In[77]:


from sklearn.model_selection import KFold
list_t = []
list_training = []
list_testing = []
list_real_testing = []
kf = KFold(n_splits=10)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=10)
i = 1
logreg = linear_model.LogisticRegression(max_iter = 2500)
for train_index, test_index in kf.split(X_train):
    X_train_cross, X_test_cross = X_train.iloc[train_index,], X_train.iloc[test_index,]
    Y_train_cross, Y_test_cross = Y_train.iloc[train_index,], Y_train.iloc[test_index,]

    logreg = logreg.fit(X_train_cross,Y_train_cross)
    training_accuracy = accuracy_score(logreg.predict(X_train_cross),Y_train_cross)
    testing_accuracy = accuracy_score(logreg.predict(X_test_cross),Y_test_cross)
    realtesting_accuracy = accuracy_score(logreg.predict(X_test),Y_test)
    list_real_testing.append(realtesting_accuracy)
    list_t.append(i)
    list_training.append(training_accuracy)
    list_testing.append(testing_accuracy)
    i+=1
plt.plot(np.array(list_t), np.array(list_training), 'r') # plotting t, a separately
plt.plot(list_t, list_testing, 'b') # plotting t, b separately
plt.plot(list_t, list_real_testing, 'g') # plotting t, b separately
plt.show()


# In[54]:


from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer().fit(Y_train)
y_onehot_test = label_binarizer.transform(Y_test)


# In[55]:


import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
y_score = logreg.fit(X_train, Y_train).predict_proba(X_test)
RocCurveDisplay.from_predictions(
    y_onehot_test.ravel(),
    y_score.ravel(),
    name="micro-average OvR",
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
plt.legend()
plt.show()


# SVM

# In[59]:


from sklearn.svm import SVC


# In[60]:


svm_model = SVC()


# In[61]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 10, stratify=Y)


# In[62]:


svm_model.fit(X_train, Y_train)


# In[63]:


Y_svm_prediction = svm_model.predict(X_test)


# In[64]:


pd.crosstab(Y_test,Y_svm_prediction)


# In[65]:


accuracy_score(Y_test,Y_svm_prediction)


# In[ ]:


print("SVM has an accuracy score of:", accuracy_score(Y_test,Y_svm_prediction))


# Naive Bayes Gaussian

# In[66]:


from sklearn.naive_bayes import GaussianNB


# In[67]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 10, stratify=Y)


# In[68]:


gaussian_model = GaussianNB()
gaussian_model.fit(X_train, Y_train)
gaussian_prediction = gaussian_model.predict(X_test)


# In[69]:


pd.crosstab(Y_test,gaussian_prediction)


# In[70]:


accuracy_score(Y_test,gaussian_prediction)


# In[80]:


print("Naive Bayes has an accuracy score of:", accuracy_score(Y_test,gaussian_prediction))


# In[ ]:




