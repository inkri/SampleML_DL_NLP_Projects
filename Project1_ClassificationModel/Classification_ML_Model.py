#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Setting Path
import os
os.chdir("C:\Work_Life_Passion\Mission2021\ExtraProjects\Project1_ClassificationModel\data")
os.getcwd()


# In[2]:


#Load Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


#Load Dataset
data=pd.read_csv("nki_cleaned.csv")
data.head()


# In[4]:


#Dataset and Data Type information
print('Dataset Shape(Rows,Columns): ',data.shape)
print('Data Types: ',data.dtypes)
print('Features: ',data.columns)
data.iloc[:, : 17].describe()


# In[5]:


#Convert categorical to factor
data['eventdeath'].replace({False: 0, True: 1}, inplace=True)
data['chemo'].replace({False: 0, True: 1}, inplace=True)
data['hormonal'].replace({False: 0, True: 1}, inplace=True)
data['amputation'].replace({False: 0, True: 1}, inplace=True)
print(data['eventdeath'].value_counts())
print(data['chemo'].value_counts())
print(data['hormonal'].value_counts())
print(data['amputation'].value_counts())


# #### 1. In the following two exercises, we will look for your ability to extract details from various sources of information and break down complex problems
# #### a. Write a brief note on the dataset and the research work from where this dataset originates. We expect no exploratory analysis.

# #### Answer a. The dataset was shared by Devi Ramanan, and she is VP - Product Collaborations @ Ayasdi - AI/MLDataset contains patient information, treatment and survival, along with Network built using only gene expression.It has records of 272 breast cancer patients.Dataset Link: https://data.world/deviramanan2016/nki-breast-cancer-data

# #### b. How do you think the creators of this dataset have calculated the values in column 17 to 1554?And what does the column names mean? Write a brief explanation based on your understanding and study. 

# #### Answer b.Data from columns 17 to 1554 contains genes data, This identified a unique subgroup of Estrogen Receptor-positive (ER+) breast cancers that express high levels of c-MYB and low levels of innateinflammatory genes via d Progression Analysis of Disease (PAD).

# #### 2. In the following two exercises, we will look for your ability to work with data and build machine learning models

# #### a. Perform some exploratory analysis using the dataset which helps in understanding the data and the problem at hand. The notebook in the link gives some starting point.

# #### Answer a.Exploratory Data Analysis (EDA)

# In[11]:


#Count of Death and Alive
print(data['eventdeath'].value_counts())
#Visualize this count 
sns.countplot(data['eventdeath'],label="Count")

#Finding:Death Rate is 28% as per the Barchart


# In[20]:


#Data Variation 
data1=data.iloc[:, : 17]
data1 = data1.drop(['patient', 'id'], axis=1)
data1=data1[['age','eventdeath','timerecurrence','survival','barcode','esr1','grade']]
data1.plot(kind='scatter', x='timerecurrence', y='age') 
plt.grid()
plt.show()

#Finding: We can see timerecurrence are higher on the older people and mostly timerecurrence less than 12


# In[30]:


#Data Variation 
data1=data.iloc[:, : 17]
data1 = data1.drop(['patient', 'id'], axis=1)
data1=data1[['age','eventdeath','timerecurrence','survival','barcode','esr1','grade']]

sns.FacetGrid(data1,hue='eventdeath', size=8).map(sns.distplot,'esr1').add_legend()

#Finding: Higher density of death when esr1 less than 0


# In[29]:


#Data Variation 
data1=data.iloc[:, : 17]
data1 = data1.drop(['patient', 'id'], axis=1)
data1=data1[['age','eventdeath','timerecurrence','survival','barcode','esr1','grade']]

sns.FacetGrid(data1,hue='eventdeath', size=8).map(sns.distplot,'age').add_legend()

#Finding: We can people with cancer are in all age group


# In[31]:


#Data Variation 
data1=data.iloc[:, : 17]
data1 = data1.drop(['patient', 'id'], axis=1)
data1=data1[['age','eventdeath','timerecurrence','survival','barcode','esr1','grade']]

sns.violinplot(x='eventdeath', y='barcode',data=data1)
plt.legend
plt.show()

#Finding: We see high death density around 6500 barcode and less death density around 4000


# In[33]:


#Data Variation 
data1=data.iloc[:, : 17]
data1 = data1.drop(['patient', 'id'], axis=1)
data1=data1[['age','eventdeath','timerecurrence','survival','barcode','esr1','grade']]


sns.jointplot(x='age',y='timerecurrence',data=data1,kind='kde')
plt.grid()
plt.show()

#Finding: People are mostly having cancer at age around 50 and timerecurrence of 8


# In[23]:


# create boxplots: grade VS eventdeath
plot = sns.boxplot(x='eventdeath', y='grade', data=data, showfliers=False)
plot.set_title("Graph of grade mean vs eventdeath of cancer")

#Finding: People dying are mostly having grade higher than 2


# In[8]:


#Data Variation wrt eventdeath
#Multivariate Analysis
data1=data.iloc[:, : 17]
data1 = data1.drop(['patient', 'id'], axis=1)
sns.pairplot(data1,hue="eventdeath")

#Findings: Via scater and density plot we can see features value variation for death/alive


# In[9]:


#Seeing the correlation b/w faetures
data1=data.iloc[:, : 17]
data1 = data1.drop(['patient', 'id'], axis=1)
corrMatrix = data1.corr()
plt.figure(figsize=(26, 18))
sns.heatmap(corrMatrix, annot=True)
plt.show()

#Findings: survival and timerecurrence are negatively correlated


# In[10]:


#Important features:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})

data1 = data.iloc[:, : 17]
data1['eventdeath'] = data1['eventdeath'].astype('category')
y = data1['eventdeath']
X = data1.drop(['patient', 'id','eventdeath'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
rf.feature_importances_


sorted_idx = rf.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")

#Findings: Top contributor to the death/life are 'timerecurrence','survival','barcode','esr1','grade'


# In[271]:


#Clustering for segmentation
from sklearn.cluster import KMeans
data1=data.iloc[:, : 17]
data1 = data1.drop(['patient', 'id','eventdeath'], axis=1)
data1=data1[['timerecurrence','survival','barcode','esr1','grade']]
data1.head()
# k means: K=2 As person can be alive or dead
kmeans = KMeans(n_clusters=2, random_state=0)
data1['cluster'] = kmeans.fit_predict(data1)
# define and map colors
colors = ['#DF2020', '#81DF20']
data1['c'] = data1.cluster.map({0:colors[0], 1:colors[1]})
plt.scatter(data1.survival, data1.esr1, c=data1.c, s=data1.grade, alpha = 1)

#Finding: Seeing feature values variation via clustering 


# In[272]:


data.iloc[:, : 17].describe()


# In[ ]:





# #### b.Choose 3 of your favorite machine learning models apart from the one given in the notebook and apply it on the dataset. Perform an analysis to compare the models using the model performance metrics. Also, show if hyper-parameter tuning improves the results in your comparison. You are free to build an ensemble model if you think it is going to be better than the individual models.

# In[273]:


#As we have noted there are 1500+ features,It is resulting in high dimensionality.
#We will do PCA for all the Numberical features:


# In[274]:


#Data Preprocessing:
#Data Type conversions: Converting numerical value to categorical 
#Missing value: No Missing value
#Outliers: No outliets
#Data Transformation: Performing PCA and Feature Scaling


# In[275]:


#PCA:
data['eventdeath'] = data['eventdeath'].astype('category')
data['chemo'] = data['chemo'].astype('category')
data['hormonal'] = data['hormonal'].astype('category')
data['amputation'] = data['amputation'].astype('category')
data.describe()


# In[276]:


df.columns


# In[34]:


#Missing Value Count
data.isnull().sum()


# In[277]:


#Data seperation: Numberic and Categorical and for PCA
df=data
df1=df[['patient','id','eventdeath','chemo','hormonal','amputation','age','survival','timerecurrence','histtype','diam','posnodes','grade','angioinv','lymphinfil','barcode']]
df2=df.drop(['patient','id','eventdeath','chemo','hormonal','amputation','age','survival','timerecurrence','histtype','diam','posnodes','grade','angioinv','lymphinfil','barcode'], axis=1)
df2.shape


# In[278]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df2)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PCA1', 'PCA2'])
principalDf.head()
finaldf = pd.concat([df1, principalDf], axis = 1)


# In[279]:


#PCA 1 and 2 of all numerical values for eventdeath(0:Alive/1:Death)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PCA1', fontsize = 15)
ax.set_ylabel('PCA2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r',  'b']
for target, color in zip(targets,colors):
    indicesToKeep = finaldf['eventdeath'] == target
    ax.scatter(finaldf.loc[indicesToKeep, 'PCA1']
               , finaldf.loc[indicesToKeep, 'PCA2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[280]:


finaldf.head()


# In[281]:


finaldf.dtypes


# In[282]:


#One-Hot Encoding for categorical features
# generate binary values using get_dummies
finaldf = pd.get_dummies(finaldf, columns=["chemo"], prefix=["chemo"] )
finaldf = pd.get_dummies(finaldf, columns=["hormonal"], prefix=["hormonal"] )
finaldf = pd.get_dummies(finaldf, columns=["amputation"], prefix=["amputation"] )
#finaldf = finaldf.drop(['chemo', 'hormonal','amputation'], axis=1)
finaldf.head()


# In[283]:


finaldf.describe()


# In[286]:


#Features Scaling: Min/Max
from sklearn.preprocessing import MinMaxScaler
Scaleddata=finaldf[['age','survival','timerecurrence','diam','posnodes','grade','histtype','lymphinfil','barcode']]
# fit scaler on data
norm = MinMaxScaler().fit(Scaleddata)

# transform data
Scaleddata2 = norm.transform(Scaleddata)
Scaleddata2 = pd.DataFrame(data=Scaleddata2, columns=['age','survival','timerecurrence','diam','posnodes','grade','histtype','lymphinfil','barcode'])
Scaleddata2.head()


# In[287]:


finaldf = finaldf.drop(['age','survival','timerecurrence','diam','posnodes','grade','histtype','lymphinfil','barcode'], axis=1)
finaldf = pd.concat([finaldf, Scaleddata2], axis = 1)
finaldf.dtypes


# In[288]:


finaldf.head()


# In[289]:


# Dataset split for Model training and testing(80/20 split)
X = finaldf.drop(['patient', 'id', 'eventdeath'], axis=1)
y = finaldf['eventdeath']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# #### Model 1: RandomForest Classification

# In[290]:


#Model 1: RandomForest Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)
# metrics are used to find accuracy or error
from sklearn import metrics  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

# classification report for precision, recall f1-score and accuracy: ,labels=[0,1]
matrix = classification_report(y_test,y_pred)
print('Classification report : \n',matrix)


# In[291]:


#AUC Curve
from plot_metric.functions import BinaryClassification
# Visualisation with plot_metric
y_pred = clf.predict_proba(X_test)[:,1]
bc = BinaryClassification(y_test, y_pred, labels=["Class 1", "Class 2"])

# Figures
plt.figure(figsize=(5,5))
bc.plot_roc_curve()
plt.show()


# #### Model 2: KNN

# In[292]:


#Model 2: KNN
from sklearn.neighbors import KNeighborsClassifier
neighbors = np.arange(1, 30)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
  
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
      
    # Compute traning and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
  
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[293]:


#Model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
  
# Predict:
y_predKNN=knn.predict(X_test)

print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_predKNN))
# classification report for precision, recall f1-score and accuracy: ,labels=[0,1]
matrix = classification_report(y_test,y_predKNN)
print('Classification report : \n',matrix)


# In[294]:


#AUC Curve
from plot_metric.functions import BinaryClassification
# Visualisation with plot_metric
y_predKNN = knn.predict_proba(X_test)[:,1]
bc = BinaryClassification(y_test, y_predKNN, labels=["Class 1", "Class 2"])

# Figures
plt.figure(figsize=(5,5))
bc.plot_roc_curve()
plt.show()


# #### Model 3: SVM
# 

# In[295]:


#Support Vector Machines Model
from sklearn.svm import SVC  
#kernel: linear/poly/rbf/sigmoid : linear gives highest accuracy
svmclf = SVC(kernel='linear',probability=True) 
# fitting x samples and y classes 
svmclf.fit(X_train, y_train) 

# Predict:
y_predSVM=svmclf.predict(X_test)

print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_predSVM))
# classification report for precision, recall f1-score and accuracy: ,labels=[0,1]
matrix = classification_report(y_test,y_predSVM)
print('Classification report : \n',matrix)


# In[296]:


#AUC Curve
from plot_metric.functions import BinaryClassification
# Visualisation with plot_metric
y_predSVM = svmclf.predict_proba(X_test)[:,1]
bc = BinaryClassification(y_test, y_predSVM, labels=["Class 1", "Class 2"])

# Figures
plt.figure(figsize=(5,5))
bc.plot_roc_curve()
plt.show()


# In[297]:


#From above Model we can say RandomForest Model > SVM Model > KNN Model on the basis of accuracy.
#Deep Learning Model will not be that much impactful due to less no. of data/volumn


# #### 3.In taking a data science project from prototype to production, highlight at least five challenges that you have faced in your past projects.
# 

# In[298]:


#Five challenges faced while building ML/DL Model
#1. Data Integration- How we can integrate live data with the model
#2. Data Understanding- Understanding each features and its data variation or associations 
#3. Data Cleaning- Sampling the right dataset for Modeling, post features transformation/Missing value Imputation/Outliers removal/etc
#4. Live data pipeline creation for the model
#5. Making clients understand the differnce b/w tradition and Machine Learning approach for implementation and other


# In[ ]:





# In[ ]:




