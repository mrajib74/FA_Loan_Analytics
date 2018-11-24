
# coding: utf-8

# # Financial Analytics Assignment

# In[1]:


import pandas as pd # data maipulation and analysis
import numpy as np # mathematical function
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt # base graphics package
import seaborn as sns # grphics package
import statsmodels.api as sm #This is for regression
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import scipy as sc
import scipy.optimize as opt
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


# In[2]:


#Reading data from the csv file
loanstatus=pd.read_csv(r'D:\Rajib\XLRI\Elective\Analytics_Assignment\2014\File\LoanStats3c_final.csv',low_memory=False)


# In[3]:


print("Current shape of dataset :",loanstatus.shape)


# In[4]:


loanstatus.head()


# In[5]:


#Generates descriptive statistics that summarize the central tendency,
#dispersion and shape of a dataset’s distribution, excluding NaN value
loanstatus.describe(include = 'all')
#u.to_csv(r"D:\Rajib\XLRI\Elective\Analytics_Assignment\2014\File\output_2-sum.csv", index=False)


# In[6]:


#Concise summary of a DataFrame
loanstatus.info()


# In[7]:


#Column labels to use for resulting frame
loanstatus.columns


# In[8]:


#########Identify outlier & remove the same
plt.boxplot(loanstatus['loan_amnt'])


# In[9]:


plt.boxplot(loanstatus['annual_inc'])


# In[10]:


from scipy.stats import normaltest, iqr
cutoffincome=np.percentile(loanstatus['annual_inc'],75)+ 1.5* iqr(loanstatus['annual_inc'])
print(cutoffincome)


# In[11]:


#loanstatus=loanstatus[loanstatus['annual_inc']<cutoffincome]
#loanstatus.shape


# In[12]:


#Mapped Target variable to 0 to 1
loanstatus['loan_status']=loanstatus['loan_status'].map({"Fully Paid":0, "Charged Off":1} )


# In[13]:


#Return the first 5 rows.
loanstatus.head()


# In[14]:


## Remove columns  not related to our target variable
removefields=[
        "id","member_id","emp_title","verification_status","url","desc"
         ,"purpose","title","zip_code","addr_state","sub_grade","term","issue_d"
         ,"pymnt_plan","dti","delinq_2yrs","earliest_cr_line","revol_bal_joint"
         ,"sec_app_earliest_cr_line","sec_app_inq_last_6mths","sec_app_mort_acc"
         ,"sec_app_open_acc","sec_app_revol_util","sec_app_open_act_il"
         ,"sec_app_num_rev_accts","sec_app_chargeoff_within_12_mths"
         ,"sec_app_collections_12_mths_ex_med","sec_app_mths_since_last_major_derog"
         ,"hardship_flag","hardship_type","hardship_reason","hardship_status"
         ,"deferral_term","hardship_amount","hardship_start_date","hardship_end_date"
          ,"payment_plan_start_date","hardship_length","hardship_dpd","hardship_loan_status","orig_projected_additional_accrued_interest"
          ,"hardship_payoff_balance_amount","hardship_last_payment_amount","disbursement_method"
         ,"debt_settlement_flag","debt_settlement_flag_date","settlement_status"
         ,"settlement_date","settlement_amount","settlement_percentage","settlement_term"
         ,"policy_code","dti_joint","last_credit_pull_d"
        ,"last_pymnt_d","next_pymnt_d","revol_util","initial_list_status"
        ,"application_type","annual_inc_joint","verification_status_joint"
        ,"tot_cur_bal","open_acc_6m","open_act_il","open_il_12m","open_il_24m","mths_since_rcnt_il"
        ,"total_bal_il","il_util","open_rv_12m","open_rv_24m","max_bal_bc","all_util"
        ,"inq_fi","total_cu_tl","inq_last_12m","delinq_amnt","num_tl_120dpd_2m"
        ,"num_tl_30dpd","num_tl_90g_dpd_24m","num_tl_op_past_12m"
        ,"mths_since_recent_revol_delinq","num_accts_ever_120_pd"
        ,"mths_since_last_record","mths_since_last_delinq","percent_bc_gt_75"
        ,"mths_since_last_major_derog","mths_since_recent_bc_dlq","mths_since_recent_inq"
        ,"mo_sin_old_il_acct","mo_sin_rcnt_tl"
        ,"mths_since_recent_bc"

]
finalloan = loanstatus.drop(labels = removefields, axis = 1) 
finalloan.to_csv(r"D:\Rajib\XLRI\Elective\Analytics_Assignment\2014\File\output.csv", index=False)


# In[15]:


print("Current shape of dataset :",finalloan.shape)


# In[16]:


features = ["funded_amnt","emp_length","annual_inc","home_ownership","grade",
            "last_pymnt_amnt", "mort_acc", "pub_rec", "open_acc","num_actv_rev_tl",
            "mo_sin_rcnt_rev_tl_op","mo_sin_old_rev_tl_op","bc_util","bc_open_to_buy",
            "avg_cur_bal","acc_open_past_24mths",
            "inq_last_6mths","total_acc","acc_now_delinq","tax_liens","int_rate"
           ,"loan_status"] 
loanstatusfinal = finalloan[features] #19 features with target var"loan_status"
print("Current shape of dataset :",loanstatusfinal.shape)


# In[17]:


##Filling Missing values
loanstatusfinal['annual_inc']=loanstatusfinal['annual_inc'].fillna(loanstatusfinal['annual_inc'].median())
loanstatusfinal['inq_last_6mths']=loanstatusfinal['inq_last_6mths'].fillna(loanstatusfinal['inq_last_6mths'].median())
loanstatusfinal['open_acc']=loanstatusfinal['open_acc'].fillna(loanstatusfinal['open_acc'].median())
loanstatusfinal['pub_rec']=loanstatusfinal['pub_rec'].fillna(loanstatusfinal['pub_rec'].median())
loanstatusfinal['total_acc']=loanstatusfinal['total_acc'].fillna(loanstatusfinal['total_acc'].median())
loanstatusfinal['acc_now_delinq']=loanstatusfinal['acc_now_delinq'].fillna(loanstatusfinal['acc_now_delinq'].median())
loanstatusfinal['tax_liens']=loanstatusfinal['tax_liens'].fillna(loanstatusfinal['tax_liens'].median())
loanstatusfinal['bc_open_to_buy']=loanstatusfinal['bc_open_to_buy'].fillna(loanstatusfinal['bc_open_to_buy'].median())
loanstatusfinal['bc_util']=loanstatusfinal['bc_util'].fillna(loanstatusfinal['bc_util'].median())
loanstatusfinal['avg_cur_bal']=loanstatusfinal['avg_cur_bal'].fillna(loanstatusfinal['avg_cur_bal'].median())


# In[18]:


#Data Transformation
loanstatusfinal['grade'] = loanstatusfinal['grade'].map({"A":7,"B":6,"C":5,"D":4,"E":3,"F":2,"G":1})
loanstatusfinal['home_ownership'] = loanstatusfinal['home_ownership'].map({"MORTGAGE":6,"RENT":5,"OWN":4,"OTHER":3,"NONE":2,"ANY":1})
#loanstatusfinal['emp_length'] = loanstatusfinal['emp_length'].replace({'years':'','year':'',' ':'','<':'','\+':'','n/a':'0'}, regex = True)
loanstatusfinal['emp_length'] = loanstatusfinal['emp_length'].replace({'years':'','year':'',' ':'','<':'','\+':'','n/a':'0'}, regex = True)
loanstatusfinal['emp_length'].fillna(0, inplace = True)
#loanstatusfinal["emp_length"] = loanstatusfinal['emp_length'].apply(lambda x:int(x))
loanstatusfinal['int_rate'] = loanstatusfinal['int_rate'].replace({'%':''}, regex = True)

loanstatusfinal.to_csv(r"D:\Rajib\XLRI\Elective\Analytics_Assignment\2014\File\output_1.csv", index=False)


# In[19]:


#Concise summary of a DataFrame
loanstatusfinal.info()


# In[20]:


##Feature scaling
#loanstatusfinal[fields].count()
scl  =  StandardScaler()
fields = loanstatusfinal.columns.values[:-1]
#fields
standardloanstatusfinal = pd.DataFrame(scl.fit_transform(loanstatusfinal[fields]), columns = fields)
standardloanstatusfinal['loan_status'] = loanstatusfinal['loan_status']
standardloanstatusfinal.to_csv(r"D:\Rajib\XLRI\Elective\Analytics_Assignment\2014\File\output_2.csv", index=False)


# In[21]:


#Concise summary of a DataFrame
standardloanstatusfinal.info()


# In[22]:


# sampled  dataset
loanstatus_0 = standardloanstatusfinal[standardloanstatusfinal["loan_status"]==0]
loanstatus_1 = standardloanstatusfinal[standardloanstatusfinal["loan_status"]==1]
subset_of_loanstatus_0 = loanstatus_0.sample(n=15500)
subset_of_loanstatus_1 = loanstatus_1.sample(n=15500)
standardloanstatusfinal = pd.concat([subset_of_loanstatus_1, subset_of_loanstatus_0])
standardloanstatusfinal = standardloanstatusfinal.sample(frac=1).reset_index(drop=True)
print("Current shape of dataset :",standardloanstatusfinal.shape)
#standardloanstatusfinal.head()


# In[23]:


#Compute pairwise correlation of columns, excluding NA/null values
standardloanstatusfinal.corr()


# In[24]:


#ROC Curve plot function
import seaborn as sns
sns.set('talk', 'whitegrid', 'dark', font_scale=1, font='Ricty',rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
def plotAUC(truth, pred, lab):
    fpr, tpr, _  =roc_curve(truth,pred)
    roc_auc =auc(fpr, tpr)
    lw = 2
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color= c,lw=lw, label= lab +'(AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve') #Receiver Operating Characteristic 
    plt.legend(loc="lower right")


# In[25]:


#Confusion Matrix Visualizaion function
import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(model, normalize=False): # This function prints and plots the confusion matrix.
    cm = confusion_matrix(Ytest, model, labels=[0, 1])
    classes=["Will Pay", "Will Default"]
    cmap = plt.cm.Blues
    title = "Confusion Matrix"
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[26]:


X=standardloanstatusfinal.drop(['loan_status'],axis=1)
y=standardloanstatusfinal['loan_status']
#X.info()


# In[27]:


#Shapiro-Wilk test

stat, p  = st.shapiro(X)

print('Statistics=%.3f, p=%.3f' % (stat, p))

#Since our p-value is much less than our Test Statistic, we have good evidence to  reject the null hypothesis at the 0.05 significance leve


# In[28]:


# ## Heteroskedasticity tests
# 
# Breush-Pagan test
#name = ['Lagrange multiplier statistic', 'p-value',   'f-value', 'f p-value']
from scipy.stats import boxcox

results = smf.OLS(y,X).fit()
print(results.summary())

#test= sms.het_breushpagan(results.resid, results.model.exog)
#test


# In[29]:


## Split te data in Training set & Testing Set
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from scipy import stats
Xtrain,Xtest,Ytrain, Ytest=train_test_split(X,y,test_size=0.3, random_state=42)


# In[30]:


#Feature Selection using RFE (Recursive Feature Elimination)
from sklearn import linear_model
from  sklearn.feature_selection  import RFE
# create the RFE model and select 3 attributes
clfLR = linear_model.LogisticRegression(C=1e30)
clfLR.fit(Xtrain,Ytrain)
rfe = RFE(clfLR, 15)
rfe = rfe.fit(standardloanstatusfinal.iloc[:,:-1].values, standardloanstatusfinal.iloc[:,-1].values)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
print(rfe.n_features_to_select)
# ["funded_amnt","emp_length","annual_inc","home_ownership","grade",
#            "last_pymnt_amnt", "mort_acc", "pub_rec", "open_acc","num_actv_rev_tl",
#            "mo_sin_rcnt_rev_tl_op","mo_sin_old_rev_tl_op","bc_util","bc_open_to_buy",
#            "avg_cur_bal","acc_open_past_24mths",
#            "inq_last_6mths","total_acc","acc_now_delinq","tax_liens","int_rate"
#           ,"loan_status"] 
#print(ranking(list(map(float, rfe.ranking_)), colnames, order=-1))


# In[31]:


features = ["funded_amnt","emp_length","annual_inc","grade","last_pymnt_amnt","mort_acc","pub_rec","open_acc"
,"mo_sin_old_rev_tl_op","bc_open_to_buy","avg_cur_bal","acc_open_past_24mths","inq_last_6mths","total_acc","loan_status"
]
Xtrain, Xtest = Xtrain[features[:-1]], Xtest[features[:-1]]
standardloanstatusfinal = standardloanstatusfinal[features]
print(Xtrain.shape)
print(standardloanstatusfinal.shape)


# In[32]:


###Heatmap
dataViz = standardloanstatusfinal
sns.set_context(context='notebook')
fig, ax = plt.subplots(figsize=(10,10)) 
corr = dataViz.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 15, s=75, l=40,n=9, center="dark",as_cmap=True)

sns.heatmap(corr, cmap=cmap,linewidths=1, vmin=-1, vmax=1, square=True, cbar=True, center=0, ax=ax, mask=mask)


# # Models

# In[33]:



#K Nearest Neighbors(KNN) 
knn=KNeighborsClassifier(n_neighbors=5,weights='uniform', algorithm='auto')


# In[34]:


knn.fit(Xtrain,Ytrain)


# In[35]:


knn.score(Xtrain,Ytrain)


# In[36]:


knn.score(Xtest,Ytest)


# In[37]:



knnpred=knn.predict(Xtest)
knnpredictproba = knn.predict_proba(Xtest)[:,1]
print(knnpredictproba)
KNNAcc = accuracy_score(Ytest,knnpred)
print("KNN accuracy is ",KNNAcc)
#print(accuracy_score(Ytest,ypred))
plotAUC(Ytest,knnpredictproba,'K Nearest Neighbors')
plt.show()
plt.figure(figsize=(6,6))
plot_confusion_matrix(knnpred, normalize=True)
plt.show()


# In[38]:


from sklearn.metrics import classification_report,confusion_matrix


# In[39]:


print(classification_report(Ytest,knnpred))


# In[40]:


from sklearn.linear_model import  LogisticRegression


# In[41]:


####Logistic Regression
clfLR=LogisticRegression()
clfLR.fit(Xtrain,Ytrain)
#clfLR.fit(X_train,y_train)
LRPredict = clfLR.predict_proba(Xtest)[:,1]
LRPredictbin = clfLR.predict(Xtest)
LRAccuracy = accuracy_score(Ytest,LRPredict.round())
print("Logistic regression accuracy is ",LRAccuracy)

plotAUC(Ytest,LRPredict,'Logistic Regression')
plt.show()
plt.figure(figsize=(6,6))
plot_confusion_matrix(LRPredictbin, normalize=True)
plt.show()


# In[42]:


#Random Forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#randomForest = RandomForestClassifier(bootstrap=True,criterion = "gini",max_features=rand.best_estimator_.max_features,random_state=0 )
randomForest=RandomForestClassifier(n_estimators=100,criterion = "gini")
randomForest.fit(Xtrain,Ytrain)
rfPredict = randomForest.predict(Xtest)
rfPredictproba = randomForest.predict_proba(Xtest)[:,1] #for ROC curve
rfAccuracy = accuracy_score(Ytest,rfPredict)
rocscore = metrics.roc_auc_score(Ytest,rfPredict)
print("Random Forest Accuracy is ", rfAccuracy)
plotAUC(Ytest,rfPredictproba, 'Random Forest')
plt.show()
plt.figure(figsize=(6,6))
plot_confusion_matrix(rfPredict, normalize=True)
plt.show()


# In[43]:


fig, ax = plt.subplots()
width=0.35
ax.bar(np.arange(len(features)-1), randomForest.feature_importances_, width, color='G')
ax.set_xticks(np.arange(len(randomForest.feature_importances_)))
ax.set_xticklabels(Xtrain.columns.values,rotation=90)
plt.title('Feature Importance from DT')
ax.set_ylabel('Normalized Gini Importance')


# In[44]:


#Support Vector Machines(SVM)
from sklearn import svm

clfsvm = svm.SVC(kernel = "rbf")
clfsvm.fit(Xtrain,Ytrain)
predictionssvm = clfsvm.predict(Xtest)
predictprobasvm = clfsvm.decision_function(Xtest)
SVMAccuracy = accuracy_score(Ytest,predictionssvm)
print("SVM accuracy is ",SVMAccuracy)
plotAUC(Ytest,predictprobasvm, 'SVM')
plt.show()
plt.figure(figsize=(6,6))
plot_confusion_matrix(predictionssvm, normalize=True)
plt.show()


# In[45]:


#Multi-Layer Perceptron Classifier

from sklearn.neural_network import MLPClassifier
clfNN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clfNN.fit(Xtrain,Ytrain)     
predictNN = clfNN.predict(Xtest)
predictprobaNN = clfNN.predict_proba(Xtest)[:,1]
NNAccuracy = accuracy_score(Ytest,predictNN)
print("MLP accuracy is ", NNAccuracy)

plotAUC(Ytest,predictprobaNN,'MLP')
plt.show()
plt.figure(figsize=(6,6))
plot_confusion_matrix(predictNN, normalize=True)
plt.show()


# In[46]:


#####Comparison between  different Model
plotAUC(Ytest,rfPredictproba, 'Random Forest')
plotAUC(Ytest,LRPredict,'Logistic Regression')
plotAUC(Ytest,predictprobasvm, 'SVM')
plotAUC(Ytest,knnpredictproba,'K Nearest Neighbors')
plotAUC(Ytest,predictprobaNN,'MLP')
plt.show()


# In[47]:


#Precision,recall,F1score for all models¶
print("RF",classification_report(Ytest, rfPredict, target_names=None))
print("SVM",classification_report(Ytest, predictionssvm, target_names=None))
print("LR",classification_report(Ytest, LRPredictbin, target_names=None))
print("KNN",classification_report(Ytest, knnpred, target_names=None))
print("MLP",classification_report(Ytest, predictNN, target_names=None))

