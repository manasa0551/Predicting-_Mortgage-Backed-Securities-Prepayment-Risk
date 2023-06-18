# Predicting-Mortgage-Backed-Securities-Prepayment-Risk
  Personal Copy

### Project Team

Mentor: [Nour M. Ibrahim](https://github.com/Nour-Ibrahim-1290)

Content:
---------
1. Requirenment & Analysis
2. Project Planning
3. Design
4. Development (Coding & Implementation)
5. Deployment
6. Conclusion

 ![SDLC](https://user-images.githubusercontent.com/93732090/209706167-09404f7c-ef4f-47fc-8a54-3771bee17f5b.png)

## 1. Requirenment & Analysis
1.1 Introduction

1.2 About Data

1.3 Tools and Technologies

### 1.1 Introduction
-  A mortgage backed security (MBS) is an investment similar to a bond that i made up of a bundle of home loans bought from the banks that issued them.
- In this system, the loans issued by the bank is in turn sold to investors at a discounted rate to free up the bank funds.
- These loans are sold in the form of bonds by investment banks wherein loans are grouped together according to their type and quality.
- For the investor, an MBS is as safe as the mortgage loans that back it up.
- Mortgage-backed securities (MBS)
- Prepayment risk is the risk involved with the premature return of principal on a fixed-income security. When debtors return part of the principal early, they do not    have to make interest payments on that part of the principal.
- This means that if the loan issuer prepays the loan, the investors will stop receiving interest on those bonds.
- Hence, it is important to evaluate the prepayment risk on the MBS and thus this is the aim of our project.

### 1.2 About Data
-The data is obtained from Freddie Mac official portal for home loans.
- https://www.freddiemac.com/research/datasets
- The size of the home loans data is (291452 x 28).
- It contains 291452 data points and 28 columns or parameters which denote different features of the data.
### 1.3 Tools & Technologies
- During this project we're using Python for coding.
- Pandas module for Data Preprocessing
- Pandas, numpy, matlpotlib.pyplot, and seaborn for EDA.
- Pandas, numpy, PCA for Feature Engineering.
- Pandas, numpy, sklearn, pickle for Modeling, and Pipelining
- flask, HTML5, CSS for App development.
- AWS - EC2 for Deployment.

## 2. Project Planning
After Careful analysis of project requirenments, and the different attribtes defined on the dataset;
The project was planned to have the following Outputs:

1. Automated Assesment of...
      - EverDeliquent(ED)
      - Prepayment Amount
      - Preferred Return On Investment (PROI)
2. Web app for the comapny to use it for the assesment process...  
      - Enter a borrower assesment data manually --- v01
      - Upload a csv file of several borrowers and assess them at the same time ---- v02
      - Automated assesment for the latest added borrowers through Bondora's API ---- v03

## 3. Design
After careful examination of the data set, we decided to have thses Design Attributed:
**High Level Design**
1. a Classofication Pipeline to asses the Probability of Default.
2. Based on domain reaserch and weight of evidence techniaues, define 3 new algorithms to calculate EMA, ELA, and PROI.
3. a MultiRegression Pipeline to asses all three new defined attributes.
4. A Web App as stated in the requirenments by the Client, yet the set of attributes to be defined after throughtful analysis of the attributes provided of the dataset.

![Full Design - High Level](https://user-images.githubusercontent.com/93732090/209706230-5532394b-36c2-49af-85fb-b2a2531f7609.png)

**Low Level Design**


![AppCreation-flowchart](https://user-images.githubusercontent.com/93732090/209706284-59ee341a-4f5d-4c92-a1be-60dc1bb04ed1.png)


## 4. Development (Coding & Implementation)

4.1. Data Preprocessing.

4.2. Explaratory Data Analysis.

4.3. Feature Engineering

4.4. Classification Modeling (Probability of Default).

4.5. Target variable creation for risk evaluation and assesment.

4.6. Regression Modeling.

4.7. Pipelines Creation (Classification and Regression).

4.8. UI: App Creation.

### 4.1 Data Preprocessing:
- The dataset contains **28** Columns and  **291451** Rows Range Index.
- At the time of checking null values, we got null values in only one column that is 'SellerName' but not fill that column with other technique because it is not as much important.
- Changing  FirstPaymentDate and MaturityDate into date format
![Screenshot (410)](https://user-images.githubusercontent.com/121864196/223394656-ff578607-cfb4-4c46-96fa-f4eb8c4476be.png)

- Now, we have a clean optimized dataset, and we're ready for EDA.

**Developers:** All team created their version individually, and we decided upon which steps to finalize.

### 4.2 Exploratory Data Analysis (EDA):
- While examining the data set through visualizations, There're some interesting trends showed in the data as the next few images suggest...


**In the Categorical Attributes**


![c2](https://user-images.githubusercontent.com/70345245/223637713-899546a8-cf47-4b42-8b2d-8cba1fb5a513.PNG)

- Occupancy shoes us mortgage is owner occupied.


![c3](https://user-images.githubusercontent.com/70345245/223638093-26502ce5-7d6c-4b78-9532-dc149508ea36.PNG)

- Channel shows us the mortgage is in T(TPO Not Specified) region, secondaly in R(Retail) region.


![c4](https://user-images.githubusercontent.com/70345245/223638325-968787e0-246b-4ebf-a03d-5722bcf962fb.PNG)

- PPM shows us mortgage is a  Not PPM type. means borrower not obligated to pay a penalty in the event of certain repayments of principal.


![c5](https://user-images.githubusercontent.com/70345245/223638377-a165901c-dac3-425e-bab0-6a4974974af1.PNG)

- It shows us that mostly mortgage loans are not deliquent.


![c6](https://user-images.githubusercontent.com/70345245/223638493-064a9491-dad4-4049-97cc-ed4c0e150cfa.PNG)

- It indicates mortgage loan is mostly the Purchase(P) mortgage,then secondly No Cash-out Refinance & Cash-out Refinance mortgage loan.  




**In the Numerical Attributes**


![c1](https://user-images.githubusercontent.com/70345245/223638655-5d5b095f-79e1-4960-b950-c36218199656.PNG)




![image](https://user-images.githubusercontent.com/122533402/223630543-e5dbc3d6-941d-4e05-a992-7caa3723c6a9.png)

- The correlation matrix shows below the strong correlation between:-

(1)OCLTV & LTV

(2)LTV & MIP

(3)OCLTV & MIP




- Creating new columns for our target variable
![Screenshot (411)](https://user-images.githubusercontent.com/121864196/223395007-41474057-aab0-46ff-af5f-f87e2b7685bf.png)

- Here we make a 4 new columns as 'CreditRange','LTVRange','RepayRange' and 'LoanTenure' .
- After we're throughly know every attribute in the dataset, It's time for Feature Engineering...

**Developers:** All team created their version individually, and we decided upon which steps to finalize.

### 4.3 Feature Engineering

a. Drop the column

b. Categorical Feature Encoding

c. Mutual Information

d. Feature Selection 

e. Splitting Data into train and test set

f. SMOTE() technique

g. Feature Scaling

h. Feature Extraction and Dimensionality reduction using PCA

**a.Drop the Unwanted columns**

- Define Highly Correlated attributes and handle them to avoid intercorrelation.
```
# Now we can drop these features from our dataset
df=df.drop(['FirstPaymentDate','MaturityDate','OCLTV','PropertyState','PropertyType','PostalCode','ProductType','LoanSeqNum','SellerName','ServicerName','MonthsDelinquent'],axis = 1)
```

**b. Categorical feture Encoding**
```
# Label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

```
```
# Encoding categorical columns
df['Homebuyer'] = le.fit_transform(df['Homebuyer'])
df['CreditRange'] = le.fit_transform(df['CreditRange'])
df['RepayRange'] = le.fit_transform(df['RepayRange'])
df['LTVRange'] = le.fit_transform(df['LTVRange'])

```

**c.Mutual Information**
```
# Spliting the data set setting the Target Variable

X=df.drop(["EverDelinquent"],axis=1)
y=df['EverDelinquent']

#Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()
discrete_features = X.dtypes == int
```
```
# Calculating MI scores for data set Features

from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores,name="MI Scores", index=X.columns)
```
![c8](https://user-images.githubusercontent.com/70345245/223638578-31b65e8b-13ff-4790-9531-8f5ecad2235b.PNG)

mi_scores = mi_scores.sort_values(ascending=False)
mi_scores

![c7](https://user-images.githubusercontent.com/70345245/223638529-7c857fb2-03b4-4060-8d62-c365bc5700b2.PNG)

**d.Feature Selection**

```
# implementing statistical method to select independent features which have strong relationship with dependent feature.

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2,k=10)
fit = bestfeatures.fit(X,y)
dfscore = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscore],axis=1)
featureScores.columns =['Specs', 'Score']
```


```
# This method also helps in finding importance of each independent with the dependent feature

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(X,y)
```
![c9](https://user-images.githubusercontent.com/70345245/223638601-cecd7b89-1daa-4cd3-b380-cc8b1c6d2745.PNG)

**e. Splitting data into training and testing sets**
```
from sklearn.model_selection import train_test_split  
X=df.drop(["EverDelinquent"],axis=1)
y=df['EverDelinquent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)  

```
**f. Applying the smote technique**
```
from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_train_sm,y_train_sm = smote.fit_resample(X_train,y_train)
x_test_sm,y_test_sm = smote.fit_resample(X_test,y_test)

```
**g. Feature Scaling**
```
#Normalizing data by rescaling the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train_sm,x_test_sm)
X_train_scl = scaler.transform(x_train_sm)
X_test_scl = scaler.transform(x_test_sm)

```
**h. Feature Extraction and Dimensionality reduction using PCA**
```
from sklearn.decomposition import PCA
pca = PCA(n_components = 8)
pca_features = pca.fit_transform(X_train_scl)
sum(pca.explained_variance_ratio_)*100
```

![c11](https://user-images.githubusercontent.com/70345245/223640551-ca0ff72e-a6b9-43ab-8cef-49f1764142b1.PNG)


- Using 2-d PCA we're preseved **98.42%** of information.

 ![pca](https://user-images.githubusercontent.com/121864196/223495836-e96edc53-d5f8-4938-8b15-040e0d75b739.png)



- We have preserved 98% of the data to test after modeling.

**Developers:** All team created their version individually, and we decided upon which steps to finalize.

### 4.4 Classification Modeling 
- In this step we'll be training 2 different Models using default settings in scikit-leran, and with Hyperparameter tunning using RandomizedCV to select the highest performance model to intergrate later into the classification pipeline.
- We used classification_report(precision | recall | f1-score ), confusion_matrix, accuracy_score, and roc_auc_score metrics from sklearn.metrics to asses each model.
- The models used for classification are...
        1.RandomForest Classifer
        2.Support Vector Classifier
        
**a.RandomForest Classifier**
```
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier(max_samples = 0.75, random_state=42)
rfc.fit(X_train_scl,y_train_sm)

# Applying RandomForest Classifier on Test Data
y_pred = rfc.predict(X_test_scl)
```
```
# Checking out for the Confusion Matrix
```
![c](https://user-images.githubusercontent.com/70345245/223636122-c185adc4-1e96-4298-84f6-0509d1da8bde.PNG)


```
# Priniting Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test_sm, y_pred))
```
![c12](https://user-images.githubusercontent.com/70345245/223642052-0f872a8c-abdc-45b1-a68a-749f0bd9b42c.PNG)

```
# Checking for Accuracy of the Test Data
from sklearn import metrics
print("Accuracy %:",metrics.accuracy_score(y_test_sm, y_pred)*100)
```
Accuracy %: 84.60287025739352

```
# The AUC,ROC_curve
```
![c10](https://user-images.githubusercontent.com/70345245/223638617-3f77203f-7b05-4881-9e73-a63bf5530fbb.PNG)

# checking for the performance of RandomForestClassifier through Roc_curve,AUC
from sklearn.metrics import roc_curve, auc
false_positive_rate,true_positive_rate, thresholds =roc_curve(y_test_sm,y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
```
-Hyperparameter Tunning
-Using RandomizedSearch for Hyperparamter tuning for RandomForestClassifier
```
```
from sklearn.model_selection import RandomizedSearchCV
rfc_grid = RandomizedSearchCV(estimator= rfc,
                              param_distributions=param_grid,
                              cv=3,
                              verbose=2,
                              n_jobs = -1)
                              
# Fitting the grid parameter on Train data
rfc_grid.fit(X_train_scl,y_train_sm)
```
```
# checking for the Accuracy after implementing Hyperparameter Tuning
from sklearn import metrics
print("Accuracy %:",metrics.accuracy_score(y_test_sm, y_pred_rfc_grid)*100)
```
Accuracy %: 85.35674859101738

**b.Support vector classifier**
```
C=1
from sklearn.svm import SVC 
svc = SVC(C=C , cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovo', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=95, shrinking=True,
  tol=1, verbose=False)
  
# Fitting the Svm model on Training Data
svc.fit(X_train_svm[:68844], y_train_sm[:68844])
```
```
# Predicting the Test set results
y_pred_svm = svc.predict(X_test_svm[:68844])
```
```
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test_sm[:68844],y_pred_svm[:68844],labels = [1,0])
```
array([[    0, 34422],
       [    0, 34422]])
```
# checking for Accuracy of SVC model
from sklearn import metrics
print("Accuracy os SVC model % :",metrics.accuracy_score(y_test_sm[:68844],y_pred_svm[:68844])*100)
#print(f'Test Accuracy for svm: {svc.score(y_test_sm[:68844],y_pred_svm[:68844]) * 100:.2f}')
```
Accuracy os SVC model % : 50.0
-Hyperparameter Tunning Using GridSearch for Hyperparamter tuning for Support Vector Machine
```
# Defining GridSearchCV for Hyperparameter Tuning on SVC model
from sklearn.model_selection import GridSearchCV

# Defining grid parameters for Hyperparametric Tuning
parameters = [{'C':[1,10,100], 'kernel': ['linear'] ,'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]}]
grid = GridSearchCV(estimator = svc,
                    param_grid = parameters,
                    refit=True, 
                    scoring = 'accuracy',
                    n_jobs = -1,
                    verbose=2)
grid = grid.fit(X_train_svm[:5000], y_train_sm[:5000])

# checking at the Accuracy of Training data 
accuracy = grid.best_score_
accuracy
```
0.8114
- The best performing Classification Model at this stage of Analysis was Random Forest Classifier, with the follwing results on evaluating metrics


                precision   recall  f1-score   support

           0       0.80      0.93      0.86     34422
           1       0.91      0.77      0.83     34422

       accuracy                        0.85     68844
      macro avg    0.86      0.85      0.85     68844
      wght  avg    0.86      0.85      0.85     68844

 
- Accuracy %: 85.14467491720411

**Developers:** All team created their version individually, and we decided upon which steps to finalize.

### 4.5 Target variable creation for risk evaluation and assesment

- After a thorught reaserch to identify new Procedures to evaluate the 2 target assesment features agrred upon on the Planning stage,
We came up with theses Algorithms...

- Theses 2 Target variable Creation Steps were added at the preprocessing Stage of the Development Cycle along with new variable creation
  that is Prefered_ROI and Prepaymemnt_amt.
- Know we're ready to go a head with Regression Modeling.
                 

### 4.6 Regression Modeling
- In order to prepare the dataset for Regeression Modeling we had to revisit the dataset at the Stage of Features Engineering,
- There we changed 2 major steps of FE...

      1. Feature Selection
      2. Categorical Fetaure Encoding      
      
      
**1. Feature Selection**

- The highly correlated set of attributes needs to be dropped to avoid intercorrelation in the dataset has changed to be...
# calculating monthly EMI
```
# calculating monthly EMI
def emi(p, r, t): 
    # for one month interest
    r = r/(12*100)  
    emi = (p*r) * (1+r)**t/(((1+r)**t)-1)
    return (emi)
```
```
# Calculating monthly EMI
dfloan['Monthy_EMI'] = dfloan.apply(lambda row: emi(row['OrigUPB'],row['OrigInterestRate'],row['OrigLoanTerm']),axis=1)

# Total Accured amount(principal + Interest)
dfloan['Total_Loan_Amt'] = round(dfloan.Monthy_EMI * dfloan.OrigLoanTerm)

# Total interest payable
dfloan['Total_loan_Int'] = dfloan.Total_Loan_Amt - dfloan.OrigUPB
```
```
# calculating monthly income from Dti ratio
dfloan['monthly_income'] = round(dfloan.Monthy_EMI / (dfloan.DTI/100))

# calculating Annual income from Dti ratio
dfloan['Annual_income'] = round(dfloan.monthly_income * 12)
```

**2. Categorical Fetaure Encoding**

```
# Label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

```







**Developers:**  All team created their version individually, and we decided upon which steps to finalize.

**Regression Modelling**
- We have trained 2 different Models with their default values, and evaluating the model and from that select model with a best score.
- We have selected mean_square_error, mean_square_error_percentage, and r2_score from sklearn.metrics to evaluate selected models.
- The two choosen models (all linear) are...
```
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


```


**Linear Regression**
```
regressor= LinearRegression()  
regressor.fit(X_train, y_train) 

#Predicting the Test set result;  
y_pred= regressor.predict(X_test)  

```
```
# checking for the regressor scores
print('Train Score: ', regressor.score(X_train, y_train))  
print('Test Score: ', regressor.score(X_test, y_test))
```
Train Score:  0.9710887972221005
Test Score:  0.9706852681006621

```
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error , mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
```
Mean absolute error: 67.52
Mean squared error: 15613.78
Root mean squared error: 124.96

```
print('r2_score :' , r2_score(y_test, y_pred))
```
r2_score : 0.9706852681006621

**Lasso Regression**
```
lasso_reg = Lasso(alpha=1)
lasso_reg.fit(X_train,y_train)

#Predicting for X_test
y_pred_lass =lasso_reg.predict(X_test)
```
```
print('R squared training set', round(lasso_reg.score(X_train, y_train)*100, 2))
print('R squared test set', round(lasso_reg.score(X_test, y_test)*100, 2))
```
```
from sklearn.metrics import mean_squared_error
# Training data
pred_train = lasso_reg.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)
print('MSE training set', round(mse_train, 2))
```
```
# Test data
mse_test =mean_squared_error(y_test, y_pred_lass)
print('MSE test set', round(mse_test, 2))
```

MSE training set 15598.96
MSE test set 15634.61
```
lasso_reg.score(X_test, y_test)
```
0.9495804845324975

-Linear regression R-square score :97%
-Lasso Regression at Aplha=1 gave 94%

- The best performing Regreesion Model was Linear with default parameters by scikit-learn.

![pic_14](https://user-images.githubusercontent.com/93732090/209707032-79c80f0d-a195-4caa-9884-7065847401af.png)

**Developers:** All team created their version individually, and we decided upon which steps to finalize.



### 4.7 Pipelines Creation (Classification and Regression)

- Now Automating all the Important steps of Feature Engieering, Modeling into 2 Pipelines.
- There'll be 2 common steps of the Pipelines which are
           - Features Scaling.
           - PCA
- For classification Pipeline, we'll use RandomForestClassifier.
- For Regression Pipeline, we'll use Linear Regression and Multilasso
- The output is 2 pickle files for the 2 pipelines to be used in UI.


**Classification Pipeline**

```
## Creating pipelines for Random Forest Classifeir
from imblearn.ensemble import RandomForestClassifier
classifier_Pipe = Pipeline([
    ('scaler',scaler),
    ('randomforest',BalancedRandomForestClassifier(class_weight="balanced"))
     ])
classifier_Pipe.fit(X_train, y_class_train)
pred_class = classifier_Pipe.predict(X_test)
print('test accuracy = ', round(accuracy_score(y_class_test, pred_class)*100, 2), '%')
```
Test accuracy =  70.79 %

**Regression Pipeline**
```
# Create Regression Model
from sklearn.linear_model import LinearRegression
Regression_pipe = Pipeline([
     ('scaler',scaler),
     ('Regressor',LinearRegression()) 
    ])
    
#Model Evaluating
from sklearn.metrics import mean_squared_error
# fit and transform the pipeline
Regression_pipe.fit(X_train, y_reg_train)

# predict using the pipeline
pred_test_lass = Regression_pipe.predict(X_test)

#print('R squared training set', round(r2_score(y_reg_train,pred_train_lass)*100, 2))
print('R squared test set', round(r2_score(y_reg_test,pred_test_lass)*100, 2))
```
R squared test set 67.8
```
print(np.sqrt(mean_squared_error(y_reg_test,pred_test_lass))) 
print(r2_score(y_reg_test, pred_test_lass)*100,2)
```
12649.439209089423
67.80406052243274 2

- Dumping 2 pipeline files
```
with mgzip.open(r'C:\\Users\\lenovo\\FinalProj\\classifier_Pipe', 'wb') as f:\n",
    pickle.dump(classifier_Pipe, f)"
#pickle.dump(classifier_Pipe, open('classifier_Pipe.pkl','wb'))
pickle.dump(Regression_pipe, open('Regression_pipe.pkl','wb'))
```               

### 4.8. UI: App Creation
- Using Flask API, and simple HTML5, and CSS; we created a web application with v01 as specified during the planning Step of the analysis.
- v02 is Deployment on Google Cloud
- During v01, developed files contained:
      1. main.py --- for Flask to run the app and steer it's way around different files.
      
      2. pipelines.py --- fpr preprocessing of input data from the Client and make it match the attributes expected by the Pipelines file, furthermore run the pipelines files.
      
      3. index.html --- v01 fourm (individual borrower entry using a fourm)
      
      4. finalpredict.html --- v01 output fourm of expected assesment criteria's
      

                


## 5. Deployment
- After agrreing on Google Cloud as the deployment platform for this app, we deployed the app for production.
- [App link](http://13.126.68.187:8080/)

**Developers:** [Nour M. Ibrahim](https://github.com/Nour-Ibrahim-1290), [Sumyukta k](https://github.com/manasa0551)

## 6. Conclusion:
- We've fullfilled all assemnet reauiremnets of the Client.
- We've developed 1st vesion of UI requiremnet (v01) which is Single borrower data entry uding a fourm.
- (v02) Multiple borrower data entry via CSV file upload is under development...
