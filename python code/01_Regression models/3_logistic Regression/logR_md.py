import pandas as pd
import numpy  as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV


from sqlalchemy import create_engine
import joblib
import pickle
df = pd.read_csv(r'Machine Downtime.csv') 

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = 'root', pw = '1122', db='project'))
df.to_sql('md', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
sql = 'select * from md;'
df = pd.read_sql_query(sql, engine)


df['Downtime'] = df['Downtime'].replace({'Machine_Failure': 1, 'No_Machine_Failure': 0})

############# handling the date column
# 1. remove column
df = df.drop(columns = 'Date', axis = 1)

# 2. typecast
# =============================================================================
# df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors='coerce')   # bcz date column contain the values with deffrent date formate 
# cf = df.select_dtypes(include = 'object').columns
# =============================================================================
 
df.info()
df.describe()



# Multivariate Analysis
sns.pairplot(df)   # original data
# Correlation Analysis on Original Data
df = df.drop(columns = ['Machine_ID', 'Assembly_Line_No'])
orig_df_cor = df.corr()
orig_df_cor


df.isnull().sum()
########## extracting the  independent and dependent data
x = df.drop('Downtime', axis=1)   
y = df['Downtime']

nf = x.select_dtypes(exclude = 'object').columns
cf = x.select_dtypes(include = 'object').columns


x.isnull().sum()
########################## creating the pipline for simpleImputer
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, nf)])

imputation = preprocessor.fit(x)
joblib.dump(imputation, 'meanimpute')

imputed_df = pd.DataFrame(imputation.transform(x), columns = nf)
imputed_df.isnull().sum()




imputed_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
plt.subplots_adjust(wspace = 0.75) 
plt.show()
################### Create a preprocessing pipeline for Winsorization
winsorizer_pipeline = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5)
X_winsorized = winsorizer_pipeline.fit(imputed_df)
joblib.dump(X_winsorized, 'winsor')

X_winsorized_df = pd.DataFrame(X_winsorized.transform(imputed_df), columns = nf)

X_winsorized_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
plt.subplots_adjust(wspace = 0.75)  
plt.show()



############################ creating pipline for MinmaxScaler
scale_pipeline = Pipeline([('scale', MinMaxScaler())])
X_scaled = scale_pipeline.fit(X_winsorized_df)
joblib.dump(X_scaled, 'minmax')

X_scaled_df = pd.DataFrame(X_scaled.transform(X_winsorized_df), columns = nf)
 


############################ creating pipline for OneHotEncoder
encoding_pipeline = Pipeline([('onehot', OneHotEncoder(sparse=False, drop='first'))])
X_encoded =  encoding_pipeline.fit(x[cf])   
joblib.dump(X_encoded, 'encoding')

X_encode_df = pd.DataFrame(X_encoded.transform(x[cf]), columns=encoding_pipeline.named_steps['onehot'].get_feature_names_out(cf))
 


clean_data = pd.concat([X_scaled_df, X_encode_df], axis = 1) 
clean_data.info()



#################### model building Statsmodel
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
logit_model = sm.Logit(y, clean_data).fit()
pickle.dump(logit_model, open('logistic.pkl', 'wb'))
# Summary
logit_model.summary()
logit_model.summary2() # for AIC (It is a statistical measure used in model selection and model comparison)

# Prediction
pred = logit_model.predict(clean_data)
 
# ROC Curve to identify the appropriate cutoff value
fpr, tpr, thresholds = roc_curve(y, pred)
optimal_idx = np.argmax(tpr - fpr)  # tpr and fpr diff should be max
optimal_threshold = thresholds[optimal_idx]
optimal_threshold


auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : %f" % auc)
# Filling all the cells with zeroes
clean_data["pred"] = np.zeros(2500)
# taking threshold value and above the prob value will be treated as correct value 
clean_data.loc[pred > optimal_threshold, "pred"] = 1


# Confusion Matrix
confusion_matrix(clean_data.pred, y)

# Accuracy score of the model
print('Test accuracy = ', accuracy_score(clean_data.pred, y))

# Classification report
classification = classification_report(clean_data["pred"], y)
print(classification)

### PLOT FOR ROC
plt.plot(fpr, tpr, label = "AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 4)
plt.show()




################################################################
# Model evaluation - Data Split
x_train, x_test, y_train, y_test = train_test_split (clean_data.iloc[:, :12], y, test_size = 0.2, random_state = 0,stratify = y)

# Fitting Logistic Regression to the training set  
logisticmodel = sm.Logit(y_train, x_train).fit()
# Evaluate on train data
y_pred_train = logisticmodel.predict(x_train)  
y_pred_train

# Metrics
# Filling all the cells with zeroes
y_train["pred"] = np.zeros(2000)

# taking threshold value and above the prob value will be treated as correct value 
y_train.loc[pred > optimal_threshold, "pred"] = 1

auc = metrics.roc_auc_score(y_train["ATTORNEY"], y_pred_train)
print("Area under the ROC curve : %f" % auc)

classification_train = classification_report(y_train["pred"], y_train["ATTORNEY"])
print(classification_train)

# confusion matrix 
confusion_matrix(y_train["pred"], y_train["ATTORNEY"])

# Accuracy score of the model
print('Train accuracy = ', accuracy_score(y_train["pred"], y_train["ATTORNEY"]))


# Validate on Test data
y_pred_test = logisticmodell.predict(x_test)  
y_pred_test

# Filling all the cells with zeroes
y_test["y_pred_test"] = np.zeros(268)

# Capturing the prediction binary values
y_test.loc[y_pred_test > optimal_threshold, "y_pred_test"] = 1

# classification report
classification1 = classification_report(y_test["y_pred_test"], y_test["ATTORNEY"])
print(classification1)

# confusion matrix 
confusion_matrix(y_test["y_pred_test"], y_test["ATTORNEY"])

# Accuracy score of the model
print('Test accuracy = ', accuracy_score(y_test["y_pred_test"], y_test["ATTORNEY"]))




















