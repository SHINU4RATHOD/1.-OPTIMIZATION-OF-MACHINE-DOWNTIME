import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
from sqlalchemy import create_engine
import joblib
import pickle

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = 'root', pw = '1122', db='project'))
# data.to_sql('sms_raw', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
sql = 'select * from machine_downtime;'
df = pd.read_sql_query(sql, engine)
x = df.info()
df_desc = df.describe()
df = df.drop(columns = ['Date','Machine_ID', 'Assembly_Line_No'])


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




imputed_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (30, 15)) 
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
encoding_pipeline = Pipeline([('onehot', OneHotEncoder())])
prep_encoding_pipeline = ColumnTransformer([('categorical', encoding_pipeline, cf)])
X_encoded =  prep_encoding_pipeline.fit(x)   # Works with categorical features only
joblib.dump(X_encoded, 'encoding')

encode_data = pd.DataFrame(X_encoded.transform(x))
# To get feature names for Categorical columns after Onehotencoding 
encode_data.columns = X_encoded.get_feature_names_out(input_features = x.columns)
encode_data.info()

clean_data = pd.concat([X_scaled_df, encode_data], axis = 1)  
clean_data.info()
 


######## splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(clean_data, y, test_size=0.2, random_state=42)
model = DT(criterion = 'entropy')
model.fit(X_train, y_train)

# Predictions on training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
 
# Classification report for test data
print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix for Test Data:")
print(confusion_matrix(y_test, y_test_pred))


################## Hyperparameter Optimization
param_grid = { 'criterion':['gini', 'entropy'], 'max_depth': np.arange(3, 15)}
dtree_model = DT()
dtree_gscv = GridSearchCV(dtree_model, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1)
dtree_gscv.fit(X_train, y_train)
# The best set of parameter values
dtree_gscv.best_params_

# Model with best parameter values
DT_best = dtree_gscv.best_estimator_
DT_best

# Predictions on training and test sets
y_train_pred = DT_best.predict(X_train)
y_test_pred = DT_best.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
 
# Classification report for test data
print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix for Test Data:")
print(confusion_matrix(y_test, y_test_pred))
# save model
pickle.dump(DT_best, open('dt.pkl','wb'))

