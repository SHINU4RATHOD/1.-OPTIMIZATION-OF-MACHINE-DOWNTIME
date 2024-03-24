import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import sklearn.metrics as skmet
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pickle
import joblib

# MySQL Database connection
from sqlalchemy import create_engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = 'root', pw = '1122', db='project'))
# data.to_sql('sms_raw', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
sql = 'select * from machine_downtime;'
df = pd.read_sql_query(sql, engine)

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


######## splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(clean_data, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors = 7)
KNN = knn.fit(X_train, y_train)

# Predictions on training and test sets
y_train_pred = KNN.predict(X_train)
y_test_pred = KNN.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
 
# Classification report for test data
print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix for Test Data:")
print(confusion_matrix(y_test, y_test_pred))




acc = []
# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values
for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, y_train)
    train_acc = np.mean(neigh.predict(X_train) == y_train)
    test_acc = np.mean(neigh.predict(X_test) == y_test)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])
acc

# Plotting the data accuracies
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "ro-")
plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], "bo-")

 

############### Hyperparameter optimization
from sklearn.model_selection import GridSearchCV
k_range = list(range(3, 50, 2))
param_grid = dict(n_neighbors = k_range)

grid = GridSearchCV(knn, param_grid, cv = 5, 
                    scoring = 'accuracy', 
                    return_train_score = False, verbose = 1)
 
KNN_new = grid.fit(X_train, y_train) 
print(KNN_new.best_params_)

accuracy = KNN_new.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )

# Predict the class on test data
pred = KNN_new.predict(X_test)
cm = skmet.confusion_matrix(y_test, pred)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Benign', 'Malignant'])
cmplot.plot()
cmplot.ax_.set(title = 'machin failure Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

# Save the model
knn_best = KNN_new.best_estimator_
pickle.dump(knn_best, open('knn.pkl', 'wb'))


 
