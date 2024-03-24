''' ************************ Project Title :- Optimization of Machine Downtime   ********
- "Client: One of the leading vehicle fuel pump manufacturers. These pumps are used to take fuel 
as input and push fuel as output at a high velocity. More the velocity, more is the speed at which 
vehicle will move.


CRISP-ML(Q):
phase-I: 1 : Business Understanding: 

    Business Problem: Machines which manufacture the pumps. Unplanned machine downtime which is leading to loss of productivity.
    Business Objective: Minimize unplanned machine downtime.
    Business Constraint: Minimize maintenance cost.

    Success Criteria:
        Business: Reduce the unplanned downtime by at least 10%
        Machine Learning: Achieve an accuracy of atleast 90%
        Economic: Achieve the accuracy of at least 90%

pahse-I: 2: Data collection and Understanding: 
Dimensions: 2500 rows * 16 cols
Date : date features Indicates the date of observation
Machine_ID : Unique identifier for each machine involved in the process
Assembly_Line_No : Identifier for the assembly line or specific section where the machine operates.
Hydraulic_Pressure(bar) : ): This column likely represents the hydraulic pressure measured in bars (unit of pressure) within a hydraulic system.
Coolant_Pressure(bar) : Indicates the pressure within the coolant system measured in bars, often used in cooling systems to maintain optimal temperatures.
Air_System_Pressure(bar) :	Represents the pressure within the air system measured in bars, commonly used in pneumatic systems.
Coolant_Temperature : It refers to the temperature of the coolant, possibly in degrees Celsius or Fahrenheit, used for heat dissipation in machinery.
Hydraulic_Oil_Temperature(Â°C) : Indicates the temperature of the hydraulic oil, usually measured in degrees Celsius, important for hydraulic system performance.
Spindle_Bearing_Temperature(Â°C) : Refers to the temperature of the spindle bearings, typically measured in degrees Celsius, which could be crucial for machinery performance.
Spindle_Vibration(Âµm) : Represents the vibration of the spindle, possibly measured in micrometers (µm), an indicator of the machinery's stability.
Tool_Vibration(Âµm) : Indicates the vibration of the tool, likely measured in micrometers (µm), crucial for tool performance and accuracy.
Spindle_Speed(RPM) : : Denotes the rotational speed of the spindle, typically measured in revolutions per minute (RPM), essential for machining processes.	
Voltage(volts) : Represents the voltage input, usually measured in volts, which is be an input parameter for the machinery
Torque(Nm) : Refers to the torque applied, measured in Newton-meters (Nm), which is pivotal in determining machinery performance and load capacity.
Cutting(kN) : Indicates the cutting force, measured in kilonewtons (kN), important in machining operations.
Downtime : Machine_Failure: This category indicates that a machine has experienced some failure or downtime.

'''
# importing all required lib and dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Hyperparameter optimization
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


from sqlalchemy import create_engine
import joblib
import pickle
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = 'root', pw = '1122', db='project'))
# data.to_sql('sms_raw', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
sql = 'select * from machine_downtime;'
df = pd.read_sql_query(sql, engine)

df = df.drop(columns = ['Date','Machine_ID', 'Assembly_Line_No'])

# extracting dependent and independent variable
x = df.drop('Downtime', axis=1)   
y = df['Downtime']

# Extracing numerical and non numerical features 
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


# Defining a function to count outliers
def count_outliers(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
    return outliers
# Count outliers before Winsorization
outliers_before = imputed_df.apply(count_outliers)
outliers_before
imputed_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
plt.subplots_adjust(wspace = 0.75) 
plt.show()
############################## Define Winsorization pipeline
winsorizer_pipeline = Winsorizer(capping_method='iqr', tail='both', fold=1.5)
X_winsorized = winsorizer_pipeline.fit_transform(imputed_df)
joblib.dump(winsorizer_pipeline, 'winsor.pkl')  

# Transform Winsorized data back to DataFrame
X_winsorized_df = pd.DataFrame(X_winsorized, columns=nf)

# Count outliers after Winsorization
outliers_after = X_winsorized_df.apply(count_outliers)
outliers_after

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
X_encoded =  prep_encoding_pipeline.fit(x)    
joblib.dump(X_encoded, 'encoding')

encode_data = pd.DataFrame(X_encoded.transform(x))
# To get feature names for Categorical columns after Onehotencoding 
encode_data.columns = X_encoded.get_feature_names_out(input_features = x.columns)
encode_data.info()

clean_data = pd.concat([X_scaled_df, encode_data], axis = 1)  
clean_data.info()

######## splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(clean_data, y, test_size=0.2, random_state=42)

##################################### Bagging Classifier Model
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
clftree = tree.DecisionTreeClassifier()
bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = -1, random_state = 42)
bagging = bag_clf.fit(X_train, y_train)

# Predictions on training and test sets
y_train_pred = bagging.predict(X_train)
y_test_pred = bagging.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(confusion_matrix(y_train,y_train_pred))
print(confusion_matrix(y_test, y_test_pred))

print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred))
# Saving the best model
pickle.dump(bagging, open('baggingmodel.pkl', 'wb'))


###### Cross Validation implementation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv = 5):

    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator = model,
                           X = _X,
                           y = _y,
                           cv = _cv,
                           scoring = _scoring,
                           return_train_score = True)

    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          })

Bagging_cv_scores = cross_validation(bag_clf, X_train, y_train, 5)
Bagging_cv_scores

def plot_result(x_label, y_label, plot_title, train_data, val_data):
        # Set size of plot
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        #ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color = 'blue', label = 'Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color = 'red', label = 'Validation')
        plt.title(plot_title, fontsize = 30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize = 14)
        plt.ylabel(y_label, fontsize = 14)
        plt.legend()
        plt.grid(True)
        plt.show()


model_name = "Bagging Classifier"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            Bagging_cv_scores["Training Accuracy scores"],
            Bagging_cv_scores["Validation Accuracy scores"])





################################ Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rf_Model = RandomForestClassifier()
# Hyperparameters
n_estimators = [int(x) for x in np.linspace(start=10, stop=80, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [2, 4]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
bootstrap = [True, False]

# Create the parameter grid with the correct parameter name 'n_estimators'
param_grid = {
    'n_estimators': n_estimators,  
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}
 
##################  Hyperparameter optimization with GridSearchCV
rf_Grid = GridSearchCV(estimator=rf_Model, param_grid=param_grid, cv=10, verbose=2, n_jobs=-1)
rf_Grid.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = rf_Grid.best_params_
cv_rf_grid = rf_Grid.best_estimator_

# Predictions on training and test sets
y_train_pred = bagging.predict(X_train)
y_test_pred = bagging.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(confusion_matrix(y_train,y_train_pred))
print(confusion_matrix(y_test, y_test_pred))

print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred))

################# Hyperparameter optimization with RandomizedSearchCV
rf_Random = RandomizedSearchCV(estimator = rf_Model, 
                               param_distributions = param_grid, cv = 10, verbose = 0, n_jobs = -1)

rf_Random.fit(X_train, y_train)
rf_Random.best_params_
cv_rf_random = rf_Random.best_estimator_

# Predictions on training and test sets
y_train_pred = bagging.predict(X_train)
y_test_pred = bagging.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(confusion_matrix(y_train,y_train_pred))
print(confusion_matrix(y_test, y_test_pred))

print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred))


########### Cross Validation implementation
def cross_validation(model, _X, _y, _cv = 5):

    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                           X = _X,
                           y = _y,
                           cv = _cv,
                           scoring = _scoring,
                           return_train_score = True)

    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          })
Random_forest_result = cross_validation(cv_rf_random, X_train, y_train, 5)
Random_forest_result


def plot_result(x_label, y_label, plot_title, train_data, val_data):
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        plt.ylim(0.40000, 1)
        plt.bar(X_axis - 0.2, train_data, 0.1, color = 'blue', label = 'Training')
        plt.bar(X_axis + 0.2, val_data, 0.1, color = 'red', label = 'Validation')
        plt.title(plot_title, fontsize = 30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize = 14)
        plt.ylabel(y_label, fontsize = 14)
        plt.legend()
        plt.grid(True)
        plt.show()

model_name = "RandomForestClassifier"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            Random_forest_result["Training Accuracy scores"],
            Random_forest_result["Validation Accuracy scores"])


#######################  AdaBoosting
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)
ada_clf1 = ada_clf.fit(X_train, y_train)

# prediction on the training and test dataset
y_train_pred = ada_clf1.predict(X_train)
y_test_pred = ada_clf1.predict(X_test)

# Evaluation on Testing Data
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix for Test Data:")
print(confusion_matrix(y_test, y_test_pred))

# Saving the best model
pickle.dump(ada_clf1, open('adaboost.pkl','wb'))



######################  Gradient Boosting   
from sklearn.ensemble import GradientBoostingClassifier
boost_clf = GradientBoostingClassifier()
boost_clf1 = boost_clf.fit(X_train, y_train)

y_train_pred = boost_clf1.predict(X_train)
y_test_pred = boost_clf1.predict(X_test)

print(accuracy_score(y_train, y_train_pred))
print(accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))

# Classification report for test data
print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix for Test Data:")
print(confusion_matrix(y_test, y_test_pred))

# Save the ML model
pickle.dump(boost_clf1, open('gradiantboost.pkl','wb'))
# load the model
grad_model = pickle.load(open('gradiantboost.pkl','rb'))


############ Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)
boost_clf_p = boost_clf2.fit(X_train, y_train)

grad_pred_p = boost_clf_p.predict(X_test)

# Evaluation on Testing Data
print(confusion_matrix(y_test, grad_pred_p))
print(accuracy_score(y_test,grad_pred_p))

# Evaluation on Training Data
print(confusion_matrix(y_train, boost_clf_p.predict(X_train)))
accuracy_score(y_train, boost_clf_p.predict(X_train))
# Save the ML model
pickle.dump(boost_clf_p, open('gradiantboostparam.pkl', 'wb'))
grad_model_p = pickle.load(open('gradiantboostparam.pkl', 'rb'))



##########################  XGBoosting
df['Downtime'] = np.where(df.Downtime == 'Machine_Failure', 1, 0)
x = df.drop('Downtime', axis = 1)
y = df['Downtime']
X_train, X_test, y_train, y_test = train_test_split(clean_data, y, test_size=0.2, random_state=42)
import xgboost as xgb
xgb_clf = xgb.XGBClassifier(max_depth = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)
xgb_clf1 = xgb_clf.fit(X_train, y_train)

# Predictions on training and test sets
y_train_pred = xgb_clf1.predict(X_train)
y_test_pred = xgb_clf1.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
 
# Classification report for test data
print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix for Test Data:")
print(confusion_matrix(y_test, y_test_pred))
xgb.plot_importance(xgb_clf)


fi = pd.DataFrame(xgb_clf1.feature_importances_.reshape(1, -1), columns = X_train.columns)
fi

# Save the ML model
pickle.dump(xgb_clf1, open('xgb.pkl', 'wb'))
xgb_model = pickle.load(open('xgb.pkl', 'rb'))


###################### Hyperparameter tuning RandomizedSearchCV for XGB
xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)
param_test1 = {'max_depth': range(3, 10, 2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0.9],
               'rag_alpha': [1e-2, 0.1, 1]}


xgb_RandomGrid = RandomizedSearchCV(estimator = xgb_clf, param_distributions = param_test1, 
                                    cv = 5, verbose = 2, n_jobs = -1)


Randomized_search1 = xgb_RandomGrid.fit(X_train, y_train)
cv_xg_clf = Randomized_search1.best_estimator_
cv_xg_clf

randomized_pred = cv_xg_clf.predict(X_test)

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(y_test, randomized_pred)
Randomized_search1.best_params_

randomized_pred_1 = cv_xg_clf.predict(X_train)

# Evaluation on Training Data with model with hyperparameters
accuracy_score(y_train, randomized_pred_1)
pickle.dump(cv_xg_clf, open('Randomizedsearch_xgb.pkl', 'wb'))
randomized_model = pickle.load(open('Randomizedsearch_xgb.pkl', 'rb'))




####################### model testing on new data
model1 = pickle.load(open('gradiantboost.pkl', 'rb'))
impute = joblib.load('meanimpute')
winzor = joblib.load('winsor')
minmax = joblib.load('minmax')
encode = joblib.load('encoding')
data = pd.read_csv('C:/Users/SHINU RATHOD/Desktop/Optimization of Machin Downtime_148/07_Data Set/code/python code/04_Ensemble model/Bagging, RF, Boosting/md_new.csv')

nf = data.select_dtypes(exclude='object').columns
cf = data.select_dtypes(include = 'object').columns

x_impute = pd.DataFrame(impute.transform(data[nf]), columns = data.columns)
x_winz = pd.DataFrame(winzor.transform(x_impute), columns = data.columns)
x_scale = pd.DataFrame(minmax.transform(x_winz), columns = data.columns)

x_encode = pd.DataFrame(encode.transform(data), columns = encode.get_feature_names_out())
clean = pd.concat([x_scale, x_encode], axis=1)
prediction = pd.DataFrame(model1.predict(clean), columns = ['machin_failure_pred'])
prediction
final = pd.concat([prediction, data], axis = 1)
