import pandas as pd
import numpy  as np 
import matplotlib.pyplot as plt
 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, PolynomialFeatures


from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve


from sqlalchemy import create_engine
import joblib
import pickle
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = 'root', pw = '1122', db='project'))
#df.to_sql('Machine_Downtime', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
sql = 'select * from Machine_Downtime;'
df = pd.read_sql_query(sql, engine)
 
df = df.drop(columns = ['Date','Machine_ID', 'Assembly_Line_No'])
df['Downtime'] = np.where(df.Downtime == 'Machine_Failure', 1, 0)
 
 
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

outlier_counts = {}
# Loop through each column in the DataFrame
for column in imputed_df.columns:
    # Find the interquartile range (IQR) for the column
    Q1 = imputed_df[column].quantile(0.25)
    Q3 = imputed_df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Determine the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count the number of outliers for the column
    num_outliers = ((imputed_df[column] < lower_bound) | (imputed_df[column] > upper_bound)).sum()
    
    # Store the outlier count for the column in the dictionary
    outlier_counts[column] = num_outliers
# Print the outlier counts for each feature
print("Outlier Counts:")
for column, count in outlier_counts.items():
    print(f"{column}: {count}")


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
encoding_pipeline = Pipeline([('onehot', OneHotEncoder(drop='first'))])
X_encoded =  encoding_pipeline.fit(x[cf])   
joblib.dump(X_encoded, 'encoding')
# Convert the encoded data to a DataFrame
X_encode_df = pd.DataFrame(X_encoded.transform(x[cf]), columns=encoding_pipeline.named_steps['onehot'].get_feature_names_out(cf))
 

clean_data = pd.concat([X_scaled_df, X_encode_df], axis = 1) 
clean_data.info()

######## splitting the dataset 
x_train, x_test, y_train, y_test = train_test_split(clean_data, y, test_size=0.2, random_state=42)
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)

# prediction on training and test dataset
y_train_pred = linear_reg.predict(x_train)
y_test_pred = linear_reg.predict(x_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
rmse      # 0.35295492913007753
# Calculate R2 score
r2 = r2_score(y_test, y_test_pred)
r2        #0.4987963195528651



 
# Saving the best model
pickle.dump(linear_reg, open('baggingmodel.pkl', 'wb'))











