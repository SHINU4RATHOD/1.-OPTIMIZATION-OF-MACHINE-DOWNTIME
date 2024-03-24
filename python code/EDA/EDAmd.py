import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv(r'Machine Downtime.csv') 

from sqlalchemy import create_engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = 'root', pw = '1122', db='project'))
df.to_sql('dummymd', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
sql = 'select * from md;'
df = pd.read_sql_query(sql, engine)

df.info()
desc = df.describe()

cn = df.columns
nf = df.select_dtypes(exclude = 'object').columns
cf = df.select_dtypes(include = 'object').columns

# ***************************auto EDA
# 1) SweetViz
import sweetviz as sv
s = sv.analyze(df)
s.show_html()


# *******************EDA
mean = df[nf].mean()  
median = df[nf].median()
mode = df[cf].mode()

# Measures of Dispersion / Second moment business decision
variance = df[nf].var()  
std = df[nf].std()  
range_val = df[nf].max() - df[nf].min()

# Third moment business decision
sk = df[nf].skew()
# Fourth moment business decision
kurt = df[nf].kurt()

# ************************* data preprocessing
# 1. typecassting
df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors='coerce')
cf = df.select_dtypes(include = 'object').columns
df.info()


# 2. handling the duplicates
df.duplicated().sum()
df.drop_duplicates()     # drop if duplicates are there


#3. handling the missing values
df.isnull().sum()

imputer = SimpleImputer(strategy='mean')   # 'mean', 'median', 'most_frequent', 'constant'                          
df[nf] = imputer.fit_transform(df[nf])      # kNN Imputation, iterativeImputation



# 4. outlier analysis
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (20, 6)) 
plt.subplots_adjust(wspace = 0.75)  
plt.show()


winsor = Winsorizer(capping_method = 'iqr',  
                          tail = 'both', 
                          fold = 1.5,
                          variables = list(df[nf]))
winz_data = winsor.fit_transform(df[nf])

winz_data.plot(kind = 'box', subplots = True, sharey = False, figsize = (20, 6)) 
plt.subplots_adjust(wspace = 0.75)  
plt.show()


# imputed and winsorized data
clean = pd.concat([df['Date'], df[cf],winz_data], axis=1)
clean.to_sql('cleanmdpy', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
clean.to_csv('cleanmdpy.csv', index = False)

# 5.dummy variable creation
# dummy = pd.get_dummies(df, drop_first=True)
 
dummy_df = pd.get_dummies(df[cf], drop_first=True)
winzonehotencoded = pd.concat([winz_data, dummy_df], axis=1)
df.drop(columns=cf, inplace=True)



# 6. tranformtion 
from scipy.stats import shapiro
stat, p = shapiro([df["Hydraulic_Pressure"]])

# Set the significance level (usually 0.05)
alpha = 0.05
# Check the p-value against alpha
if p > alpha:
    print("Data looks normally distributed (fail to reject H0)")
else:
    print("Data does not look normally distributed (reject H0)")



import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

# Plot histogram with density plot (Kernel Density Estimation - KDE)
plt.figure(figsize=(8, 6))
sns.histplot(df['Hydraulic_Pressure'], kde=True, color='skyblue')
plt.title('Histogram with Density Plot')
plt.xlabel('Values')
plt.ylabel('Frequency/Density')
plt.show()

# Plot a Q-Q plot
plt.subplot(1, 2, 2)
stats.probplot(df["Hydraulic_Pressure"], dist="norm", plot=plt)
plt.title('Q-Q Plot')

plt.tight_layout()
plt.show()  



