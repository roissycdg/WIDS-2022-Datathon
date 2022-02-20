#team_model-behavior



# Load all the necessary libraries
from turtle import shape
import numpy as np  # numerical computation with arrays
import pandas as pd # library to manipulate datasets using dataframes
import scipy as sp  # statistical library
import sklearn
import random,os
import math
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import decomposition
from sklearn.impute import KNNImputer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import QuantileTransformer
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.ensemble
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

# load data
train=pd.read_csv('/users/jennifershelton/Desktop/wids2022/train.csv')
train.set_index('id')
test=pd.read_csv('/users/jennifershelton/Desktop/wids2022/test.csv')
sample = pd.read_csv('/users/jennifershelton/Desktop/wids2022/sample_solution.csv')

#set ID and target values
ID = "id"
TARGET = "site_eui"

# seed everything
SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

print("Data loaded", flush=True)


# Print a summary statistics for each column in the dataset
#print(train.describe())

#print("... Drop columns with too many NAs ...", flush=True)
        # Drop columns with > 50% missing data
#n_cols_before_drop = len(train.columns.values)
#train = train[train.columns[train.isna().mean() <= 0.50]]
#n_cols_after_drop = len(train.columns.values)
#print(n_cols_after_drop)


# Explore correlations between numerical features and the target variable
# Select only numerical (float) features
#num_features = train.select_dtypes(include=['float64']).columns.values
# Compute correlations between all numerical features
#df_corr = train[num_features].corr()
#print(df_corr.sort_values(by='site_eui', key=abs, ascending=False)['site_eui'])

# Create any new features - e.g. the difference in average temperature in January (typically coldest) vs. July (typically hottest).
train['highest_temp_diff'] = train['july_max_temp'] - train['january_min_temp']
train['extremedays'] = train['days_below_30F'] + train['days_below_20F']+ train['days_below_10F'] + train['days_below_0F'] + train['days_above_80F'] + train['days_above_90F']
train['av_temp_diff'] = train['july_avg_temp'] - train['january_avg_temp']
test['highest_temp_diff'] = test['july_max_temp'] - test['january_min_temp']
test['extremedays'] = test['days_below_30F'] + test['days_below_20F']+ test['days_below_10F'] + test['days_below_0F'] + test['days_above_80F'] + test['days_above_90F']
test['av_temp_diff'] = test['july_avg_temp'] - test['january_avg_temp']



#Elevation is very lowly correlated so we will drop it
train = train.drop(['ELEVATION'], axis=1)
test = test.drop(['ELEVATION'], axis=1)
#State factor increases RMSE score so we remove it
train = train.drop(['State_Factor'], axis=1)
test = test.drop(['State_Factor'], axis=1)
#These two columns are very sparse so we will drop
train = train.drop(['days_above_110F'], axis=1)
train = train.drop(['days_above_100F'], axis=1)
test = test.drop(['days_above_110F'], axis=1)
test = test.drop(['days_above_100F'], axis=1)

#bin the extreme weather days 
groups = ['Low', 'Med', 'High']
train['ext_bin'] = pd.qcut(train['extremedays'], q=3, labels=groups)
test['ext_bin'] = pd.qcut(test['extremedays'], q=3, labels=groups)

print("Encoding categorical data...", flush=True)
#one-hot encoding
train = pd.get_dummies(train)
test = pd.get_dummies(test)


print("impute missing train data", flush=True)
#define a simple imputer, which replaces missing values using the median, using SimpleImputer class
#imr = SimpleImputer(strategy = "median")
# Step 2: impute the missing data
#train = pd.DataFrame(imr.fit_transform(train), columns=train.columns, index=train.index)
#test = pd.DataFrame(imr.fit_transform(test), columns=test.columns, index=test.index)

#trying out alternate imputation methods
imp = IterativeImputer(max_iter=6, random_state=0)
train = pd.DataFrame(imp.fit_transform(train), columns=train.columns, index=train.index)
print("impute missing test data", flush=True)
test = pd.DataFrame(imp.fit_transform(test), columns=test.columns, index=test.index)

#imputer = KNNImputer(n_neighbors=2, weights="uniform")
#train = pd.DataFrame(imputer.fit_transform(train), columns=train.columns, index=train.index)
#test = pd.DataFrame(imputer.fit_transform(test), columns=test.columns, index=test.index)

print("scaling with quantile transformer...", flush=True)
qt = QuantileTransformer(n_quantiles=10, random_state=0)
qt.fit_transform(train)
qt.fit_transform(test)

#print("scaling with Min max scaler")
#scaler = MinMaxScaler()
#values = train.values
#scaler.fit(values)
#values_scaled = scaler.transform(values)
#train = pd.DataFrame(values_scaled, columns=train.columns, index=train.index)

print("Train test split...", flush=True)
# Split training dataframe into training and test set for model fitting
predictors = [feature_name for feature_name in train.columns.values if feature_name != 'site_eui']
X = train[predictors]
y = train['site_eui']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


print("Regression...")

#regr = MLPRegressor(random_state=1, max_iter=1000).fit(X_train, y_train)
#y_pred = regr.predict(X_test)
#print(regr.score(X_test, y_test))
# reg score = .35114719702097674
#after binning and new features, 0.367029
# RMSE = 66.99777879225572 with one-hot encoding and quantile transformer


#regr_1 = DecisionTreeRegressor(max_depth=2)
#regr_2 = DecisionTreeRegressor(max_depth=5)
#regr_1.fit(X, y)
#regr_2.fit(X, y)
# Predict
#y_1 = regr_1.predict(X_test)
#y_2 = regr_2.predict(X_test)
#mse1 = sklearn.metrics.mean_squared_error(y_test, y_1)
#rmse1 = math.sqrt(mse1)
#print(rmse1)
#RMSE = 55.38602469416471
#mse2 = sklearn.metrics.mean_squared_error(y_test, y_2)
#rmse2 = math.sqrt(mse2)
#print(rmse2)
# RMSE = 49.83392907972399
#ye = regr_2.predict(test)
# the decision tree regressor solo with max depth of 5 scored 58 in kaggle


#adding a gaussian kernel regressor 
kernel = DotProduct() + WhiteKernel()
#Voting Regressor
reg1 = GaussianProcessRegressor(kernel=kernel, random_state=0)
reg2 = RandomForestRegressor(random_state=1)
reg3 = AdaBoostRegressor(random_state=0, n_estimators=100)
ereg = VotingRegressor(estimators=[('gauss', reg1), ('rf', reg2), ('ada', reg3)])
ereg = ereg.fit(X_train, y_train)
e_y = ereg.predict(X_test)
emse = sklearn.metrics.mean_squared_error(y_test, e_y)
ermse = math.sqrt(emse)
print(ermse)
ye = ereg.predict(test)

print("writing submission file....")
SUBMISSION_PATH = "submission.csv"
sample[TARGET] = ye
sample.to_csv(SUBMISSION_PATH,index=False)
#print(sample.head())