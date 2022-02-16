#team_model-behavior



# Load all the necessary libraries
import numpy as np  # numerical computation with arrays
import pandas as pd # library to manipulate datasets using dataframes
import scipy as sp  # statistical library
import sklearn

# Load sklearn libraries for machine learning
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import decomposition

# Load plotting libraries
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.ensemble
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



train=pd.read_csv('/users/jennifershelton/Desktop/wids2022/train.csv')
train.set_index('id')
test=pd.read_csv('/users/jennifershelton/Desktop/wids2022/test.csv')


print("Data loaded", flush=True)

print(train.shape)
# Print a summary statistics for each column in the dataset
#print(train.describe())

print("... Drop columns with too many NAs ...", flush=True)
        # Drop columns with > 50% missing data
n_cols_before_drop = len(train.columns.values)
train = train[train.columns[train.isna().mean() <= 0.50]]
n_cols_after_drop = len(train.columns.values)
print(n_cols_after_drop)


#sns.jointplot(data=train, x="floor_area", y="site_eui")

# Explore correlations between numerical features and the target variable
# Select only numerical (float) features
num_features = train.select_dtypes(include=['float64']).columns.values
# Compute correlations between all numerical features
df_corr = train[num_features].corr()
#print(df_corr.sort_values(by='site_eui', key=abs, ascending=False)['site_eui'])

# Create any new features - e.g. the difference in average temperature in January (typically coldest) vs. July (typically hottest).
train['highest_temp_diff'] = train['july_max_temp'] - train['january_min_temp']
train['extremedays'] = train['days_below_30F'] + train['days_below_20F']+ train['days_below_10F'] + train['days_below_0F'] + train['days_above_80F'] + train['days_above_90F']
#train['highest_temp_diff'] = train['august_max_temp'] - train['january_min_temp']
#train['highest_temp_diff'] = train['august_max_temp'] - train['january_min_temp']



#Elevation is very lowly correlated so we will drop it
train = train.drop(['ELEVATION'], axis=1)
train = train.drop(['State_Factor'], axis=1)

#Elevation is very lowly correlated so we will drop it
train = train.drop(['days_above_110F'], axis=1)
train = train.drop(['days_above_100F'], axis=1)


groups = ['Low', 'Med', 'High']

train['ext_bin'] = pd.qcut(train['extremedays'], q=3, labels=groups)
print(train[['extremedays', 'ext_bin']].head())

#train = pd.get_dummies(train)

#print("Scaling data...", flush=True)
# Step 1: define a MinMax scalar that will transform the data values into values in (0, 1)
#scaler = MinMaxScaler()
# Step 2: fit the MinMaxScaler using our data 
#values = train.values
#scaler.fit(values)
# Step 3: scale the values in our dataset
# Hint: use .transform()
#values_scaled = scaler.transform(values)
#train = pd.DataFrame(values_scaled, columns=train.columns, index=train.index)

from sklearn.preprocessing import QuantileTransformer
#qt = QuantileTransformer(n_quantiles=10, random_state=0)
#qt.fit_transform(df_train)

#define a simple imputer, which replaces missing values using the median, using SimpleImputer class
imr = SimpleImputer(strategy = "median")

# Step 2: impute the missing data
#train = pd.DataFrame(imr.fit_transform(train), columns=train.columns, index=train.index)

#print(train.head(10))

# Split into training and test set
#predictors = [feature_name for feature_name in df_train.columns.values if feature_name != 'site_eui']
#X = train[predictors]
#y = train['site_eui']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Train linear regression model
#reg = LinearRegression().fit(X_train, y_train)
# Score the model - here the best possible score is 1.0.
# Source for how score is computed: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score
# Note that you can compute other metrics to score the model (e.g. root-mean-square deviatio https://en.wikipedia.org/wiki/Root-mean-square_deviation)
#print(reg.score(X_test, y_test))

# with quantile transformer, lin reg score was .36853099
# with minmaxscaler, lin reg score was .3557438


#from sklearn.neighbors import KNeighborsRegressor
#knn = KNeighborsRegressor(n_neighbors=3)
#knn.fit(X_train, y_train)
#print(knn.score(X_test, y_test))
#knn reg score = .2904356

#from sklearn.neural_network import MLPRegressor
#regr = MLPRegressor(random_state=1, max_iter=1000).fit(X_train, y_train)
#y_pred = regr.predict(X_test)
#print(regr.score(X_test, y_test))
#print(mean_squared_error(y_test, y_pred))
# reg score = .35114719702097674

