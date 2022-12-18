import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import axes
import seaborn as sns

dataset = pd.read_excel("HousePricePrediction.xlsx")

# Printing the first 5 records of the data set
print(dataset.head(5))

# Printing the dimensions of the data set
print(f"Dimensions: {dataset.shape}")

'''
Data Pre-Processing
    - Categorizing features (columns) depending on their datatype and calculating the number of them
'''

feat_obj = (dataset.dtypes == "object")     # evaluating the features of the dataset with datatype object
feat_obj_col = list(feat_obj[feat_obj].index)       # storing all the columns that evaluate to true
print(f"Categorical variables: {len(feat_obj_col)}")

feat_int = (dataset.dtypes == "int")     # evaluating the features of the dataset with datatype integer
feat_int_col = list(feat_int[feat_int].index)       # storing all the columns that evaluate to true
print(f"Integer variables: {len(feat_int_col)}")

feat_float = (dataset.dtypes == "float")        # evaluating the features of the dataset with datatype float
feat_float_col = list(feat_float[feat_float].index)     # storing all the columns that evaluate to true
print(f"Float variables: {len(feat_float_col)}")
print("\n")

'''
Exploratory Data Analysis:
    - Deep analysis of data to discover patterns and spot anomalies before making any inferences.
    - Creating a Heatmap using the Seaborn library (https://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap#seaborn.heatmap)
        - Heatmap == data visualization technique that shows the magnitude of a phenomenon as color in two dimensions;
            variation of colour hues or intensity indicates how the studied phenomenon varies over space.
        - https://en.wikipedia.org/wiki/Heat_map
    - Drawing a barplot to compare the number of unique values in each object feature
    - Drawing a barplot for each feature to obtain the actual count of each category

Note: set numeric_only to True so the corr() method can ignore columns that contain non-numbers

'''

# Heat Map using Seaborn

plt.figure(figsize = (12, 6))
sns.heatmap(dataset.corr(numeric_only=True),    # corr() method of the Pandas module returns the Correlation matrix of the data frame; needed since Seaborn's heatmap() accepts a 2d dataset only
            cmap = "BrBG",                      # Choosing a Color Map in Matlplotlib. BrBG is the code for a Diverging Color Map (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
            fmt = ".2f",                        # Format for the data to be displayed (here, floating point decimal with 2 decimal spaces)
            linewidths = 2,                     # Width of the lines that divide each cell
            annot = True)                       # True ==> write data value in each cell
plt.show()

# Barplot, comparing number of unique values in each object feature

unique_values = []
for col in feat_obj_col:        # storing all the unique values
    unique_values.append(dataset[col].unique().size)
plt.figure(figsize = (10, 6))
plt.title("Number of Unique Values of Categorical Features")
sns.barplot(x = feat_obj_col, y = unique_values)        # creating a barplot of the unique values versus the object columns
plt.show()

# Barplot, showing the actual count of each unique value of each object feature

plt.figure(figsize = (18, 36))
plt.title("Categorial Features Distribution")
plt.xticks(rotation = 90)       # rotating the x-axis labels by 90deg

num_sub_plot = 1
for col in feat_obj_col:
    y = dataset[col].value_counts()
    plt.subplot(11, 4, num_sub_plot)
    plt.xticks(rotation = 90)
    sns.barplot(x = list(y.index), y=y)
    num_sub_plot += 1

plt.show()

'''
Data Cleaning:
    - Improvising the dataset (removing incorrect, corrupted, or irrelevant data)
        1. Dropping the Id column, since it does not participate in any prediction (it is irrelevant)
        2. Sale Price column has a few empty values, so fill those empty values with the mean of the SalePrice column (to make the data symmetric)

'''

dataset.drop(["Id"], axis = 1, inplace = True)      # Id column is irrelevant to prediction
salePrice_mean = dataset["SalePrice"].mean()
dataset["SalePrice"] = dataset["SalePrice"].fillna(salePrice_mean)      # Filling the empty values of SalePrice column with the mean of the column

# Un-comment the following line of code to check if there are any empty values remaining in the dataset
new_dataset = dataset.dropna()
#print(new_dataset.isnull().sum())

'''
One Hot Encoder, by SciKit-Learn
    - Converts object data (i.e., non-numerical data) into integer type. This is to avoid machine learning models misinterpreting non-numerical data 
        as numerical data by placing any significance on them
    - https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/ 
    - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder.fit_transform 
'''

from sklearn.preprocessing import OneHotEncoder

# Gathering a list of all categorical data
s = (new_dataset.dtypes == "object")
feat_obj_col = list(s[s].index)
print("Categorical Variable: ", feat_obj_col)
print("Number of categorical features: ", len(feat_obj_col))
print("\n")

# Applying OneHotEncoder to the list
OH_encoder = OneHotEncoder(sparse_output = False)      # sparse parameter indicates whether to return a Sparse matrix (default set to True)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[feat_obj_col]))     # fits transformer to data, then return transformed version of data
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()        # returns the output feature names of the transformation
df_final = new_dataset.drop(feat_obj_col, axis = 1)     # dropping empty values along a particular axis (0 is index, 1 is column)
df_final = pd.concat([df_final, OH_cols], axis = 1)     # concatenate panda objects along a particular axis; axis=0 is along index, axis=1 is along columns

# Splitting Dataset into Training and Testing  (Y is SalePrice, X is the other columns)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_final.drop(["SalePrice"], axis = 1)      # Dropping the SalePrice column; storing remaining columns into X
Y = df_final["SalePrice"]

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = 0)

'''
Model and Accuracy:
    - Using the following regression models to train the model and determine the continuous values:
        1. Support Vector Machine       (https://www.geeksforgeeks.org/support-vector-machine-algorithm/)
        2. Random Forest Regressor
        3. Linear Regressor
    - Calculating loss using Mean Absolute Percentage Error 

'''

# Support Vector Machine
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
Y_pred = model_SVR.predict(X_valid)

print("Loss in Support Vector Machine: ", mean_absolute_percentage_error(Y_valid, Y_pred))

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(n_estimators = 10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)

print("Loss in Random Forest Regressor: ", mean_absolute_percentage_error(Y_valid, Y_pred))

# Linear Regression
from sklearn.linear_model import LinearRegression

model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)

print("Loss in Linear Regression: ", mean_absolute_percentage_error(Y_valid, Y_pred))

