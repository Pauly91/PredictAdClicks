from pandas import read_csv, Series, DataFrame, to_datetime, TimeGrouper, concat, rolling_mean
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox, multivariate_normal
from numpy import ones, log
from pandas import to_numeric, options, tools, scatter_matrix, DataFrame
from matplotlib import pyplot
import numpy as np
from scipy import stats
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import csv
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA


matplotlib.style.use('ggplot')
'''


Variable	Description
ID	Unique ID
datetime	timestamp
siteid	website id
offerid	offer id (commission based offers)
category	offer category
merchant	seller ID
countrycode	country where affiliates reach is present
browserid	browser used
devid	device used
click	target variable
'''


var = ['datetime','siteid', 'offerid','category', 'merchant', 'countrycode' ,'browserid' ,'devid']
target = ['click']
def dataSplitter(df, y, type):
    '''

    :param df: The data  
    :param y: responses for data
    :param type: type of ML problem
    :return: Train and validation data


    Read about Stratified splitting of data.

    Material on class imbalance : https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/

    '''
    ySplit = y.values
    print(np.shape(ySplit))
    ySplit = np.reshape(ySplit, [len(ySplit), ])
    if type == 1:  # Imbalanced classification

        eval_size = 0.10
        kf = StratifiedKFold(ySplit[:], round(1. / eval_size), shuffle=True)
        train_indices, valid_indices = next(iter(kf))
        print(max(train_indices))
        print(max(valid_indices))
        x_train, y_train = df.ix[train_indices], y.ix[train_indices]
        x_valid, y_valid = df.ix[valid_indices], y.ix[valid_indices]

    # fill rest of the methods for splitting the data

    return x_train, y_train, x_valid, y_valid


def categoricalDataAnalysis(df):
    dtype_df = df[catVar].dtypes.reset_index()
    dtype_df.columns = ["Count", "Column Type"]

    print(df['Product_Info_2'].value_counts())  # The only categorical data without numerical val


def categoricalDataHandling(df):
    # Converts labels to numbers i.e encodes the data
    labelEncoder = LabelEncoder()
    labelEncoder.fit(df['Product_Info_2'])
    product_Info_2LabelEncoder = labelEncoder.transform(df['Product_Info_2'])
    df['product_Info_2LabelEncoded'] = product_Info_2LabelEncoder

    df.drop('Product_Info_2', axis=1, inplace=True)
    print(df.head(10))

    # Refer this : http://biggyani.blogspot.in/2014/08/using-onehot-with-categorical.html
    # The idea to transform DataFrame to np.array and process it. Read about it more to get a feel of it


    train_categorical_values = np.array(df)
    ohe = OneHotEncoder()

    train_cat_data = ohe.fit_transform(train_categorical_values)

    train_cat_data_df = DataFrame(train_cat_data.toarray())
    print(train_cat_data_df.describe)

    # PCA on the data

    pca = PCA(n_components=60)  # how to choose the components

    train_cat_data_array = np.array(train_cat_data_df)
    pca.fit(train_cat_data_df)
    train_cat_data_PCA = pca.transform(train_cat_data_array)
    print(len(train_cat_data_PCA))
    train_cat_data_df_pca = DataFrame(train_cat_data_PCA)
    print(train_cat_data_df_pca.describe)

    return train_cat_data_df_pca


def featurePriority(df):
    pass


# Trying to implement this :
# http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/


def statAnalysis(df):
    print(df.describe())
    print(df.isnull())
    print(df['siteid'])


def getSpecificCSV(featureList):
    dfTrain = read_csv("train.csv", header=0)
    for features in featureList:
        feature = ['click', features]
        dfOfferId = dfTrain[feature]
        dfOfferId.to_csv(features+'.csv', index=False,header=feature)
def testFunction():
    dfTrain = read_csv("trainSample.csv", header=0)
    #dfTest = read_csv("test.csv", header=0)
    dfTrain.drop('ID', axis=1, inplace = True)
    for v in var:
        dfTrain[v] = dfTrain[v].astype('category')
    df = DataFrame(dfTrain[['click','offerid']])
    dfGroupBy = df.groupby('offerid')
    print(dfGroupBy.count())


def featureBuilder(df):
    dfOfferID = read_csv("offerid.csv", header=0)

    # refer this link to work on dataframes : http://www.gregreda.com/2013/10/26/working-with-pandas-dataframes/
    '''
    
    Ideas:
    
    1. Build a feature builder, based on the number of times that corresponding 
    offer ID was clicked. Build a 10 (or higher) level confidence for the 
    ID.
    
    2. K means to group ID together
    
    '''

def main():
    testFunction()
    #featureList = ['siteid', 'offerid']
    #getSpecificCSV(featureList)
    '''
    dfTrain = read_csv("trainSample.csv", header=0)
    #dfTest = read_csv("test.csv", header=0)
    dfTrain.drop('ID', axis=1, inplace = True)
    dfTrain.drop('click', axis=1, inplace= True)
    for v in var:
        dfTrain[v] = dfTrain[v].astype('category')
    statAnalysis(dfTrain)
    '''
    #y = DataFrame(dfTrain['Response'].astype('category'))
    #dfTrain.drop('Response', axis=1, inplace=True)
    #dfTrain.drop('Id', axis=1, inplace=True)

    '''
    yCount = y['Response'].value_counts()
    print(yCount)

    The above results reveal that it is imbalanced multi-class classification problem 

    # Use of Stratified splitting; ref : http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/

    '''
    #[x_train, y_train, x_valid, y_valid] = dataSplitter(dfTrain, y, 1)
    #print(x_train[catVar].head(10))
    #categoricalDataAnalysis(x_train[catVar])
    #df = categoricalDataHandling(x_train[catVar])

    #featurePriority(df)


if __name__ == '__main__':
    main()
'''


Variable	Description
ID	Unique ID
datetime	timestamp
siteid	website id
offerid	offer id (commission based offers)
category	offer category
merchant	seller ID
countrycode	country where affiliates reach is present
browserid	browser used
devid	device used
click	target variable
'''
