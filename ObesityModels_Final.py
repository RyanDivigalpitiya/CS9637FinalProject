#Ryan Divigalpitiya
#CS9637 Final Project - Predicting Obesity based on Lifestyle Choices

#IMPORT BASIC PACKAGES
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import math

#IMPORT MODEL PACKAGES
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import pickle
import warnings
import os
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

#METHODS
def importData(targetVariable, path): #returns 'importedData' dictionary. Keys are self-explanatory, below:
    # importData() extracts all the important data from the imported CSV file that we are building our model from:
    # 'RawData'                        # the raw dataframe itself, untouched
    # 'allFeatures'                    # list of column labels ie. features
    # 'numericalFeatureList'           # list of numerical features
    # 'categoricalFeatureList'         # list of categorical features
    # 'categoriesPerCategoricalColumn' : dictionary where each key is a categorical feature, key returns list of possible categories for that feature
    # 'targetClasses'                  : list of all possible target classes found in the target variable

    #read in csv fata
    df = pd.read_csv(path)
    # Split Numerical vs. Categorical Data, return them as individual lists:
    def getNumericalAndCatColumnLists():
        numerical_columns = list(df.drop(targetVariable,axis=1).describe().columns)
        categorical_columns = list(set(df.drop(targetVariable,axis=1).columns) - set(numerical_columns))
        df.dtypes[numerical_columns]   #sanity check. should all be either ints or floats
        df.dtypes[categorical_columns] #sanity check. should all be either strings or objects
        return numerical_columns, categorical_columns
    #returns categoriesPerColumn (dictionary[key = 'column name'] = list of categories in that column)
    def extractCategoriesInFeatures(categoricalFeatureList):
        x_features = df.drop(targetVariable,axis=1)
        categoriesPerColumn = {}
        for catorigicalColumn in categoricalFeatureList:
            categoriesPerColumn[catorigicalColumn] = list(df[catorigicalColumn].value_counts().index)
        return categoriesPerColumn

    featureList = getNumericalAndCatColumnLists()
    categoriesPerCategoricalColumn = extractCategoriesInFeatures(featureList[1])
    #Dictionary 'importedData' that will be returned
    importedData = {
        'RawData'                        : df,
        'allFeatures'                    : list(df.drop(targetVariable,axis=1).columns),
        'numericalFeatureList'           : featureList[0],
        'categoricalFeatureList'         : featureList[1],
        'categoriesPerCategoricalColumn' : categoriesPerCategoricalColumn,
        'targetClasses'                  : list(df[targetVariable].value_counts().index)
    }
    return importedData
def compute_performance_Series(yhat, y):
    correctCounter = 0
    for index in range((y.shape[0])):
        if (y.iloc[index]) == yhat[index]:
            correctCounter += 1
    acc = (correctCounter / y.shape[0])
    return (round(acc*100,2))
def compute_performance_Array(yhat, y):
    correctCounter = 0
    for index in range((y.shape[0])):
        if (y[index]) == yhat[index]:
            correctCounter += 1
    acc = (correctCounter / y.shape[0])
    return (round(acc*100,2))
def targetClassProportions():
    #Ouput number of 1)target classes, 2) counts per class,3) proportion of each class
    target_distr = RawData[targetVariable].value_counts()
    ds = pd.Series(index = target_distr.index).astype('str')
    for index,item in enumerate(target_distr):
        ds[index] = str(round((item/2111)*100,1))+'%'
    RawData_targetDistr = pd.DataFrame(columns = [str(len(target_distr.index))+' Classes','Counts','Percentages'], index = target_distr.index)
    RawData_targetDistr[str(len(target_distr.index))+' Classes'] = target_distr.index
    RawData_targetDistr['Counts'] = RawData[targetVariable].value_counts()
    RawData_targetDistr['Percentages'] = ds
    RawData_targetDistr.reset_index(inplace=True)
    RawData_targetDistr.drop(columns = ['index'],inplace=True)
    return RawData_targetDistr
def n_choose_k(n,k):
    return math.factorial(n)/(math.factorial(k)*(math.factorial(n-k)))
def calculateNumIterations(numberOfFeatures):
    n = numberOfFeatures
    results = []
    for k in range(2,n+1):
        results.append(n_choose_k(n,k))
    arr = np.array(results)
    print("For n=",n,sep='')
    print(str(arr.sum().astype(int))+" total iterations")
    return pd.Series(results, index = range(2,n+1))
def visualCheck(yhat,y): #returns concatenated yhat + y
    yhatSeries = pd.Series(data=yhat)
    ySeries = pd.Series(data=y)
    compareY_df = pd.concat([ySeries,yhatSeries],axis=1)
    return compareY_df

##PIPELINES

class X_Encoder(BaseEstimator, TransformerMixin):
    def __init__(self,selectedFeatures):
        self.selectedFeatures = selectedFeatures

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        # encodeXPipline encodes categorical features
        X_nonencoded = X[self.selectedFeatures]
        for columnName in categoriesPerCategoricalColumn:
            if columnName in self.selectedFeatures:
                X_nonencoded[columnName] = pd.Categorical(X_nonencoded[columnName], categories=categoriesPerCategoricalColumn[columnName])
        X_encoded = pd.get_dummies(X_nonencoded,drop_first=False)
        return X_encoded

#Testing pipe:
pipe_X_encoder = X_Encoder(allFeatures)
pipe_X_encoder.transform(RawData.drop(targetVariable,axis=1))

# pre-processing
def encodeXPipline(dataframe,selectedFeatures): #returns X_encoded (array)
    # encodeXPipline encodes categorical features
    X_nonencoded = dataframe[selectedFeatures]
    # categoricalColumns = categoriesPerColumn.keys()
    for columnName in categoriesPerCategoricalColumn:
        if columnName in selectedFeatures:
            X_nonencoded[columnName] = pd.Categorical(X_nonencoded[columnName], categories=categoriesPerCategoricalColumn[columnName])
    X_encoded = pd.get_dummies(X_nonencoded,drop_first=False)
    return X_encoded
def scaleXPipline(x): #returns X_normalized
    #Normalize X
    return StandardScaler().fit_transform(x)
def encodeYPipline(dataframe): #returns encodedTargets (y-values array)
    # encodeYPipline encodes target labels if dealing with classifcation.
    # Handles binary + multi-classifaction
    encodedTargets = np.zeros(dataframe.shape[0])
    for i in range(dataframe.shape[0]):
        for index,label in enumerate(targetClasses):
            if dataframe[targetVariable].iloc[i] == label:
                encodedTargets[i] = index+1
                break
    return encodedTargets
def preprocessXpipline(dataframe,selectedFeatures): #returns scaleXPipline(encodeXPipline(dataframe,selectedFeatures))
    return scaleXPipline(encodeXPipline(dataframe,selectedFeatures))
def splitXYPipline(X,y,test_size): #returns injectProcessedData (4-item tuple)
    # splitXYPipline splits into train and test sets
    # returns injectXtrain, injectYtrain, injectXtest, injectYtest
    # to be injected into next pipline
    ##################################################################
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=0, stratify=y) # maintain class proportions
    ##################################################################
    #Final pre-processed data to be used in training + evaluating models
    ##################################################################
    injectXtrain = Xtrain
    injectYtrain = ytrain
    ###################################
    injectXtest = Xtest
    injectYtest = ytest
    ##################################################################
    injectProcessedData = []
    injectProcessedData.append(injectXtrain)
    injectProcessedData.append(injectYtrain)
    injectProcessedData.append(injectXtest)
    injectProcessedData.append(injectYtest)
    return injectProcessedData
# models
def trainEvallogregPipeline( injectProcessedData, hyper_parameters): #returns (acc,model)
    #LOGISTICAL REGRESSION PIPELINE - returns 2-item tuple: (test set accuracy (as a percentage), model object)
    #PIPELINE CONTAINS STEPS FOR:
    #CONSTRUCTING MODEL WITH SPECIFIED HYPER-PARAMETERS
    #TRAINING
    #EVALUATING MODEL FOR SELECTED FEATURES + HYPER-PARAMETERS
    ## ******************************************************************************************************** ##

    ## BUILD MODEL ##
    if hyper_parameters == None:
        lm = LogisticRegression(multi_class = 'ovr',
                               solver      = 'liblinear')
    else:
        lm = LogisticRegression(multi_class = 'ovr',
                                solver      = 'liblinear',
                                max_iter    = hyper_parameters[0],
                                penalty     = hyper_parameters[1],
                                C           = hyper_parameters[2])

    ## UNPACK PRE-PROCESSED DATA ##
    ##################################################################
    injectXtrain = injectProcessedData[0]
    injectYtrain = injectProcessedData[1]
    ###################################
    injectXtest = injectProcessedData[2]
    injectYtest = injectProcessedData[3]
    ##################################################################
    ## TRAIN MODEL ##
    lm.fit(injectXtrain, injectYtrain)
    ## EVALUATE MODEL ##
    ypred = lm.predict(injectXtest)
    acc = compute_performance_Array(ypred,injectYtest)
    return (acc,lm)
def trainEvalforestPipeline( injectProcessedData, hyper_parameters): #returns (acc,model)
    ## BUILD MODEL ##
    if hyper_parameters == None:
        rf = RandomForestClassifier(random_state = 1, n_estimators=10)
    else:
        rf = RandomForestClassifier(random_state = 1,
                                    n_estimators        = hyper_parameters[0],
                                    bootstrap           = hyper_parameters[1],
                                    max_depth           = hyper_parameters[2],
                                    max_features        = hyper_parameters[3],
                                    min_samples_leaf    = hyper_parameters[4],
                                    min_samples_split   = hyper_parameters[5])

    ## UNPACK PRE-PROCESSED DATA ##
    ##################################################################
    injectXtrain = injectProcessedData[0]
    injectYtrain = injectProcessedData[1]
    ###################################
    injectXtest = injectProcessedData[2]
    injectYtest = injectProcessedData[3]
    ##################################################################
    ## TRAIN MODEL ##
    rf.fit(injectXtrain, injectYtrain)
    ## EVALUATE MODEL ##
    ypred = rf.predict(injectXtest)
    acc = compute_performance_Array(ypred,injectYtest)
    return (acc,rf)
def trainEvalxgbPipeline(    injectProcessedData, hyper_parameters): #retursn (acc,model)
    ## BUILD MODEL ##
    if hyper_parameters == None:
        xgb = XGBClassifier(random_state = 1)
    else:
        xgb = XGBClassifier(random_state = 1,
                                    n_estimators        = hyper_parameters[0],
                                    colsample_bytree    = hyper_parameters[1],
                                    max_depth           = hyper_parameters[2],
                                    reg_alpha           = hyper_parameters[3],
                                    reg_lambda          = hyper_parameters[4],
                                    subsample           = hyper_parameters[5],
                                    learning_rate       = hyper_parameters[6],
                                    gamma               = hyper_parameters[7],
                                    min_child_weight    = hyper_parameters[8],
                                    sampling_method     = hyper_parameters[9])

    ## UNPACK PRE-PROCESSED DATA ##
    ##################################################################
    injectXtrain = injectProcessedData[0]
    injectYtrain = injectProcessedData[1]
    ###################################
    injectXtest = injectProcessedData[2]
    injectYtest = injectProcessedData[3]
    ##################################################################
    ## TRAIN MODEL ##
    xgb.fit(injectXtrain, injectYtrain)
    ## EVALUATE MODEL ##
    ypred = xgb.predict(injectXtest)
    acc = compute_performance_Array(ypred,injectYtest)
    return (acc,xgb)
def trainEvalknnPipeline(    injectProcessedData, hyper_parameters): #retursn (acc,model)
    ## BUILD MODEL ##
    if hyper_parameters == None:
        knn = KNeighborsClassifier()
    else:
        knn = KNeighborsClassifier( n_neighbors = hyper_parameters[0],
                                    weights     = hyper_parameters[1],
                                    algorithm   = hyper_parameters[2],
                                    p           = hyper_parameters[3])

    ## UNPACK PRE-PROCESSED DATA ##
    ##################################################################
    injectXtrain = injectProcessedData[0]
    injectYtrain = injectProcessedData[1]
    ###################################
    injectXtest = injectProcessedData[2]
    injectYtest = injectProcessedData[3]
    ##################################################################
    ## TRAIN MODEL ##
    knn.fit(injectXtrain, injectYtrain)
    ## EVALUATE MODEL ##
    ypred = knn.predict(injectXtest)
    acc = compute_performance_Array(ypred,injectYtest)
    return (acc,knn)
# train + evaluate multiple models at once: options for models = { "lr": hyper_parameters, "rf": hyper_parameters, "gb": hyper_parameters }
def runEntirePipline(dataframe, selectedFeatures, models, printAcc=False): #returns modelResults (dictionary, "lr":(acc,modelObject))
    ## PRE-PROCESSING ##
    X = scaleXPipline( encodeXPipline(dataframe, selectedFeatures))
    y = encodeYPipline(dataframe)
    injectProcessedData = splitXYPipline(X,y,0.25)
    ## UNPACK REQUESTED MODELS + TRAIN/EVALUATE EACH ##
    # models = { "lr": hyper_parameters, "rf": hyper_parameters, "gb": hyper_parameters, "kn":hyper_parameters }
    modelResults={}
    #Log Reg Model
    if "lr" in models:
        acc, model = trainEvallogregPipeline( injectProcessedData, hyper_parameters = models["lr"])
        modelResults["lr"] = (acc, model)
        if printAcc:
            print("Log Reg Accuracy: ", acc,"%",sep='')
    #Random Forest Model
    if "rf" in models:
        acc, model = trainEvalforestPipeline( injectProcessedData, hyper_parameters = models["rf"])
        modelResults["rf"] = (acc, model)
        if printAcc:
            print("Random Forest Accuracy: ", acc,"%",sep='')
    #XGBoost Model
    if "gb" in models:
        acc, model = trainEvalxgbPipeline( injectProcessedData, hyper_parameters = models["gb"])
        modelResults["gb"] = (acc, model)
        if printAcc:
            print("XGB Accuracy: ", acc,"%",sep='')
    #kNN Model
    if "kn" in models:
        acc, model = trainEvalknnPipeline( injectProcessedData, hyper_parameters = models["kn"])
        modelResults["kn"] = (acc, model)
        if printAcc:
            print("kNN Accuracy: ", acc,"%",sep='')

    return modelResults

##GRAPHS
def showFeatureImportances(model,selectedFeatures): #for Random Forest estimators, visualize feature impotances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fm, ax = plt.subplots(figsize=(3,8))
    plt.title("Variable Importances - XGBoost")
    sns.set_color_codes("pastel")
    dfe = encodeXPipline(RawData,selectedFeatures)
    columnsEnc = list(dfe.columns)
    sns.barplot(y=[columnsEnc[i] for i in indices], x=importances[indices], label='Total',color='b')
    ax.set(ylabel="Variable",xlabel = "Variable Importance (Gini)")
    sns.despine(left=True, bottom=True)
#EndOfImports+Definitions-------------------------------------------------------------------------------------------------------------------------------
#BeginningofScript-------------------------------------------------------------------------------------------------------------------------------

## READ IN DATA ## (modify for Mac vs. Windows)
#############################################################################################################################################################
targetVariable                  = 'NObeyesdad'
importedData                    = importData(targetVariable, '/Users/ryandivigalpitiya/Virtual Envs/TensorFlow/My Python Files/ObesityModels/ObesityData.csv')
#############################################################################################################################################################
#These must be defined in order for the above functions to work (most of them make use of these variables)
RawData                         = importedData['RawData']
allFeatures                     = importedData['allFeatures']
numericalFeatureList            = importedData['numericalFeatureList']
categoricalFeatureList          = importedData['categoricalFeatureList']
categoriesPerCategoricalColumn  = importedData['categoriesPerCategoricalColumn']
targetClasses                   = importedData['targetClasses']
#############################################################################################################################################################

## OPTIONAL: LOAD MODEL FROM DISK ##
############################################################
# filename = r'C:\Users\Admin\Downloads\bestRFmodelV1.sav'
# loadedModel = pickle.load(open(filename,'rb'))
# loadedModel
############################################################

#Class Proportions:
classProportions = pd.DataFrame(columns=['Counts','Proportions'], index = RawData[targetVariable].value_counts())
classProportions['Counts'] = RawData[targetVariable].value_counts()
classProportions['Proportions'] = round((RawData[targetVariable].value_counts() / 2111)*100,1)
classProportions.to_csv('/Users/ryandivigalpitiya/Python Notebooks/CS 9637/Project/classProp.csv',index=False)


#Baseline performance (No tuning)
# acc,model = runEntirePipline(RawData,allFeatures,targetVariable,printAcc=True)
hyper_parameters = None
# optFeatures = ['Gender','Age','family_history_with_overweight','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
# models = { "lr": hyper_parameters, "rf": hyper_parameters, "gb": hyper_parameters }
#Test all models for baseeline tests:
model = { "lr":None, "rf":None, "gb":None }
#Remove Height and Weight features (and target variable)
features = RawData.drop(columns=[targetVariable,'Height','Weight']).columns
modelResults = runEntirePipline(RawData, features, model, printAcc=True)
#Remove Age + Family History in additon to Height and Weight features (and target variable) for comparison
features = RawData.drop(columns=[targetVariable,'Height','Weight','Age','family_history_with_overweight']).columns
modelResults = runEntirePipline(RawData, features, model, printAcc=True)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## TESTING BASELINE MODELS ON ALL DATA ##
lrModel = modelResults['lr'][1]
rfModel = modelResults['rf'][1]
gbModel = modelResults['gb'][1]
##
X = preprocessXpipline(RawData,features)
lrYhat = lrModel.predict(X)
rfYhat = rfModel.predict(X)
gbYhat = gbModel.predict(X)
lrPerformanceAllData = compute_performance_Array(lrYhat,encodeYPipline(RawData))
rfPerformanceAllData = compute_performance_Array(rfYhat,encodeYPipline(RawData))
gbPerformanceAllData = compute_performance_Array(gbYhat,encodeYPipline(RawData))

lrPerformanceAllData
rfPerformanceAllData
gbPerformanceAllData

visualCheck(gbYhat,encodeYPipline(RawData)).head(25)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## TUNING ##:

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## FEATURE SPACE TUNING ##
# We'll conduct feature reduction tests to see how reducing the feature space impacts performance

# There are a total of 16 features

# We thus have to write a method that generates every possible combination of these features
# and then train/evaluate our models for each combination of features to find which feature combo delivers the highest accuracy
# First, we need to determine if this is computationally feasible - ie. How many unique combinations can result from a list of 14 unique items?
# If the number is too large, then we'll have to try a different approach
# We'll have to use Combination mathematics to determine this.
# We can use the formula n-choose-k: (n k) = (!n)/(k!(n-k)!), where,
#   n = number of elements in the list (this is 16)
#   k = the size of the combination (this starts at 16, decreases down to 2 iteratively)
# which gives the total number of possible combinations.
# We then place this formula in a For Loop to determine the total number of combinations by summing up the results of the formula evaluated for each k
# (k ranges from 2 to 16)
# we have defined two methods to do this: n_choose_k(n,k) and calculateNumIterations(numberOfFeatures) which is called in the below method

# For the actual generation of the feature combinations, we will use the package itertools.combinations(list_of_size_n,k))
#find optimal combo of features for specified features and model type
def optimalFeatureCombos(fullFeatureList,model):
    #Show how many iterations need to be computed:
    calculateNumIterations(len(fullFeatureList))
    # Outer For Loop, looping from k = 2 to 16
    y = encodeYPipline(RawData)
    results = []
    for k in range(2,len(fullFeatureList)+1):
        print("\nk combinations:",k,"\t",int(n_choose_k(len(fullFeatureList),k)),"iterations") #number of combinations (iterations) for this k)
        featureSpace = list(itertools.combinations((fullFeatureList),k)) # train/evaluate model for each feature combination
        accuracies = []
        featureComboUsed = []
        for index,attributeCombo in enumerate(featureSpace):   #loop over featureSpace
            selectedFeatures = list(attributeCombo)
            #X = scaleXPipline( encodeXPipline(df, selectedFeatures))
            #injectProcessedData = splitXYPipline(X,y,0.25)
            #acc,_ = trainEvalxgbPipeline(   injectProcessedData, None) #train/evaluate model for each attribute combination
            #For reference: models = { "lr": hyper_parameters, "rf": hyper_parameters, "gb": hyper_parameters }
            modelResults = runEntirePipline(RawData,selectedFeatures,model,printAcc=False)
            acc = modelResults[list(model.keys())[0]][0]
            accuracies.append(acc)
            featureComboUsed.append(attributeCombo)
            # print(index+1,end=" ")
        #record feature combo @ k that gave highest accuracy + store in results[] along with that feature combo
        max = np.array(accuracies).max()
        index = accuracies.index(max)
        results.append((featureSpace[index],max)) #this list will be of size k

    #find highest accuracy and feature combo that gave that result
    allAcc = []
    for i in range(len(results)):
        allAcc.append(results[i][1])
    val =  np.array(allAcc).max()
    indices=[]
    for index,acc in enumerate(allAcc): # there may be multiple feature combos that gave the same highest accuracy (although rare)
        if acc == val:
            indices.append(index)

    #OUTPUT
    # 1) Most optimal combination of features that yields the highest accuracy:
    output = []
    for index in indices:
        output.append((results[index][0],results[index][1]))
        # output:
        print("Optimal features:",results[index][0])
        print("Accuracy:",results[index][1])
    return output, results

fullFeatureList = list(RawData.drop([targetVariable,'Height','Weight'],axis=1).columns)
model = { "rf": None}
output, results = optimalFeatureCombos(fullFeatureList,model)

# bestFeatureCombo = output[0][0] #if output has mutliple values, that means any one of them achieved the highest accuracy
bestFeatureCombo = list(output[0][0])
bestAcc = output[0][1]
bestFeatureCombo
bestAcc

#HYPER-PARAMTER TUNING--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## LOGISITCAL REGRESSION ## <----- IGNORE! HAS NOT BEEN UPDATED. WE ARE DROPPING LOG REG DUE TO POOR PERFOMANCEE

# We will search over  hyper-parmeter sets for most optimal params
#outdated method, update it before using:
def hyperParamSearch_LogReg(features):

    # Search over this hyper-parameter space
    max_iters    = [100,1000,10000]         # (1) number of iterations
    penalties    = ['l1', 'l2']             # (2) type of regularization
    C_values     = np.logspace(-4, 4, 20)   # (3) inverse of regularization strength
    multi_class  = 'ovr'                    # (_) this is a multi-classification problem (non-binary), thus, we use one-vs-all multi-class logistical regression
    solver       = 'liblinear'              # (_) using one-vs-all multi-classifaction, thus, must use liblinear solver
    ###############################################################
    hyper_parameters = [max_iters, penalties, C_values]
    ###############################################################
    #Number of hyper-params in hyper-parameter space:
    print("Number of Hyper-Parameters:",len(hyper_parameters[0])*len(hyper_parameters[1])*len(hyper_parameters[2]))

    bestAcc = 0
    progressBar = 0
    X = scaleXPipline( encodeXPipline(RawData, features))
    y = encodeYPipline(RawData)
    injectProcessedData = splitXYPipline(X,y,0.25)
    # iterating over hyper-parameter space:
    for max_iter in hyper_parameters[0]:
        for penalty in hyper_parameters[1]:
            for C in hyper_parameters[2]:
                progressBar += 1
                print(progressBar,end=" ")
                selected_hyper_parameters = [max_iter,penalty,C]
                acc, model = trainEvallogregPipeline( injectProcessedData, selected_hyper_parameters)

                if acc > bestAcc:
                    bestAcc = acc
                    bestModel = model
                    optimHypParameters = hyper_parameters
    print("LogReg Best Tuned Accuracy: ",bestAcc,"%",sep='')
    return bestAcc, bestModel, optimHypParameters

bestAccLogReg, optimHypParametersReg, optimFeatureComboReg = hyperParamSearch_LogReg(allFeatures)

##########
bestAccReg
##########
optimHypParametersReg
##########

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## XTREME GRADIENT BOOST TUNING ##

# We will search over the sets for most optimal params
def hyperParamSearch_GB(features):

    # Search over this hyper-parameter space:
    n_estimators        = [600,10000]
    colsample_bytree    = [0.75,0.8,0.85]
    max_depth           = [20, 50, 75]
    reg_alpha           = [1]
    reg_lambda          = [2, 5, 10]
    subsample           = [0.55, 0.6, .65]
    learning_rate       = [0.5]
    gamma               = [.5,1,2]
    min_child_weight    = [0.01]
    sampling_method     = ['uniform']

    ###############################################################
    hyper_parameters = [n_estimators, colsample_bytree, max_depth, reg_alpha, reg_lambda, subsample, learning_rate, gamma, min_child_weight, sampling_method]
    ###############################################################
    #Number of hyper-params in hyper-parameter space:

    paramCounter = len(n_estimators)* len(colsample_bytree)* len(max_depth)* len(reg_alpha)* len(reg_lambda)* len(subsample)* len(learning_rate)* len(gamma)* len(min_child_weight)* len(sampling_method)
    print("Number of Hyper-Parameters:",paramCounter)

    bestAcc = 0
    progressBarOne = 0
    progressBarTwo = 0
    X = scaleXPipline( encodeXPipline(RawData, features))
    y = encodeYPipline(RawData)
    injectProcessedData = splitXYPipline(X,y,0.25)
    # iterating over hyper-parameter space:
    for p1 in hyper_parameters[0]:
        for p2 in hyper_parameters[1]:
            for p3 in hyper_parameters[2]:
                for p4 in hyper_parameters[3]:
                    for p5 in hyper_parameters[4]:
                        progressBarOne += 1
                        print("\n",progressBarOne)
                        for p6 in hyper_parameters[5]:
                            for p7 in hyper_parameters[6]:
                                for p8 in hyper_parameters[7]:
                                    for p9 in hyper_parameters[8]:
                                        for p10 in hyper_parameters[9]:
                                            progressBarTwo += 1
                                            print(progressBarTwo,end=" ")
                                            selected_hyper_parameters = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10]
                                            acc, model = trainEvalxgbPipeline( injectProcessedData, hyper_parameters = selected_hyper_parameters)

                                            if acc > bestAcc:
                                                bestAcc = acc
                                                bestModel = model
                                                optimHypParameters = selected_hyper_parameters

    print("XGB Best Tuned Accuracy: ",bestAcc,"%",sep='')
    return bestAcc, bestModel, optimHypParameters

bestFeatureCombo = ['Gender','Age','family_history_with_overweight','FAVC','NCP','CAEC','CH2O','SCC','FAF','TUE','CALC']
bestAccGB, bestModelGB, optimHypParametersGB = hyperParamSearch_GB(bestFeatureCombo)
##########
bestAccGB
##########
optimHypParametersGB
##########

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## RANDOM FOREST TUNING ##

# We will search over hyper-parameter sets for most optimal params
def hyperParamSearch_RF(features):

    # Search over this hyper-parameter space:
    n_estimators        = [10000,100000]
    bootstrap           = [False]
    max_depth           = [10,20,50,None]
    max_features        = ['auto']
    min_samples_leaf    = [1,2,4]
    min_samples_split   = [2,5]
    ###############################################################
    hyper_parameters = [n_estimators, bootstrap, max_depth, max_features, min_samples_leaf, min_samples_split]
    ###############################################################
    #Number of hyper-params in hyper-parameter space:

    paramCounter = len(n_estimators)* len(bootstrap)* len(max_depth)* len(max_features)* len(min_samples_leaf)* len(min_samples_split)
    print("Number of Hyper-Parameters:",paramCounter)

    bestAcc = 0
    progressBarOne = 0
    progressBarTwo = 0
    X = scaleXPipline( encodeXPipline( RawData, features))
    y = encodeYPipline(RawData)
    injectProcessedData = splitXYPipline(X,y,0.25)
    # iterating over hyper-parameter space:
    for p1 in hyper_parameters[0]:
        print("\n",p1,"n_estimators")
        for p2 in hyper_parameters[1]:
            for p3 in hyper_parameters[2]:
                progressBarOne += 1
                print("\n",progressBarOne)
                for p4 in hyper_parameters[3]:
                    for p5 in hyper_parameters[4]:
                        for p6 in hyper_parameters[5]:
                            progressBarTwo += 1
                            print(progressBarTwo,end=" ")
                            selected_hyper_parameters = [p1,p2,p3,p4,p5,p6]
                            acc, model = trainEvalforestPipeline( injectProcessedData, hyper_parameters = selected_hyper_parameters)

                            if acc > bestAcc:
                                bestAcc = acc
                                bestModel = model
                                optimHypParameters = selected_hyper_parameters

    print("RF Best Tuned Accuracy: ",bestAcc,"%",sep='')
    return bestAcc, bestModel, optimHypParameters

bestRFAcc, bestRFmodel, optimHypParametersRF = hyperParamSearch_RF(bestFeatureComboRF)

##########
bestRFAcc
##########
optimHypParametersRF
##########

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## K-NEAREST NEIGHBOURS TUNING ##

# We will search over the above 10 hyper-parameter sets for most optimal params
def hyperParamSearch_kNN(features):

    # Search over this hyper-parameter space for XGB:
    n_neighbors = [3,5,7,9,11,13,15]
    weights     = ['uniform', 'distance']
    algorithm   = ['auto', 'ball_tree','kd_tree']
    p           = [1,2]
    ###############################################################
    hyper_parameters = [n_neighbors, weights, algorithm, p]
    ###############################################################
    #Number of hyper-params in hyper-parameter space:

    paramCounter = len(n_neighbors)* len(weights)* len(algorithm)* len(p)
    print("Number of Hyper-Parameters:",paramCounter)

    bestAcc = 0
    progressBarOne = 0
    progressBarTwo = 0
    X = scaleXPipline( encodeXPipline( RawData, features))
    y = encodeYPipline(RawData)
    injectProcessedData = splitXYPipline(X,y,0.25)
    # iterating over hyper-parameter space:
    for p1 in hyper_parameters[0]:
        print("\n",p1,"n_neighbors")
        for p2 in hyper_parameters[1]:
            for p3 in hyper_parameters[2]:
                for p4 in hyper_parameters[3]:
                    progressBarTwo += 1
                    print(progressBarTwo,end=" ")
                    selected_hyper_parameters = [p1,p2,p3,p4]
                    acc, model = trainEvalknnPipeline( injectProcessedData, selected_hyper_parameters)

                    if acc > bestAcc:
                        bestAcc = acc
                        bestModel = model
                        optimHypParameters = selected_hyper_parameters

    print("kNN Best Tuned Accuracy: ",bestAcc,"%",sep='')
    return bestAcc, bestModel, optimHypParameters

bestkNNAcc, bestkNNmodel, optimHypParameterskNN = hyperParamSearch_kNN(bestFeatureCombo)

##########
bestkNNAcc
##########
optimHypParameterskNN
##########

#MODEL PERSISTANCE--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

filename = r'C:\Users\Admin\Downloads\bestRFmodelV1.sav'
pickle.dump(bestRFmodel,open(filename,'wb'))

# loadedModel = pickle.load(open(filename,'rb'))
# loadedModel

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Test on All Data (as sanity check - should be higher accuracy than test set)
tunedBestModel = ?
performance = compute_performance_Array(tunedBestModel.predict(scaleXPipline(encodeXPipline(RawData,bestFeatureCombo))),encodeYPipline(RawData))
performance

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## OPTIONAL: LOAD MODEL FROM DISK ##
################################################################################################################################
filename = '/Users/ryandivigalpitiya/Python Notebooks/CS 9637/Project/bestRFmodel.sav'
loadedModel = pickle.load(open(filename,'rb'))
loadedModel
################################################################################################################################
# Or, re-train using tuned HP+Features:
tunedFeaturesRF = ['Gender','Age','family_history_with_overweight','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
modelWithTunedHP_RF = {'rf':[10000, False, 50, 'auto', 1, 2]}
modelResults = runEntirePipline(RawData,tunedFeaturesRF,modelWithTunedHP_RF) #Running line below takes 24 seconds on MacBook Pro:
print("RF Accuracy:",modelResults['rf'][0]) #Should be 86.55%
rfModel = modelResults['rf'][1] #extract rf model
#Save model to disk:
filename = '/Users/ryandivigalpitiya/Python Notebooks/CS 9637/Project/bestRFmodel.sav'
pickle.dump(rfModel,open(filename,'wb'))


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# FEATURE IMPORTANCES PER MODEL

#RF
#trained model:
# showFeatureImportances(rfModel,tunedFeaturesRF)
#loaded model from disk:
showFeatureImportances(loadedModel,tunedFeaturesRF)

#XGBoost
tunedFeaturesGB = ['Gender', 'Age', 'family_history_with_overweight', 'FAVC', 'NCP', 'CAEC', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC']
modelWithTunedHP_GB = {'gb':[600, 0.75, 20, 1, 5, 0.6, 0.5, 0.5, 0.01, 'uniform']}
modelResults = runEntirePipline(RawData,tunedFeaturesGB,modelWithTunedHP_GB)
print("GB Accuracy:",modelResults['gb'][0])
gbModel = modelResults['gb'][1] #extract gb model
showFeatureImportances(gbModel,tunedFeaturesGB)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## GENERATING MULTI-CLASSIFICATION CONFUSION MATRIX ##

#generate test set, use test set to generate my Predicted column + Actual column:
X = scaleXPipline( encodeXPipline(RawData, optFeatures))
y = encodeYPipline(RawData)
injectProcessedData = splitXYPipline(X,y,0.25)

# injectProcessedData.append(injectXtrain)
# injectProcessedData.append(injectYtrain)
# injectProcessedData.append(injectXtest)
# injectProcessedData.append(injectYtest)
# Thus: xtest = injectProcessedData[2], ytest = injectProcessedData[3]
xtest = injectProcessedData[2]
ytest = injectProcessedData[3] #length y: 528
# compute_performance_Array(loadedModel.predict(xtest),ytest)
predVSactual = np.zeros((ytest,2))
pred = loadedModel.predict(xtest)
predVSactual[:,0] = pred
predVSactual[:,1] = ytest #actual

# predVSactual_df = pd.DataFrame(data=predVSactual,columns=['pred','actual'])
# predVSactual_df.to_csv('/Users/ryandivigalpitiya/Python Notebooks/CS 9637/Project/predVSactual.csv',index=False)
cf = np.zeros((7,7))
finalIndex=0
for i in range(len(predVSactual)):
    predicted = int(predVSactual[i][0]-1)
    actual    = int(predVSactual[i][1]-1)
    cf[predicted][actual] = cf[predicted][actual] + 1

# create empty CF Matrix:
classRange = np.linspace(1,7,7)
columns = list(classRange.astype(int))
index = list(classRange.astype(int))
cf_df = pd.DataFrame(data=cf,columns=columns,index=index)
cf_df.to_csv('/Users/ryandivigalpitiya/Python Notebooks/CS 9637/Project/confusionMatrix.csv',index=False)

unique_elements, counts_elements = np.unique(ytest, return_counts=True)
print(unique_elements )
print(counts_elements)
counts = np.zeros((7,2))
counts[:,0] = unique_elements
counts[:,1] = counts_elements

dfCounts = pd.DataFrame(data=counts,index=index)
dfCounts.to_csv('/Users/ryandivigalpitiya/Python Notebooks/CS 9637/Project/dfCounts.csv',index=False)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## ATTEMPT TO GENERATE AUROC CURVES ##
## SKLearn suggests a technique called macro-averaging which we will do here
## It revolves around approaching TP and FP calculations for multiclassifcation with a binarized one-versus-all approach
## Macro-averaging is suggested if classes are balanced as they are in our dataset
# This code was taken and repurposed from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import interp

y_test_enc_df = pd.get_dummies(ytest,drop_first=False)
y_test_enc = y_test_enc_df.to_numpy()

y_score = loadedModel.predict_proba(xtest)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(7):
    fpr[i], tpr[i], _ = roc_curve(y_test_enc[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_enc.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(7)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(7):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 7

# Calculate macro-average ROC curve and ROC area
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#Visualize macro-averaged AUROC:
plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle='-', linewidth=4)

plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-Average AUROC')
plt.show()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------































#end of file
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
