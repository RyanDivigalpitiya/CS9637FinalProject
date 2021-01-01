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
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc
import pickle
import warnings
import os
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

#FUNCTIONS
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
def compute_performance_Array(yhat, y):
    correctCounter = 0
    for index in range((y.shape[0])):
        if (y[index]) == yhat[index]:
            correctCounter += 1
    acc = (correctCounter / y.shape[0])
    return (round(acc*100,2))
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
def hyperParameterSearch(model,paramGrid,detailedProgressBar=False): #returns bestAccuracy, bestModel, bestParameters
    #Takes ParamterGrid() and a Model() estimator and
    #searches for best set of hyper-parameters using compute_performance_Array()

    numOfParams  = len(paramGrid)
    loopCounter  = 0
    endOfLine=0
    bestAccuracy = 0

    print("Starting Search. 0% Complete...",end='\n')
    for paramaterSet in paramGrid:
        #Progress bar
        loopCounter += 1
        if detailedProgressBar:
            if loopCounter%10 == 0:
                endOfLine+=1
                print(round(100*(loopCounter/numOfParams)),"%",sep='',end='... ')
                if endOfLine == 20:
                    print()
                    endOfLine=0
        else:
            if loopCounter   == round(numOfParams*0.10):
                print("10% Complete...",end='\n')
            elif loopCounter == round(numOfParams*0.20):
                print("20% Complete...",end='\n')
            elif loopCounter == round(numOfParams*0.30):
                print("30% Complete...",end='\n')
            elif loopCounter == round(numOfParams*0.40):
                print("40% Complete...",end='\n')
            elif loopCounter == round(numOfParams*0.50):
                print("50% Complete...",end='\n')
            elif loopCounter == round(numOfParams*0.60):
                print("60% Complete...",end='\n')
            elif loopCounter == round(numOfParams*0.70):
                print("70% Complete...",end='\n')
            elif loopCounter == round(numOfParams*0.80):
                print("80% Complete...",end='\n')
            elif loopCounter == round(numOfParams*0.90):
                print("90% Complete...",end='\n')

        #Train/fit model on selected hyper-parameters
        model.set_params(**paramaterSet)
        accuracy = compute_performance_Array( model.fit(Xtrain,ytrain).predict(Xtest) , ytest)
        # Save best results
        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestModel = model
            bestParameters = paramaterSet
    #Print results
    print("Best Found Accuracy: "  , bestAccuracy, "%", sep='')
    print("Best Found Parameters:" , bestParameters)
    #Return results
    return bestAccuracy, bestModel, bestParameters
def exhaustiveFeatureSearch(model, startingFeatures): #returns output, results
    # exhaustiveFeatureSearch generates every possible combination of these features
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

    #Show how many iterations need to be computed:
    calculateNumIterations(len(startingFeatures))
    # Outer For Loop, looping from k = 2 to 16
    y = encodeYPipline(RawData)
    results = []
    for k in range(2,len(startingFeatures)+1):
        print("\nk combinations:",k,"\t",int(n_choose_k(len(startingFeatures),k)),"iterations") #number of combinations (iterations) for this k)
        featureSpace = list(itertools.combinations((startingFeatures),k)) # train/evaluate model for each feature combination
        accuracies = []
        featureComboUsed = []
        for index,attributeCombo in enumerate(featureSpace):   #loop over featureSpace

            selectedFeatures = list(attributeCombo)

            XtrainReduced, XtestReduced, ytrainReduced, ytestReduced = preProcessingPipeline( selectedFeatures )
            acc = compute_performance_Array(model.fit(XtrainReduced,ytrainReduced).predict(XtestReduced), ytestReduced)
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
def recursiveFeatureSearch(model, startingFeatures): #returns bestAcc, bestFeatureSet, accAccumulator
    #powered by sklearn.feature_selection.RFE
    #we use a loop to iteratively reduce 'n_features_to_select' down to 2
    #we record the best found accuracy and its corresponding feature set out of all iterations

    XtrainStartingFeatures, XtestStartingFeatures, ytrainStartingFeatures, ytestStartingFeatures = preProcessingPipeline( startingFeatures )
    bestAcc = 0
    accAccumulator=[]
    for numOfFeatures in range(len(startingFeatures)-1,1,-1):
        selector = RFE(model, n_features_to_select=numOfFeatures)
        selector = selector.fit(XtrainStartingFeatures, ytrainStartingFeatures)
        mask = selector.support_
        reducedFeatureListIterator = itertools.compress(startingFeatures, mask)
        reducedFeatureList=[]
        for i in reducedFeatureListIterator: reducedFeatureList.append(i)
        #Pre-process only selected features from reducedFeatureList
        XtrainReduced, XtestReduced, ytrainReduced, ytestReduced = preProcessingPipeline( reducedFeatureList )
        acc = compute_performance_Array(RandomForestClassifier().fit(XtrainReduced,ytrainReduced).predict(XtestReduced), ytestReduced)
        accAccumulator.append(acc)
        if acc > bestAcc:
            bestAcc = acc
            bestFeatureSet = reducedFeatureList

    print("Best Found Accuracy: "  , bestAcc, "%", sep='')
    print("Best Found Features:\n" , bestFeatureSet)

    return bestAcc, bestFeatureSet, accAccumulator

##PIPES + FINAL PIPELINE
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
def preProcessingPipeline(selectedFeatures, splitTrainTest=True): #returns Xtrain, Xtest, ytrain, ytest. If splitTrainTest=False, returns X,y

    ## CREATE PIPES ##
    X_pipe = make_pipeline(X_Encoder( selectedFeatures ), StandardScaler()).fit(RawData[selectedFeatures])
    y_pipe = LabelEncoder().fit(RawData[targetVariable])

    if splitTrainTest:
        ## SPLIT TRAINING/TESTING PRE-PROCESSED DATA ##
        return train_test_split(
            X_pipe.transform(RawData[ selectedFeatures ]),
            y_pipe.transform(RawData[targetVariable]),
            test_size=0.25,
            random_state=0 )
    else:
        return X_pipe.transform(RawData[selectedFeatures]), y_pipe.transform(RawData[targetVariable])

##VISUALIZATIONS
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
def displayClassProportions(exportToCSV=False):
    #Displays class proportions, set 'exportToCSV' to True if you want to export them to a CSV file
    classProportions = pd.DataFrame(columns=['Counts','Proportions'], index = RawData[targetVariable].value_counts().index)
    classProportions['Counts'] = RawData[targetVariable].value_counts().values
    classProportions['Proportions'] = (round((RawData[targetVariable].value_counts() / 2111)*100,1)).values
    if exportToCSV:
        classProportions.to_csv('/Users/ryandivigalpitiya/Python Notebooks/CS 9637/Project/classProp.csv',index=False)
    classProportions.drop('Counts',axis=1).plot.bar()
    return classProportions

#EndOfImports+FunctionDefinitions----------------------------------------------------------------------------------------------------------------
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

## BASELINE TESTS ##
#############################################################################################################################################################

Xtrain, Xtest, ytrain, ytest = preProcessingPipeline( set(allFeatures)-set(['Weight','Height']) )

#Compute baseline performance (accuracy on test set) for each model type:
acc_rf = compute_performance_Array(  RandomForestClassifier(random_state=1) .fit(Xtrain,ytrain).predict(Xtest), ytest) #RandomForestClassifier
acc_lg = compute_performance_Array(  LogisticRegression(random_state=1)     .fit(Xtrain,ytrain).predict(Xtest), ytest) #LogisticRegressionClassifier
acc_nn = compute_performance_Array(  KNeighborsClassifier()                 .fit(Xtrain,ytrain).predict(Xtest), ytest) #KNeighborsClassifier
acc_gb = compute_performance_Array(  XGBClassifier(random_state=1)          .fit(Xtrain,ytrain).predict(Xtest), ytest) #XGBClassifier
acc_nb = compute_performance_Array(  GaussianNB()                           .fit(Xtrain,ytrain).predict(Xtest), ytest) #XGBClassifier

print("Random Forest Accuracy:"         , acc_rf)
print("Logistic Regression Accuracy:"   , acc_lg)
print("k-Nearest Neighbours Accuracy:"  , acc_nn)
print("XGBoost Accuracy:"               , acc_gb)
print("Naive Bayes Accuracy:"           , acc_nb)

#############################################################################################################################################################
## FEATURE TUNING ##
#############################################################################################################################################################

feature_results_rf = recursiveFeatureSearch( RandomForestClassifier(random_state=1), list(set(allFeatures) - set(['Weight','Height'])) )
feature_results_gb = recursiveFeatureSearch( XGBClassifier(random_state=1),          list(set(allFeatures) - set(['Weight','Height'])) )

#features dropped:
(set(allFeatures) - set(['Weight','Height'])) - set(feature_results_rf[1])
(set(allFeatures) - set(['Weight','Height'])) - set(feature_results_gb[1])

print("Random Forest Accuracy:" , feature_results_rf[0])
print("XGBoost Accuracy:"       , feature_results_gb[0])


#############################################################################################################################################################
## HYPER-PARAMTER TUNING ##
#############################################################################################################################################################

# Random Forest HP:
paramGrid_rf = ParameterGrid({
    'random_state'        :[1],
    'n_estimators'        :[100,500,1000,10000],
    'bootstrap'           :[True,False],
    'max_depth'           :[3,5,10,20,50,75,100,None],
    'min_samples_leaf'    :[1,2,4,10],
    'min_samples_split'   :[2,5,10]
    })

# XGBOOST HP:
paramGrid_XGB = ParameterGrid({
    'random_state'      :[1],
    'n_estimators'      :[600,10000],
    'colsample_bytree'  :[0.75,0.8,0.85],
    'max_depth'         :[6, 20, 50, 75],
    'reg_alpha'         :[1],
    'reg_lambda'        :[2,5,10],
    'subsample'         :[0.55, 0.6, .65],
    'learning_rate'     :[0.5],
    'gamma'             :[.5,1,2],
    'min_child_weight'  :[0.01],
    'sampling_method'   :['uniform'] })


hp_results_rf = hyperParameterSearch( RandomForestClassifier(random_state=1),                                            paramGrid_rf )
hp_results_gb = hyperParameterSearch( XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='mlogloss'),    paramGrid_XGB )

#Results:
#RF Hyper-Parameters:
# Best Found Accuracy: 86.17%
# Best Found Parameters: {'bootstrap': False, 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 10000, 'random_state': 1}

#############################################################################################################################################################
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#MODEL PERSISTANCE---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

filename = '/Users/ryandivigalpitiya/Virtual\ Envs/TensorFlow/My\ Python\ Files/ObesityModels/bestRFmodel.sav'
pickle.dump(  > model <  ,open(filename,'wb'))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## OPTIONAL: LOAD MODEL FROM DISK ##
################################################################################################################################
filename = '/Users/ryandivigalpitiya/Virtual\ Envs/TensorFlow/My\ Python\ Files/ObesityModels/bestRFmodel.sav'
loadedModel = pickle.load(open(filename,'rb'))
################################################################################################################################
# Or, re-train using tuned HP+Features:
tunedFeaturesRF = ['Gender','Age','family_history_with_overweight','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
paramGrid_rf = ParameterGrid({
    'random_state'        :[1],
    'n_estimators'        :[10000],
    'bootstrap'           :[False],
    'max_depth'           :[20],
    'min_samples_leaf'    :[1],
    'min_samples_split'   :[2]
    })
Xtrain,Xtest,ytrain,ytest = preProcessingPipeline(tunedFeaturesRF)

model_rf = RandomForestClassifier(random_state=1)
model_rf.set_params(**paramGrid_rf[0]) #set tuned parameters
acc = compute_performance_Array(model_rf.fit(Xtrain,ytrain).predict(Xtest), ytest)
print("RF Accuracy:",acc) #Should be 86%

#Save model to disk:
filename = '/Users/ryandivigalpitiya/Virtual\ Envs/TensorFlow/My\ Python\ Files/ObesityModels/bestRFmodel.sav'
pickle.dump( model_rf ,open(filename,'wb'))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## FEATURE IMPORTANCES PER MODEL ##

#RF
#trained model:
# showFeatureImportances(model_rf,tunedFeaturesRF)
#loaded model from disk:
showFeatureImportances(loadedModel,tunedFeaturesRF)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## GENERATING MULTI-CLASSIFICATION CONFUSION MATRIX ##

#generate test set, use test set to generate my Predicted column + Actual column:
Xtrain,Xtest,ytrain,ytest = preProcessingPipeline(tunedFeaturesRF)

predVSactual = np.zeros((len(ytest),2))
pred = model_rf.predict(Xtest)
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

## GENERATE MULTI-CLASS AUROC CURVES ##
## SKLearn suggests a technique called macro-averaging which we will do here
## It revolves around approaching TP and FP calculations for multiclassifcation with a binarized one-versus-all approach
## Macro-averaging is suggested if classes are balanced as they are in our dataset
# This code was taken and repurposed from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import interp

y_test_enc_df = pd.get_dummies(ytest,drop_first=False)
y_test_enc = y_test_enc_df.to_numpy()

y_score = model_rf.predict_proba(Xtest)
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
