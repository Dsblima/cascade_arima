import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from sklearn.metrics import mean_squared_error
import File
from DataHandler import *
import Padronizar
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from math import sqrt
from JsonManager import *
from Chart import *
def add_1(arr):
    return np.append(arr,1)
  
def addBias(matrix):#this function adds the value 1, at the last position in each array from matrix
    
  bias = np.ones((matrix.__len__(),1))
  # return np.array(list(map(add_1, matrix)))
  return np.concatenate((matrix,bias),axis=1)

def roundNumber(number):
  return round(number,4)

def printErrors(mape,mse):
    print("mean_absolute_percentage_error")
    print(mape)
    print("mean_squared_error")
    print(mse)
    print()
    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
  
def load_and_preprocess_data(baseName,dimension, lowerLimit, upperLimit):
    
    data = File.ler('../data/'+baseName+'.txt')
    dh:DataHandler = DataHandler(data, dimension, 60, 20,20)
    X_train, y_train, val_set, val_target, X_test, y_test, arima_train, arima_val, arima_test= dh.redimensiondata(data, dimension, 60, 20,20)

    y = [[]]
    x = np.concatenate( (X_train,X_test) )   

    y = np.matrix( np.concatenate( (np.array(y_train),np.array(y_test)) ))   
    data = np.concatenate((x, y.T), axis=1)
    x, y = Padronizar.dividir(data, dimension, 1)


    # data = Arquivo.ler('../data/projeto base completa.csv')
    # x, y = Padronizar.dividir(data, 4, 1)
    # print("y")
    # print(y[y.columns[0]])
    # minmaxscaler = MinMaxScaler(feature_range=(0,1))
    # dataNX, listMin,  listMax  = Padronizar.normalizarLinear(x, lowerLimit, upperLimit)
    # dataNY, listMinY, listMaxY = Padronizar.normalizarLinear(y, lowerLimit, upperLimit)
    # scalerX,scalerY, dataNormalizadoX, dataNormalizadoY = Padronizar.normalizar(x,y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.6, test_size  = 0.4)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size = 0.5, test_size  = 0.5)
    
    X_train = addBias(X_train.values)    
    X_val  = addBias(X_val.values)
    X_test  = addBias(X_test.values)
    
    # WE WILL USE ONLY THE FIRST COLUMN 														
    y_train = y_train[y_train.columns[0]].values
    y_test = y_test[y_test.columns[0]].values
    y_val = y_val[y_val.columns[0]].values
    
    return   X_train, y_train, X_val, y_val, X_test, y_test

def calculateResidualError(trueValues,pred):    
    # print(np.matrix(trueValues).T)
    # trueValues = Padronizar.desnormalizarLinear(pd.DataFrame(np.matrix(trueValues).T), self.listMaxY, self.listMinY, 0.1, 0.9)
    # print(trueValues)
    # pred = Padronizar.desnormalizarLinear(pd.DataFrame(np.matrix(pred)), self.listMaxY, self.listMinY, 0.1, 0.9)
    # print(pred)
    # print("Residual Error")
    mape = mean_absolute_percentage_error(trueValues,pred)
    
    mse = mean_squared_error(trueValues,pred)
    
    rmse = sqrt(mse)
    
    return mape, mse, rmse
def getErrors(dictToRead):
    
    mapeValArray = []
    mseValArray = []
    mapeTestArray = []
    mseTestArray = []
    for execution in dictToRead['executions']:
        
        for error in execution['errors']:
            mapeValArray.append(error['mapeVal'])
            mseValArray.append(error['mseVal'])
            mapeTestArray.append(error['mapeTest'])
            mseTestArray.append(error['mseTest'])
    
    return mapeValArray, mseValArray, mapeTestArray, mseTestArray

def getSmallestError(errorList):
    smallestError = 10000000
    numHiddenNode = 1
    numHiddenNodeSmallestError = 1
    
    for error in errorList:
        if error < smallestError:
            smallestError = error
            numHiddenNodeSmallestError = numHiddenNode
        numHiddenNode += 1
    return smallestError, numHiddenNodeSmallestError

def getNumHiddenNodesSmallestError(bases,folder, model):
    errorsDict = []
    mseVal = []
    mseTest = []
    for base in bases:
        
        print(base)
        fileName = folder+base        
        loadedDict = readJsonFile(fileName+'.json')        
        mapeValArray, mseValArray, mapeTestArray, mseTestArray = getErrors(loadedDict)
        smallestErrorVal, numHiddenNodeSmallestErrorVal = getSmallestError(mseValArray)
        smallestErrorTest, numHiddenNodeSmallestErrorTest = getSmallestError(mseTestArray)
        val = str(round(smallestErrorVal,5))+'-'+str(numHiddenNodeSmallestErrorVal)
        test = str(round(smallestErrorTest,5))+'-'+str(numHiddenNodeSmallestErrorTest)        
        mseVal.append(val)
        mseTest.append(test)
        print("mseVal")
        print(val)        
        print("mseTest")
        print(test)        
        print()        
    df = pd.DataFrame(data={"Database": bases, "MSE Val": mseVal, "MSE Test":mseTest})
    df.to_csv(folder+model+'_smallestErrors.csv', sep=',',index=False)    
        
def getPredAndTrueValues(dictToRead, node):
    predVal = []
    trueVal = []
    predTest = []
    trueTest = []
    for execution in dictToRead['executions']:
        if execution['numHiddenNodes'] == node:
            predVal = execution['predVal']
            trueVal = execution['trueVal']
            predTest = execution['predTest']
            trueTest = execution['trueTest']

    return predVal, trueVal, predTest, trueTest

def visualizeResults(bases, dirs, titles, model, saveChart, showChart, folderToSave, maxHiddenNodes):    
    for base in bases:
        mseVal = []
        mseTest = []
        chart:Chart = Chart()
        
        for folder, title in zip(dirs, titles):
        
            fileName = folder+base
        
            loadedDict = readJsonFile(fileName+'.json')    
            
            mapeValArray, mseValArray, mapeTestArray, mseTestArray = getErrors(loadedDict)
            mseVal.append(mseValArray)
            mseTest.append(mseTestArray)
            predVal, trueVal, predTest, trueTest = getPredAndTrueValues(loadedDict,2)
            
        chart.plotValidationAndTest(base, model, maxHiddenNodes, mseVal, mseTest, "Validation", "Test", titles, showChart, saveChart, folderToSave)
    