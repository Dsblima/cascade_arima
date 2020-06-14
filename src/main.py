import sys
import os
import numpy as np
from cascade import *
from elm import *
import pandas as pd
from datetime import date
from cascadeArima import *
from sklearn.preprocessing import MinMaxScaler
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'utils'
    )
)
import Padronizar as pad
from Chart import *
from util import *
from activation_functions import *
from JsonManager import *

def executeCascade(today,bases,upperLimit, lowerLimit, dimensions, maxHiddenNodes, minHiddenNodes, iterations, model):
    
    lambdaValues = [1, 10,100,1000, 10000, 100000]
    for base, dimension in zip(bases, dimensions):
        for lambdaValue in lambdaValues:
            mseValArray = []
            mseTestArray = []
            dictToSave = {}
            
            print(base)
            data = pd.read_csv('../data/'+base+'.txt', header=None)
            dataSNorm = np.array(data).copy().T[0]
            dataNorm, listMin, listMax = pad.normalizarLinear(data, lowerLimit,upperLimit)
            dataNorm = np.array(dataNorm).T[0]

            dataBases = [dataNorm]
            for dataBase in dataBases:
                
                mseTestByNumHiddenNodesList = []
                mapeTestByNumHiddenNodesList = []
                mseValByNumHiddenNodesList = []
                mapeValByNumHiddenNodesList = []
                predValList = []
                targetValList = []
                predTestList = []
                targetTestList = []
                listNodes = list(range(minHiddenNodes, maxHiddenNodes+1))
                
                for numHiddenNodes in listNodes:
                    print (base+' - '+str(numHiddenNodes))                

                    mseTestListCascade = []
                    mapeTestListCascade = []
                    mseValListCascade = []
                    mapeValListCascade = []
                    
                    for i in list(range(1, iterations+1)):
                        # print("Iteration: " + str(i))
                        cascadeArima: CascadeArima = CascadeArima(dataBase, dimension, 10, round(
                            len(dataBase)-len(dataBase)*0.8), numHiddenNodes, lambdaValue)
                        mapeTest, mseTest, rmseTest, mapeVal, mseVal, rmseVal, optimalNumHiddenNodes, targetVal, predVal,  targetTest, predTest = cascadeArima.start()
                        
                        mapeTestListCascade.append(mapeTest)
                        mseTestListCascade.append(mseTest)
                        mapeValListCascade.append(mapeVal)
                        mseValListCascade.append(mseVal)
                        
                        predValList.append(predVal)
                        targetValList.append(targetVal)
                        predTestList.append(predTest)
                        targetTestList.append(targetTest)
                    
                    # print("Optimal number of Hidden Nodes: " +
                    #       str(optimalNumHiddenNodes))
                    # print()
                    mseTestByNumHiddenNodesList.append(np.mean(mseTestListCascade))
                    mapeTestByNumHiddenNodesList.append(np.mean(mapeTestListCascade))
                    mseValByNumHiddenNodesList.append(np.mean(mseValListCascade))
                    mapeValByNumHiddenNodesList.append(np.mean(mapeValListCascade))
                
                dictToSave['model'] = model
                dictToSave['activationFunction'] = 'sigmoid'
                dictToSave['inputsize']  = dimension
                dictToSave['executions'] = []
                
                for mapeValValue, mseValValue, mapeTestValue, mseTestValue, valPredValues, valTargetValues, testPredValues, testTargetValues, numHiddenNodes in zip( mapeValByNumHiddenNodesList, mseValByNumHiddenNodesList,mapeTestByNumHiddenNodesList, mseTestByNumHiddenNodesList,  predValList, targetValList, predTestList, targetTestList, listNodes ):
                
                    dictToSave['executions'].append(
                        {
                            "numHiddenNodes":numHiddenNodes,
                            "predVal":valPredValues.tolist(),
                            "trueVal":valTargetValues.tolist(),
                            "predTest":testPredValues.tolist(),
                            "trueTest":testTargetValues.tolist(),
                            "errors":[
                                {
                                    "mapeVal":mapeValValue,
                                    "mseVal":mseValValue,
                                    "rmseVal":"0",
                                    "mapeTest":mapeTestValue,
                                    "mseTest":mseTestValue,
                                    "rmseTest":"0"
                                }
                            ]
                        }
                    )
                
                writeJsonFile(dictToSave, base, today+" lambda = "+str(lambdaValue))                       
                    
# Go throughout a dict, get the errors and returns arrays with them
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

def visualizeResults(bases, dirs, titles, model, saveChart, showChart, folderToSave):
    
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
            
        chart.plotValidationAndTest(base, model, maxHiddenNodes, mseVal,
                                mseTest, "Validation", "Test", title1, title2, showChart, saveChart, folderToSave) 
    # chart.plotTable(mseValArray,filename+'MSEVal.csv')

if __name__ == '__main__':
    bases = ["airlines2", "Daily Female Births Dataset",'Colorado River','Eletric','Gas','Lake Erie','Pollution','redwine', "Monthly Sunspot Dataset", "Minimum Daily Temperatures Dataset"]
    dimensions = [12,12,12,12,12,12,12,12,11,12]
    # bases = ['Minimum Daily Temperatures Dataset']
    # dimensions = [12]
    upperLimit = 1
    lowerLimit = -1
    maxHiddenNodes = 70
    minHiddenNodes = 1
    iterations = 1    
    today = str(date.today())            
    # executeCascade(today, bases, upperLimit, lowerLimit, dimensions, maxHiddenNodes, minHiddenNodes, iterations, model)    
    
    model = "Cascade - ARIMA"
    saveChart = True
    showChart = False   
    dir1 = '../data/simulations/2020-06-12 lambda = 10000/'
    dir2 = '../data/simulations/2020-06-12N regularization/'
    dirs = [dir1, dir2]
    title1 = " with lambda = 10000" 
    title2 = " without regularization"
    folderToSave = "reg x no reg/"
    titles = [title1, title2]
    visualizeResults(bases, dirs, titles, model, saveChart, showChart, folderToSave)
    

    
    