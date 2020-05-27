import sys
import os
import numpy as np
from cascade import *
from elm import *
import pandas as pd
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

def executeCascade(bases, dimensions, maxHiddenNodes, minHiddenNodes, iterations, model):
    

    for base, dimension in zip(bases, dimensions):
        mseValArray = []
        mseTestArray = []
        dictToSave = {}
        
        print(base)
        data = pd.read_csv('../data/'+base+'.txt', header=None)
        dataSNorm = np.array(data).copy().T[0]
        dataNorm, listMin, listMax = pad.normalizarLinear(data, 0.1, 0.9)
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
                # print("numHiddenNodes: " + str(numHiddenNodes))                

                mseTestListCascade = []
                mapeTestListCascade = []
                mseValListCascade = []
                mapeValListCascade = []
                
                for i in list(range(1, iterations+1)):
                    # print("Iteration: " + str(i))
                    cascadeArima: CascadeArima = CascadeArima(dataBase, dimension, 10, round(
                        len(dataBase)-len(dataBase)*0.8), numHiddenNodes)
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
             
            writeJsonFile(dictToSave, base)                       
                    
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

if __name__ == '__main__':
    bases = ["airlines2", "Monthly Sunspot Dataset", "Minimum Daily Temperatures Dataset", "Daily Female Births Dataset",'Colorado River','Eletric','Gas','Lake Erie','Pollution','redwine']
    dimensions = [12,11,12,12,12,12,12,12,12,12]
    # bases = ['Pollution']
    # dimensions = [12]
    maxHiddenNodes = 70
    minHiddenNodes = 1
    iterations = 1    
    model = "Arima Cascade"
    saveChart = False
    showChart = True
    chart:Chart = Chart()
    executeCascade(bases, dimensions, maxHiddenNodes, minHiddenNodes, iterations, model)
    
    base = 'Pollution'
    loadedDict = readJsonFile('../data/simulations/2020-05-27/'+base+'.json')
    mapeValArray, mseValArray, mapeTestArray, mseTestArray = getErrors(loadedDict)
    predVal, trueVal, predTest, trueTest = getPredAndTrueValues(loadedDict,2)
    chart.plotValidationAndTest(base, "Arima Cascade", maxHiddenNodes, mseValArray,
                              mseTestArray, "Validation", "Test", showChart, saveChart)

    
    