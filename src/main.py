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

def executeCascade():
    bases = ["airlines2", "Monthly Sunspot Dataset", "Minimum Daily Temperatures Dataset", "Daily Female Births Dataset",'Colorado River','Eletric','Gas','Lake Erie','Pollution','redwine']
    dimensions = [12,11,12,12,12,12,12,12,12,12]
    bases = ['Pollution']
    dimensions = [12]

    maxHiddenNodes = 10
    minHiddenNodes = 1
    iterations = 1
    saveChart = False
    showChart = True
    model = "Arima Cascade"
    chart:Chart = Chart()

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
            
            mseValArray.append(mseTestByNumHiddenNodesList)            

            mseTestArray.append(mseValByNumHiddenNodesList)
            
        # chart.plotValidationAndTest(base, "Arima Cascade", maxHiddenNodes, mseValArray,
        #                       mseTestArray, "Validation", "Test", showChart, saveChart)

# def 
if __name__ == '__main__':

    executeCascade()
    # loadedDict = readJsonFile('../data/simulations/2020-05-25/Pollution.json')
    # for execution in loadedDict['executions']:
    #     print('Num hidden '+ str(execution['numHiddenNodes']))
    #     for key in execution['errors']:
    #         print(key['mse'])
    
    