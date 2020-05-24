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

if __name__ == '__main__':

    # bases = ["airlines2", "Monthly Sunspot Dataset", "Minimum Daily Temperatures Dataset", "Daily Female Births Dataset",'Colorado River','Eletric','Gas','Lake Erie','Pollution','redwine']
    # dimensions = [12,11,12,12,12,12,12,12,12,12]
    bases = ['Pollution']
    dimensions = [12]

    maxHiddenNodes = 10
    minHiddenNodes = 1
    iterations = 1
    saveChart = False
    showChart = True
    chart:Chart = Chart()

    for base, dimension in zip(bases, dimensions):
        mseValArray = []
        mseTestArray = []
        print(base)
        data = pd.read_csv('../data/'+base+'.txt', header=None)
        dataSNorm = np.array(data).copy().T[0]
        dataNorm, listMin, listMax = pad.normalizarLinear(data, 0.1, 0.9)
        dataNorm = np.array(dataNorm).T[0]

        dataBases = [dataNorm, dataSNorm]
        for dataBase in dataBases:
            mseTestByNumHiddenNodesList = []
            mseValByNumHiddenNodesList = []
            for numHiddenNodes in list(range(minHiddenNodes, maxHiddenNodes+1)):
                print("numHiddenNodes: " + str(numHiddenNodes))

                # mapeListCascade = []

                mseTestListCascade = []
                mseValListCascade = []
                for i in list(range(1, iterations+1)):
                    print("Iteration: " + str(i))
                    cascadeArima: CascadeArima = CascadeArima(dataBase, dimension, 10, round(
                        len(dataBase)-len(dataBase)*0.8), numHiddenNodes)
                    mape, mse, rmse, mapeVal, mseVal, rmseVal, optimalNumHiddenNodes, pred, target = cascadeArima.start()
                    # mapeListCascade.append(mape)
                    mseTestListCascade.append(mse)
                    mseValListCascade.append(mseVal)

                # print(np.mean(mapeListCascade))
                # print(np.mean(mseListCascade))
                print("Optimal number of Hidden Nodes: " +
                      str(optimalNumHiddenNodes))
                print()
                mseTestByNumHiddenNodesList.append(np.mean(mseTestListCascade))
                mseValByNumHiddenNodesList.append(np.mean(mseValListCascade))

            mseValArray.append(mseTestByNumHiddenNodesList)
            # mseTest.append(mseTestByNumHiddenNodesList)

            mseTestArray.append(mseValByNumHiddenNodesList)
            # mseVal.append(mseValByNumHiddenNodesList)
        # print(mseByNumHiddenNodesList)
        chart.plotValidationAndTest(base, "Arima Cascade", maxHiddenNodes, mseValArray,
                              mseTestArray, "Validation", "Test", showChart, saveChart)
