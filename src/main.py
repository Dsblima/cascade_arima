import sys, os
import numpy as np
from cascade import *
from elm import *
import pandas as pd
from activation_functions import *
from cascadeArima import * 
from sklearn.preprocessing import MinMaxScaler
sys.path.append(
	os.path.join(
		os.path.dirname(__file__), 
		'..', 
		'utils'
	)
)
from util import *
from cascade import *
import Padronizar as pad

if __name__ == '__main__':    
    
    bases = ["airlines2", "Monthly Sunspot Dataset", "Minimum Daily Temperatures Dataset", "Daily Female Births Dataset",'Colorado River','Eletric','Gas','Lake Erie','Pollution','redwine']
    dimensions = [12,11,12,12,12,12,12,12,12,12]
    maxHiddenNodes = 70
    iterations = 30
    
    for base, dimension in zip(bases, dimensions):
        mseValArray = []
        mseTestArray = []
        print(base)
        data = pd.read_csv('../data/'+base+'.txt', header=None)
        dataSNorm = np.array(data).copy().T[0]
        dataNorm,listMin,listMax = pad.normalizarLinear(data,0.1,0.9)
        dataNorm = np.array(dataNorm).T[0]
                
        dataBases = [dataNorm,dataSNorm]
        for dataBase in dataBases:
            mseTestByNumHiddenNodesList = []
            mseValByNumHiddenNodesList = []
            for numHiddenNodes  in list(range(1,maxHiddenNodes+1)):
                print("numHiddenNodes: "+  str(numHiddenNodes))
                
                # mapeListCascade = []                
                
                mseTestListCascade = []
                mseValListCascade = []            
                for i in list(range(1,iterations+1)):
                    # print(str(i) + ' '+ base)
                    cascadeArima: CascadeArima = CascadeArima(dataBase,dimension,10,round(len(dataBase)-len(dataBase)*0.8),numHiddenNodes)
                    mape, mse, rmse, predFinalN,mapeVal, mseVal, rmseVal, optimalNumHiddenNodes = cascadeArima.start()
                    # mapeListCascade.append(mape)
                    mseTestListCascade.append(mse)              
                    mseValListCascade.append(mseVal)              
                            
                # print(np.mean(mapeListCascade))
                # print(np.mean(mseListCascade))
                print("Optimal number of Hidden Nodes "+str(optimalNumHiddenNodes))
                print()
                mseTestByNumHiddenNodesList.append(np.mean(mseTestListCascade))
                mseValByNumHiddenNodesList.append(np.mean(mseValListCascade))
            
            
            mseValArray.append(mseTestByNumHiddenNodesList)
            # mseTest.append(mseTestByNumHiddenNodesList)
            
            mseTestArray.append(mseValByNumHiddenNodesList)
            # mseVal.append(mseValByNumHiddenNodesList)
        # print(mseByNumHiddenNodesList) 
        plot(base,"Arima Cascade",maxHiddenNodes,mseValArray,mseTestArray,[],"Validation","Test","",False, True)   
            
        
           
    	