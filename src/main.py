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
    
    # bases = ["airlines2"]
    # dimensions = [12]
    for base, dimension in zip(bases, dimensions):
        print(base)
        mseByNumHiddenNodesList = []
        mseValByNumHiddenNodesList = []
        for numHiddenNodes  in list(range(1,71)):
            print("numHiddenNodes: "+  str(numHiddenNodes))
            data = pd.read_csv('../data/'+base+'.txt', header=None)
            dataNorm,listMin,listMax = pad.normalizarLinear(data,0.1,0.9)
            dataNorm = np.array(dataNorm).T[0]
            
            mapeListCascade = []
            mseListCascade = []
            mseValListCascade = []
            for i in list(range(1,2)):
                # print(str(i) + ' '+ base)
                cascadeArima: CascadeArima = CascadeArima(dataNorm,dimension,10,round(len(data)-len(data)*0.8),numHiddenNodes)
                mape, mse, rmse, predFinalN,mapeVal, mseVal, rmseVal, optimalNumHiddenNodes = cascadeArima.start()
                mapeListCascade.append(mape)
                mseListCascade.append(mse)              
                mseValListCascade.append(mseVal)              
                        
            # print(np.mean(mapeListCascade))
            # print(np.mean(mseListCascade))
            print("Optimal number of Hidden Nodes "+str(optimalNumHiddenNodes))
            print()
            mseByNumHiddenNodesList.append(np.mean(mseListCascade))
            mseValByNumHiddenNodesList.append(np.mean(mseValListCascade))
        
        # print(mseByNumHiddenNodesList) 
        plot(base,"Arima Cascade Normalizado",70,mseByNumHiddenNodesList,mseValByNumHiddenNodesList,[],"Validation","Test","",True, False)   
            
        
           
    	