# %%R
import pandas as pd
import numpy
import sys,os
# import ExponentialSmoothing as es
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import scipy.stats as st
import rpy2
import rpy2.robjects as r
import rpy2.robjects.numpy2ri
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
rpy2.robjects.numpy2ri.activate()
import DataHandler as dh
from cascade import *
sys.path.append(
	os.path.join(
		os.path.dirname(__file__), 
		'..', 
		'utils'
	)
)
from util import *
from Padronizar import *


class CascadeArima:
    def __init__(self,data,dimension,neurons,testNO, cascadeNumHiddenNodes):
        # print("rpy2.__path__")
        # print(rpy2.__path__)
	    # data = dados que serão trabahados
		# dimension = dimensão dos dados, (janela)
		# neurons = número de neurônios escondidos
		# testNO = número de instâncias de teste
        self.data=data
        self.dimension=dimension
        self.trainset=0.6
        self.valset=0.2
        self.neurons=neurons
        self.testNO=testNO
        self.cascadeNumHiddenNodes = cascadeNumHiddenNodes
    def start(self):
        dh2=dh.DataHandler(self.data,self.dimension,self.trainset,self.valset,self.testNO)
        # separando os dados em treino, teste e validação
        train_set, train_target, val_set, val_target, test_set, test_target, arima_train, arima_val, arima_test= dh2.redimensiondata(self.data,self.dimension,self.trainset,self.valset,self.testNO)
        
        # self.checkDatadivision(train_set, val_set, test_set, arima_train, arima_val, arima_test)
                        
        traindats=[]
        traindats.extend(arima_train)
        traindats.extend(arima_val)
        
        r.r('library(forecast)')
        arima = r.r('auto.arima') #instanciando um objeto arima
        arimaTest=r.r('Arima')
        ordem = r.r('c')
        #arima_train.extend(arima_val)
        numeric = r.r('as.numeric')
        fit = arima(numeric(arima_train))
        fitted = r.r('fitted') #extrai fitted values
        predTreino = fitted(fit) #extracting fitted values from arima_train
        fit2 = arimaTest(numeric(arima_val),model=fit)
        fit3 = arimaTest(numeric(arima_test), model=fit)
        predVal = fitted(fit2) # previsão validação
        predTest = fitted(fit3) # previsão de teste

        predTudo=[]

        predTudo.extend(predTreino)
        predTudo.extend(predVal)
       
        residualTreino=numpy.array(arima_train)-(predTreino)
        predTudo.extend(predTest)
        residual=self.data-predTudo

        residualNorm= (residual-min(residualTreino))/(max(residualTreino)-min(residualTreino))

        train_set2, train_target2, val_set2, val_target2, test_set2, test_target2, arima_train2, arima_val2, arima_test2 = dh2.redimensiondata(
        residualNorm, self.dimension, self.trainset, self.valset,self.testNO)
        train_set2.extend(val_set2)
        train_target2.extend(val_target2)
        
        num_hidden_nodes = self.cascadeNumHiddenNodes
        cascade: Cascade = Cascade(num_hidden_nodes)
        cascade.X_val, cascade.y_val = val_set, val_target
        cascade.fit(train_set2,train_target2)
           
        predRNA = cascade.predict(test_set)
        predRNAVal = cascade.predict(val_set)
        # print(predRNA)
        # predRNA = np.array(predRNA)[:,0]
        
        predRNAD=predRNA*(max(residualTreino)-min(residualTreino))+min(residualTreino)
        predRNADVal=predRNAVal*(max(residualTreino)-min(residualTreino))+min(residualTreino)

        predFinal=numpy.asarray(predTest)+numpy.asarray(predRNAD)
        predFinalVal=numpy.asarray(predVal)+numpy.asarray(predRNADVal)
        # predFinal=numpy.asarray(predTest)

        predFinalN=(numpy.asarray(predFinal)-min(traindats))/(max(traindats)-min(traindats))
        predFinalNVal=(numpy.asarray(predFinalVal)-min(traindats))/(max(traindats)-min(traindats))
        testTarget=(numpy.asarray(arima_test)-min(traindats))/(max(traindats)-min(traindats))
        valTarget=(numpy.asarray(arima_val)-min(traindats))/(max(traindats)-min(traindats))
        
        # predFinalN=(numpy.asarray(predFinal))
        # predFinalNVal=(numpy.asarray(predFinalVal))
        
        # testTarget=(numpy.asarray(arima_test))
        # valTarget=(numpy.asarray(arima_val))
        # print("predFinalN")
        # print(predFinalN)
        # print("testTarget")
        # print(testTarget)
        # testTarget = desnormalizar(targetTestScaler,numpy.array(testTarget).reshape(-1, 1))
        # predFinalN = desnormalizar(targetTestScaler,numpy.array(predFinalN).reshape(-1, 1))
                
        mape, mse, rmse = calculateResidualError(testTarget, predFinalN)                
        mapeVal, mseVal, rmseVal = calculateResidualError(valTarget, predFinalNVal)                
        
        return mape, mse, rmse, predFinalN,mapeVal, mseVal, rmseVal, cascade.optimalNumHiddenNodes
    
    def checkDatadivision(self,train_set, val_set, test_set, arima_train, arima_val, arima_test):
        print("numpy.array(train_set).shape")                
        print(numpy.array(train_set).shape)
                        
        print("numpy.array(val_set).shape")                
        print(numpy.array(val_set).shape) 
                       
        print("numpy.array(test_set).shape")
        print(numpy.array(test_set).shape)
        
        print("numpy.array(arima_train).shape")                
        print(numpy.array(arima_train).shape)
                        
        print("numpy.array(arima_val).shape")                
        print(numpy.array(arima_val).shape) 
                       
        print("numpy.array(arima_test).shape")
        print(numpy.array(arima_test).shape)
        sys.exit(-1)            


