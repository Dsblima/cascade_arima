# %%R
import pandas as pd
import numpy
import sys
# import ExponentialSmoothing as es
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import scipy.stats as st
import rpy2
import rpy2.robjects as r
import rpy2.robjects.numpy2ri
from sklearn.metrics import mean_squared_error
rpy2.robjects.numpy2ri.activate()
import DataHandler as dh
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV



class Zhang:
    def __init__(self,data,dimension,neurons,testNO):
        # print("rpy2.__path__")
        # print(rpy2.__path__)
	    # data = dados que serão trabahados
		# dimension = dimensão dos dados, (janela)
		# neurons = número de neurônios escondidos
		# testNO = número de instâncias de teste
        self.data=data
        self.dimension=dimension
        self.trainset=0.6
        self.valset=0.4
        self.neurons=neurons
        self.testNO=testNO
    def start(self):
        dh2=dh.DataHandler(self.data,self.dimension,self.trainset,self.valset,self.testNO)
        # separando os dados em treino, teste e validação
        train_set, train_target, val_set, val_target, test_set, test_target, arima_train, arima_val, arima_test= dh2.redimensiondata(self.data,self.dimension,self.trainset,self.valset,self.testNO)
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
       # target=[]
        #target.extend(arima_train)
        #arget.extend(arima_val)
        residualTreino=numpy.array(arima_train)-(predTreino)
        predTudo.extend(predTest)
        residual=self.data-predTudo

        residualNorm= (residual-min(residualTreino))/(max(residualTreino)-min(residualTreino))

        train_set2, train_target2, val_set2, val_target2, test_set2, test_target2, arima_train2, arima_val2, arima_test2 = dh2.redimensiondata(
            residualNorm, self.dimension, self.trainset, self.valset,self.testNO)
        train_set2.extend(val_set2)
        train_target2.extend(val_target2)
        nn1 = MLPRegressor(activation='tanh', solver='lbfgs', shuffle=False,max_iter=5000, learning_rate_init=0.05)
        rna = GridSearchCV(nn1, param_grid={
            'hidden_layer_sizes': [(2,), (5,), (10,), (15,), (20,)]})
        rna.fit(train_set2,train_target2)

        ranks=(rna.cv_results_['rank_test_score'])
        index = numpy.argmin(ranks)
        print("index")
        print(index)
        neuronlist = [2,5,10,15,20]
#        print('Number of hidden neurons %d'%(neuronlist[index]))
        predRNA=rna.predict(test_set2)

        predRNAD=predRNA*(max(residualTreino)-min(residualTreino))+min(residualTreino)

        predFinal=numpy.asarray(predTest)+numpy.asarray(predRNAD)

        predFinalN=(numpy.asarray(predFinal)-min(traindats))/(max(traindats)-min(traindats))

        testTarget=(numpy.asarray(arima_test)-min(traindats))/(max(traindats)-min(traindats))
        mse=mean_squared_error(testTarget,predFinalN)
        # print("neurolist")
        return neuronlist[index],mse,predFinalN


