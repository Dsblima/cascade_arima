import sys, os
import numpy as np
import pandas as pd
from elm import *
import pickle
sys.path.append(
	os.path.join(
		os.path.dirname(__file__), 
		'..', 
		'utils'
	)
)
from activation_functions import *
from util import *
from DataHandler import *
import Padronizar
from ErroPercentualAbsoluto import *
import Arquivo

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing   import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn                 import preprocessing

class Model(object):
  def __init__(self,wi,wo):
    self.wi = wi
    self.wh = wo

class Cascade(object):

  def __init__(self, numMaxHiddenNodes = 800000):
    self.numMaxHiddenNodes = numMaxHiddenNodes
    self.ensemble = {}
    self.weightsArray = {}
    
    # ERROR ARRAYS
    self.residualError = []
    self.residualErrorTest = []
    self.mapeArray = []
    self.mapeArrayTest = []

    self.X_train=[]
    self.X_test=[]
    self.y_train=[]
    self.y_test=[]
    self.menor = 100000000
    
    
  def load_and_preprocess_data(self):
    # TESTE COM A BASE airlines 2
    data = Arquivo.ler('../data/airlines2.txt')
    dh:DataHandler = DataHandler(data, 12, 60, 20,20)
    self.X_train, self.y_train, val_set, val_target, self.X_test, self.y_test, arima_train, arima_val, arima_test= dh.redimensiondata(data, 12, 60, 20,20)

    y = [[]]
    x = np.concatenate( (self.X_train,self.X_test) )   

    y = np.matrix( np.concatenate( (np.array(self.y_train),np.array(self.y_test)) ))   
    data = np.concatenate((x, y.T), axis=1)
    x, y = Padronizar.dividir(data, 12, 1)


    # data = Arquivo.ler('../data/projeto base completa.csv')
    # x, y = Padronizar.dividir(data, 4, 1)
    # print("y")
    # print(y[y.columns[0]])
    # minmaxscaler = MinMaxScaler(feature_range=(0,1))
    self.dataNX, self.listMin,  self.listMax  = Padronizar.normalizarLinear(x, 0.1, 0.9)
    self.dataNY, self.listMinY, self.listMaxY = Padronizar.normalizarLinear(y, 0.1, 0.9)
    # self.scalerX,self.scalerY, self.dataNormalizadoX, self.dataNormalizadoY = Padronizar.normalizar(x,y)
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataNX, self.dataNY, train_size = 0.8, test_size  = 0.2)
    self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(self.X_test, self.y_test, train_size = 0.5, test_size  = 0.5)
    
    self.X_train = addBias(self.X_train.values)    
    self.X_val  = addBias(self.X_val.values)
    self.X_test  = addBias(self.X_test.values)
    
    # WE WILL USE ONLY THE FIRST COLUMN 														
    self.y_train = self.y_train[self.y_train.columns[0]].values
    self.y_test = self.y_test[self.y_test.columns[0]].values
    self.y_val = self.y_val[self.y_val.columns[0]].values

  def init_weights(self,xcol):    
    return (np.random.rand(xcol,1))

  def insertHiddenUnit(self,i):
    numCol = self.X_train[0].__len__()+i
    
    wh = self.init_weights(numCol)

    self.weightsArray[i] = wh
    
    return self.forward(self.weightsArray,self.X_train)

  def forward(self,wh,input):
    netis = [[]]    
    for node, weights in wh.items():
      if node == 0:
        
        ent = np.dot( input,weights)
        neti = sigmoid(ent)      
        netis = neti
      else:
        
        input = np.concatenate((input,neti),axis=1)
        ent = np.dot( input,weights)
        neti = sigmoid(ent)      
        netis = np.concatenate((netis,neti), axis=1)            
    netis = addBias(np.matrix(netis))
    
    return netis

  def calcPred(self,w0,neti):
    return neti.dot(w0.T)

  def calculateResidualError(self,trueValues,pred):    
    # print(np.matrix(trueValues).T)
    # trueValues = Padronizar.desnormalizarLinear(pd.DataFrame(np.matrix(trueValues).T), self.listMaxY, self.listMinY, 0.1, 0.9)
    # print(trueValues)
    # pred = Padronizar.desnormalizarLinear(pd.DataFrame(np.matrix(pred)), self.listMaxY, self.listMinY, 0.1, 0.9)
    # print(pred)
    # print("Residual Error")
    mape = mean_absolute_percentage_error(trueValues,pred)
    
    mse = mean_squared_error(trueValues,pred)
    
    return mse,mape
  
  def saveModel(self,model,position):
    self.ensemble[position] = model
    
  def fit(self, train_set, train_target):
    print("fit")
    neti = np.zeros((self.numMaxHiddenNodes,1))
    for i in list(range(self.numMaxHiddenNodes)):
        print("Node: ",i+1)
        neti= self.insertHiddenUnit(i)
        elm = ELM(1,self.X_train,self.y_train)
        netInv = elm.pinv(neti)
        w0 = elm.getW0(netInv)
        model:Model = Model(self.weightsArray.copy(),w0)
        self.saveModel(model,i)
        
        predTraining = self.calcPred(w0,neti)
        
        print("Validation")
        netiVal = self.forward(self.weightsArray,self.X_val)
        predVal = self.calcPred(w0,netiVal)
        
        mse,mape = self.calculateResidualError(self.y_val, predVal)
        self.mapeArray.append(mape)
        self.residualError.append(mse)
        
        printErrors(mape,mse)
    
    
  def predict(self,input, target):
    print("predict")  
    for i, model in self.ensemble.items():
       
      netiTeste = self.forward(model.wi,input)
      predTeste = self.calcPred(model.wh,netiTeste)
      mse,mape  = self.calculateResidualError(target, predTeste)
      self.mapeArrayTest.append(mape)
      self.residualErrorTest.append(mse)
      printErrors(mape,mse)
      
    return predTeste