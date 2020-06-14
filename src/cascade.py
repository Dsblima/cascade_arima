import sys, os
import numpy as np
import pandas as pd
import math
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

  def __init__(self, numMaxHiddenNodes = 800000,lambdaReg = 100000 ):
    self.numMaxHiddenNodes = numMaxHiddenNodes
    self.optimalNumHiddenNodes = numMaxHiddenNodes
    self.lambdaReg = lambdaReg
    self.ensemble = {}
    self.weightsArray = {}
    self.candidatesWeightsArray = {}
    
    # ERROR ARRAYS
    self.mseArray = []
    self.mseArrayTest = []
    self.mapeArray = []
    self.mapeArrayTest = []

    self.X_train=[]
    self.X_test=[]
    self.y_train=[]
    self.y_test=[]
    self.minimumError = math.inf
    self.optimalWi = []    
    self.optimalWo = []    
      
  def init_weights(self,xcol):    
    return np.random.rand(xcol,1)

  def insertHiddenUnit(self,i):
    numCol = self.X_train[0].__len__()+i
    
    # self.generateCandidates(50,numCol)
    
    wh = self.init_weights(numCol)

    self.weightsArray[i] = wh
    
    return self.forward(self.weightsArray,self.X_train)

  def generateCandidates(self,numCandidates,numColumns):
    for candidate in list(range(numCandidates)):
      self.candidatesWeightsArray[candidate] = self.init_weights(numCol)
  
  def selectBestCandidate(self):
    return []
    
  def calculateCorrelation(self):
    return 0
  
  def regularization(self, neti, y):
       netiT = neti.T
       prod = neti.dot(netiT)
       inv = np.linalg.pinv(1/self.lambdaReg + prod)
       prodInv = netiT.dot(inv) 
       return prodInv.dot(y)
    
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
    return np.array(neti.dot(w0.T)).T[0]  
  
  def saveModel(self,model,position):
    self.ensemble[position] = model
    
  def fit(self, train_set, train_target):
    self.X_train = train_set
    self.y_train = train_target    
    neti = np.zeros((self.numMaxHiddenNodes,1))
    for i in list(range(self.numMaxHiddenNodes)):        
        neti= self.insertHiddenUnit(i)
        # elm = ELM(1,self.X_train,self.y_train)
        netInv = np.linalg.pinv(neti)
        w0 = np.dot(netInv,self.y_train)
        
        # w0 = self.regularization(neti,self.y_train)
        
        # print("w0 - whitout reg")
        # print(w02)
        # print("w0  - whit reg")
        # print(w0)
        
        model:Model = Model(self.weightsArray.copy(),w0.copy())
        self.saveModel(model,i)
        
        predTraining = self.calcPred(w0,neti)
                
        netiVal = self.forward(self.weightsArray,self.X_val)
        predVal = self.calcPred(w0,netiVal)
        
        mape, mse, rmse = calculateResidualError(self.y_val, predVal)
        
        if mse < self.minimumError:
          self.optimalWi = self.weightsArray.copy()
          self.optimalWo = w0.copy()
          self.minimumError = mse
          self.optimalNumHiddenNodes = i+1
        
        self.mapeArray.append(mape)
        self.mseArray.append(mse)                    
    
  def predict(self,input):
      
    # for i, model in self.ensemble.items():
       
    netiTeste = self.forward(self.optimalWi,input)
    predTeste = self.calcPred(self.optimalWo,netiTeste)      
      
      # printErrors(mape,mse)
      
    return predTeste
  
  