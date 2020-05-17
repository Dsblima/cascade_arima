import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from sklearn.metrics import mean_squared_error
import Arquivo
from DataHandler import *
import Padronizar
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from math import sqrt

def add_1(arr):
    return np.append(arr,1)
  
def addBias(matrix):#this function adds the value 1, at the last position in each array from matrix
    
  bias = np.ones((matrix.__len__(),1))
  # return np.array(list(map(add_1, matrix)))
  return np.concatenate((matrix,bias),axis=1)

def roundNumber(number):
  return round(number,4)

def printErrors(mape,mse):
    print("mean_absolute_percentage_error")
    print(mape)
    print("mean_squared_error")
    print(mse)
    print()
    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
  
def load_and_preprocess_data(baseName,dimension):
    
    data = Arquivo.ler('../data/'+baseName+'.txt')
    dh:DataHandler = DataHandler(data, dimension, 60, 20,20)
    X_train, y_train, val_set, val_target, X_test, y_test, arima_train, arima_val, arima_test= dh.redimensiondata(data, dimension, 60, 20,20)

    y = [[]]
    x = np.concatenate( (X_train,X_test) )   

    y = np.matrix( np.concatenate( (np.array(y_train),np.array(y_test)) ))   
    data = np.concatenate((x, y.T), axis=1)
    x, y = Padronizar.dividir(data, dimension, 1)


    # data = Arquivo.ler('../data/projeto base completa.csv')
    # x, y = Padronizar.dividir(data, 4, 1)
    # print("y")
    # print(y[y.columns[0]])
    # minmaxscaler = MinMaxScaler(feature_range=(0,1))
    dataNX, listMin,  listMax  = Padronizar.normalizarLinear(x, 0.1, 0.9)
    dataNY, listMinY, listMaxY = Padronizar.normalizarLinear(y, 0.1, 0.9)
    # scalerX,scalerY, dataNormalizadoX, dataNormalizadoY = Padronizar.normalizar(x,y)
    X_train, X_test, y_train, y_test = train_test_split(dataNX, dataNY, train_size = 0.8, test_size  = 0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size = 0.5, test_size  = 0.5)
    
    X_train = addBias(X_train.values)    
    X_val  = addBias(X_val.values)
    X_test  = addBias(X_test.values)
    
    # WE WILL USE ONLY THE FIRST COLUMN 														
    y_train = y_train[y_train.columns[0]].values
    y_test = y_test[y_test.columns[0]].values
    y_val = y_val[y_val.columns[0]].values
    
    return   X_train, y_train, X_val, y_val, X_test, y_test

def calculateResidualError(trueValues,pred):    
    # print(np.matrix(trueValues).T)
    # trueValues = Padronizar.desnormalizarLinear(pd.DataFrame(np.matrix(trueValues).T), self.listMaxY, self.listMinY, 0.1, 0.9)
    # print(trueValues)
    # pred = Padronizar.desnormalizarLinear(pd.DataFrame(np.matrix(pred)), self.listMaxY, self.listMinY, 0.1, 0.9)
    # print(pred)
    # print("Residual Error")
    mape = mean_absolute_percentage_error(trueValues,pred)
    
    mse = mean_squared_error(trueValues,pred)
    
    rmse = sqrt(mse)
    
    return mape, mse, rmse
def plot(base="",model="",num_hidden_nodes=50,y1=[],y2=[],y3=[],label1="",label2="",label3="",show=False, save=True):
  
  df=pd.DataFrame({'x': range(1,num_hidden_nodes+1), 'y1': y1})
        
  fig = plt.figure()
  fig.subplots_adjust(top=0.8)
  ax1 = fig.add_subplot(211)
  ax1.set_ylabel('MSE')
  ax1.set_xlabel('Num hidden nodes')
  ax1.set_title(base)
  
  plt.plot( 'x', 'y1', data=df, marker='', markerfacecolor='grey', markersize=12, color='grey', linewidth=4,label=label1)
  
  if len(y2)!=0:
    df['y2'] = y2
    plt.plot( 'x', 'y2', data=df, marker='+', markerfacecolor='black', markersize=12, color='black', linewidth=4,label=label2)
  
  if len(y3)!=0:
    df['y3'] = y3
    plt.plot( 'x', 'y3', data=df, marker='*', markerfacecolor='red', markersize=12, color='red', linewidth=4,label=label3)
  
  fig = plt.gcf()  
  fig.set_size_inches(16.5, 10.5, forward=True)
  
  # Add a legend
  plt.legend(prop={"size":20})
  
  if show:
    plt.show()    
  if save:
    plt.savefig(base+' '+model+'.png')
  plt.close()  
    