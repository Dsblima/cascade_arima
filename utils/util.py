import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

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
    