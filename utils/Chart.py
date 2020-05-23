from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

def plotValidationAndTest(base="",model="",num_hidden_nodes=50,y1=[],
                          y2=[],y3=[],label1="",label2="",label3="",show=False, save=True):
  
  normalizedData=pd.DataFrame({'x': range(1,num_hidden_nodes+1), 'y1': y1[0]})
  noNormalizedData=pd.DataFrame({'x': range(1,num_hidden_nodes+1), 'y1': y1[1]})
        
  fig = plt.figure()
  fig.subplots_adjust(top=0.8)
  ax1 = fig.add_subplot(211)
  ax1.set_ylabel('MSE')
  ax1.set_xlabel('Num hidden nodes')
  ax1.set_title(base+" with normalization")  

  
  plt.plot( 'x', 'y1', data=normalizedData, marker='', markerfacecolor='grey', markersize=12, color='grey', linewidth=4,label=label1)
  
  if len(y2)!=0:
    normalizedData['y2'] = y2[0]
    noNormalizedData['y2'] = y2[1]
    plt.plot( 'x', 'y2', data=normalizedData, marker='', markerfacecolor='black', markersize=12, color='black', linewidth=4,label=label2)
  
  if len(y3)!=0:
    df['y3'] = y3[0]
    noNormalizedData['y3'] = y3[1]
    plt.plot( 'x', 'y3', data=normalizedData, marker='*', markerfacecolor='red', markersize=12, color='red', linewidth=4,label=label3)
  
  # fig = plt.gcf()  
  # fig.set_size_inches(16.5, 10.5, forward=True)
  # Add a legend
  plt.legend(prop={"size":20})
  fig.tight_layout(pad=3.0)
  ax2 = fig.add_subplot(212)
  # ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.3])
  ax2.set_ylabel('MSE')
  ax2.set_xlabel('Num hidden nodes')
  ax2.set_title(base+" without normalization")
  plt.plot( 'x', 'y1', data=noNormalizedData, marker='', markerfacecolor='grey', markersize=12, color='grey', linewidth=4,label=label1)
  plt.plot( 'x', 'y2', data=noNormalizedData, marker='', markerfacecolor='black', markersize=12, color='black', linewidth=4,label=label2)
  # ax2.plot(df['x'],df['y1'],df['y2'])

  fig = plt.gcf()  
  fig.set_size_inches(16.5, 10.5, forward=True)    
  
  if show:
    plt.show()    
  if save:
    plt.savefig(base+' '+model+'.png')
  plt.close()  