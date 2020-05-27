from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd


class Chart(object):
    def __init__(self):
        self.subplot = 211
        self.fig = plt.figure()
    def plotValidationAndTest(self,base="", model="", num_hidden_nodes=50, valSet=[],
                              testSet=[], label1="", label2="", show=False, save=True):
        
        self.fig.subplots_adjust(top=0.8)
        
        self.plotSubChart(base,num_hidden_nodes, valSet,testSet, label1, 
                 label2, "MSE", "Num hidden nodes", base+" with normalization")
        
        plt.legend(prop={"size": 20})
        self.fig.tight_layout(pad=3.0)
        
        # self.plotSubChart(base,num_hidden_nodes, valSet[1],testSet[1], label1, 
        #          label2, "MSE", "Num hidden nodes", base+" without normalization")        

        self.fig = plt.gcf()
        self.fig.set_size_inches(16.5, 10.5, forward=True)

        if show:
            plt.show()
        if save:
            plt.savefig(base+' '+model+'.png')
        plt.close()

    def plotSubChart(self,base="",num_hidden_nodes=50, valSet=[],testSet=[], label1="", 
                 label2="", metric="", xlabel="", title=""):
        df = pd.DataFrame(
            {'x': range(1, num_hidden_nodes+1), 'valSet': valSet, 'testSet' : testSet })
        
        ax = self.fig.add_subplot(self.subplot)
        ax.set_ylabel(metric)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        self.subplot += 1
        
        plt.plot('x', 'valSet', data=df, marker='', markerfacecolor='grey',
                 markersize=12, color='grey', linewidth=4, label=label1)
        plt.plot('x', 'testSet', data=df, marker='', markerfacecolor='black',
                     markersize=12, color='black', linewidth=4, label=label2)
        
