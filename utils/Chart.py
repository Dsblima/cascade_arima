from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.subplots import make_subplots

class Chart(object):
    def __init__(self):
        self.subplot = 211
        self.fig = plt.figure()
    def plotValidationAndTest(self,base="", model="", num_hidden_nodes=50, valSet=[],
                              testSet=[], label1="", label2="", titles=[], show=False, save=True, folderToSave=''):
        
        self.fig.subplots_adjust(top=0.8)
        
        self.plotSubChart(base,num_hidden_nodes, valSet[0],testSet[0], label1, 
                 label2, "MSE", "Num hidden nodes", base+titles[0])
        
        plt.legend(prop={"size": 20})
        self.fig.tight_layout(pad=3.0)
        
        if len(valSet) > 1:
            self.plotSubChart(base,num_hidden_nodes, valSet[1],testSet[1], label1, 
                    label2, "MSE", "Num hidden nodes", base+titles[1])        

        self.fig = plt.gcf()
        self.fig.set_size_inches(16.5, 10.5, forward=True)
        # self.plotTable()

        if show:
            plt.show()
        if save:
            plt.savefig(folderToSave+base+' '+model+'.png')
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

    def plotTable(self,valuesArray,path):        
        nodeList =list(range(1,len(valuesArray)+1))
        # valuesArray = np.matrix(valuesArray)
        
        matrixList = []
                
        # data =np.concatenate((nodeList, valuesArray)).T
        for node, value in zip(nodeList, valuesArray):
            matrixList.append([node, value])
         
        df = pd.DataFrame(matrixList, columns = ['Num HD', 'MSE'])
        # df.to_csv(path, index=False)
        
        fig = make_subplots(
            rows=1, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            specs=[[{"type": "table"}]]
        )
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["NUM HD", "MSE"],
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[df[k].tolist() for k in df.columns[1:]],
                    align = "left")
            ),
            row=1, col=1
        )
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Error by number of hidden nodes",
        )

        fig.show()

        # table = plt.table(cellText=matrixList, colLabels=['Num HD', 'MSE'], loc='center', 
        #                 cellLoc='center', colColours=['#FFFFFF', '#F3CC32'])
        # table.auto_set_font_size(False)
        # h = table.get_celld()[(0,0)].get_height()
        # w = table.get_celld()[(0,0)].get_width()

        # # Create an additional Header
        # # header = [table.add_cell(-1,pos, w, h, loc="center", facecolor="none") for pos in [0,1]]
        # # header[0].visible_edges = "TBL"
        # # header[1].visible_edges = "TB"
        # # # header[2].visible_edges = "TBR"
        # # header[1].get_text().set_text("Header Header Header Header")

        # plt.axis('off')
        # plt.show()
