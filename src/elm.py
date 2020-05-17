import sys, os
import numpy as np

sys.path.append(
	os.path.join(
		os.path.dirname(__file__), 
		'..', 
		'utils'
	)
)

from util import *
from activation_functions import *

class ELM(object):
    def __init__(self,hn):
        self.hiddennodes = hn
        self.input  = []
        self.output = []
        self.wi = [[]]    
        self.wo = [[]]
            
    def init_weights(self,xcol,hiddennodes):
        '''
        FUNÇÃO QUE INICIALIZA OS PESO QUE LIGAM A CAMADA DE ENTRADA À CAMADA ESCONDIDA, Wh
        RECEBE COMO ARGUMENTOS O NÚMERO DE COLUNAS DE DA ENTRADA E O NÚMERO DE NEURÔNIOS NA CAMADA ESCONDIDA.  
        '''     
        return (np.random.rand(xcol,hiddennodes))
    def pinv(self,matrix):
        '''
        RETORNA A PSEUDO INVERSA DE UMA MATRIZ 
        '''
        return np.linalg.pinv(matrix)
    def getW0(self,netInv):
        '''
        RETORNA OS PESOS QUE LIGAM A CAMADA ESCONDIDA À SAÍDA
        '''
        return np.dot(netInv,self.output)
    
    def fit(self,x,y):
        self.input  = x
        self.output = y
        self.wi = self.init_weights(self.input[0].__len__(),self.hiddennodes)
        net = self.input.dot(self.wi)
        net = sigmoid(net)
        netInv = self.pinv(net)
        self.w0 = self.getW0(netInv)
    
    def pred(self,input):
        net = input.dot(self.wi)
        net = sigmoid(net)
        return net.dot(self.w0.T)
             