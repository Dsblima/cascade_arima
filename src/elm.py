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

class ELM(object):
    def __init__(self,hn,x,y):
        self.hiddennodes = hn
        self.input  = x
        self.output = y    
    def init_weights(self,xcol,hiddennodes):
        '''
        FUNÇÃO QUE INICIALIZA OS PESO QUE LIGAM A CAMADA DE ENTRADA À CAMADA ESCONDIDA, Wh
        RECEBE COMO ARGUMENTOS O NÚMERO DE COLUNAS DE DA ENTRADA E O NÚMERO DE NEURÔNIOS NA CAMADA ESCONDIDA.  
        '''     
        return (np.random.rand(xcol,hiddennodes))
    def pinv(self,matrix):
        '''
        RETORNA APSEUDO INVERSA DE UMA MATRIZ 
        '''
        return np.linalg.pinv(matrix)
    def getW0(self,netInv):
        '''
        RETORNA OS PESOS QUE LIGAM A CAMADA ESCONDIDA À SAÍDA
        '''
        return np.dot(netInv,self.output)