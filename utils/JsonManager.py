import sys
import os
import json
from datetime import date

def writeJsonFile(dictToSave, base):
    today = date.today()
    path = '../data/simulations/'+str(today)
    
    if not os.path.isdir(path):
        os.mkdir(path)
        
    with open(path+'/'+base+'.json', 'w') as outfile:
        json.dump(dictToSave, outfile)
        
def readJsonFile(path):
    print()        
    
    