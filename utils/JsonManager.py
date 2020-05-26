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

#READ A JSON FILE E RETURN A DICT 
def readJsonFile(path):
    with open(path) as json_file:
        loadedDict = json.load(json_file)
    return loadedDict
    
    