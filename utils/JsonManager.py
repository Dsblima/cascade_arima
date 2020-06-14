import sys
import os
import json

def writeJsonFile(dictToSave, base, folderToSave):
    
    path = '../data/simulations/'+folderToSave
    
    if not os.path.isdir(path):
        os.mkdir(path)
        
    with open(path+'/'+base+'.json', 'w') as outfile:
        json.dump(dictToSave, outfile)

#READ A JSON FILE E RETURN A DICT 
def readJsonFile(path):
    with open(path) as json_file:
        loadedDict = json.load(json_file)
    return loadedDict
    
    