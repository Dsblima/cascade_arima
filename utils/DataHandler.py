import pandas as pd
import numpy


class DataHandler:
    def __init__(self, data, dimension, trainset, valset,testNO):
        self.data=data
        self.dimension=dimension
        self.trainset=trainset
        self.valset=valset
        self.ndata=(data-min(data))/(max(data)-min(data))
        self.testNO=testNO
    def redimensiondata(self, data, dimension, trainset, valset,testNO):

        s1 = pd.Series(data)
        lin2 = len(data)
        res = pd.concat([s1.shift(i) for i in range(0, dimension + 1)], axis=1)
        res2 = res
        lin = len(res2)

        test = res2.iloc[lin-testNO:lin+1, 1:dimension + 1]
        test_target = res2.iloc[lin-testNO:lin+1, 0]

        l3=lin2-testNO

        tra = res2.iloc[dimension:int(numpy.floor(trainset * l3)), 1:dimension + 1]
        tra_target = res2.iloc[dimension:int(numpy.floor(trainset * l3)), 0]
        tra=tra.dropna()

        val = res2.iloc[int(numpy.floor((trainset) * l3)):l3, 1:dimension + 1]
        val_target = res2.iloc[int(numpy.floor((trainset) * l3)):l3, 0]

        #test = res2.iloc[int(numpy.floor((trainset + valset) * lin)):lin + 1, 1:dimension + 1]
        #test_target = res2.iloc[int(numpy.floor((trainset + valset) * lin)):lin + 1, 0]


        lintra=len(tra_target)
        linval=len(val_target)
        lintest=len(test_target)
     #   tra = res2.iloc[dimension:int(numpy.floor(trainset * lin)), 1:dimension + 1]
        arima_train = data[0:int(numpy.floor(trainset * l3)) ]
        arima_val = data[int(numpy.floor(trainset * l3)) :l3]
        arima_test = data[l3:lin+1]
       # arima_train=tra_target
       # arima_val=val_target
        #arima_test=test_target
        return tra.values.tolist(), tra_target.values.tolist(), val.values.tolist(), val_target.values.tolist(), test.values.tolist(), test_target.values.tolist(), arima_train.tolist(), arima_val.tolist(), arima_test.tolist()
