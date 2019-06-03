import deltafestimator
import pandas as pd
import numpy as np
from sys import argv


inFN = argv[1]
outFN= argv[2]
lmin,lmax,lstep = -5, 1, 50
lambdas = np.logspace(lmin, lmax, lstep)


tolerance = 1e-7
intensitykey = 'ipm2'
maxiter = 2000
referencekey = 'FCALC'


I = pd.read_csv(inFN)


columns = {
    'RUN' : int,
    'PHINUMBER': int,
    'SERIES': str, 
    'FOBS': float,
    'SIGMA(FOBS)': float,
    'FCALC': float,
    'PHIC': float,
    'H': int,
    'K': int, 
    'L': int,
    'MERGEDH': int,
    'MERGEDK': int,
    'MERGEDL': int,
    'IOBS': float,
    'SIGMA(IOBS)': float,
    'D': float, 
    'ipm2': float,
    'ipm3': float,
    'ipm2_xpos': float,
    'ipm3_xpos': float,
    'ipm2_ypos': float,
    'ipm3_ypos': float,
}      

for k in I:
    if k in columns:
        I[k] = I[k].astype(columns[k])
    else:
        del I[k]

Ftrace1 = pd.DataFrame()
Ftrace2 = pd.DataFrame()
Icryst = pd.DataFrame()

I['MASK'] = True
I['MASK'] = np.array(I[['RUN', 'PHINUMBER', 'MASK']].groupby(['RUN', 'PHINUMBER']).transform(lambda x: np.random.random()) < 0.5)


n = deltafestimator.deltafestimator_physical_gaussian(I[I['MASK']])
n.train({'LAMBDA': lambdas}, maxiter=maxiter, tolerance=tolerance)
_Ftrace1 = n.result['Miller']
_Icryst1 = n.result['Icryst']
_Icryst1[intensitykey] = n.result[intensitykey][intensitykey]

Ftrace1 = pd.concat((Ftrace1, _Ftrace1))
_Icryst1['HALF'] = 1
Icryst = pd.concat((Icryst, _Icryst1))

n = deltafestimator.deltafestimator_physical_gaussian(I[~I['MASK']])
n.train({'LAMBDA': lambdas}, maxiter=maxiter, tolerance=tolerance)
_Ftrace2 = n.result['Miller']
_Icryst2 = n.result['Icryst']
_Icryst2[intensitykey] = n.result[intensitykey][intensitykey]

Ftrace2 = pd.concat((Ftrace2, _Ftrace2))
_Icryst2['HALF'] = 2
Icryst = pd.concat((Icryst, _Icryst2))



Ftrace = Ftrace1.reset_index().join(Ftrace2.reset_index().set_index(['LAMBDA',  'H', 'K', 'L']), ['LAMBDA', 'H', 'K', 'L'], rsuffix='_2').dropna()
Ftrace.to_csv(outFN)
