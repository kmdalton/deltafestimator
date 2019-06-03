import deltafestimator
import pandas as pd
import numpy as np
from sys import argv


#usage:
# python do_crossvalidation.py input_file.csv output_file.csv [-c]
# -c flag for control setting. this will ditch the 'off' data and 
# replace it with one half of the 'on' data to serve as a negative control


inFN = argv[1]
outFN= argv[2]
#lmin,lmax,lstep = -5, 1, 50
lmin,lmax,lstep = -5, 1, 1
lambdas = np.logspace(lmin, lmax, lstep)


tolerance = 1e-7
intensitykey = 'ipm2'
#maxiter = 2000
maxiter = 2
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

I['HALF'] = True
I['HALF'] = np.array(I[['RUN', 'PHINUMBER', 'SERIES', 'HALF']].groupby(['RUN', 'PHINUMBER', 'SERIES']).transform(lambda x: np.random.random()) < 0.5)

print(f"{len(I)} reflection observations before removing one third of the on data")
#Either remove the off data and rename half the on data or...
if len(argv) > 3 and argv[3] == '-c':
    print("This is a control run -- deleting the off data")
    from re import sub
    I = I[I.SERIES.str.contains('on')]
    I.loc[I.HALF, 'SERIES'] = I.loc[I.HALF, 'SERIES'].apply(lambda x: sub('on', 'off', x))
#Ditch half of the on data so we are comparing apples to apples
else:
    I = I[I.SERIES.str.contains('off') | I.HALF]
print(f"{len(I)} reflection observations after removing one third of the on data")

I['on']  = I.SERIES.str.contains('on')
I['off']  = I.SERIES.str.contains('off')
I = I.groupby(['H', 'K', 'L', 'RUN', 'PHINUMBER']).filter(lambda x: x.on.max() and x.off.max())
I['HALF'] = np.array(I[['RUN', 'PHINUMBER', 'HALF']].groupby(['RUN', 'PHINUMBER']).transform(lambda x: np.random.random()) < 0.5)
print(f"{len(I)} reflection observations after paring incomplete phi numbers")
print("""
Data partitions
Half:     1       2
  on:  {}  {}
 off:  {}  {}
""".format(
    (I.HALF  & I.SERIES.str.contains('on')).sum(),
    (~I.HALF & I.SERIES.str.contains('on')).sum(),
    (I.HALF  & I.SERIES.str.contains('off')).sum(),
    (~I.HALF & I.SERIES.str.contains('off')).sum(),
))
print(f"{I.HALF.sum()} reflection observations in the on data set")
print(f"{np.sum(~I.HALF)} reflection observations in the off data set")


n = deltafestimator.deltafestimator_physical_gaussian(I[I['HALF']])
n.train({'LAMBDA': lambdas}, maxiter=maxiter, tolerance=tolerance)
_Ftrace1 = n.result['Miller']
_Icryst1 = n.result['Icryst']
_Icryst1[intensitykey] = n.result[intensitykey][intensitykey]

Ftrace1 = pd.concat((Ftrace1, _Ftrace1))
_Icryst1['HALF'] = 1
Icryst = pd.concat((Icryst, _Icryst1))

n = deltafestimator.deltafestimator_physical_gaussian(I[~I['HALF']])
n.train({'LAMBDA': lambdas}, maxiter=maxiter, tolerance=tolerance)
_Ftrace2 = n.result['Miller']
_Icryst2 = n.result['Icryst']
_Icryst2[intensitykey] = n.result[intensitykey][intensitykey]

Ftrace2 = pd.concat((Ftrace2, _Ftrace2))
_Icryst2['HALF'] = 2
Icryst = pd.concat((Icryst, _Icryst2))



Ftrace = Ftrace1.reset_index().join(Ftrace2.reset_index().set_index(['LAMBDA',  'H', 'K', 'L']), ['LAMBDA', 'H', 'K', 'L'], rsuffix='_2').dropna()
Ftrace.to_csv(outFN)
