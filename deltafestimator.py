import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
from sys import stdout
from scipy import sparse
from tensorflow.python.ops.parallel_for.gradients import jacobian
from tensorflow.contrib.layers import dense_to_sparse
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow import dtypes

pd.options.mode.chained_assignment = None
tf.logging.set_verbosity(tf.logging.ERROR)


"""
Implementation of delta F estimation using the Coppens ratio method with the random diffuse model. 
This implementation will individually weight reflection intensities by their corresponding Io estimates.
I will also seek to implement dose corrections for radiation damage. 

TODO:
1) Implement without weights or radiation damage
2) Add weights
3) Add radiation Damage
"""


class base_model:
    """
    Base class for models with hyperparameters that need crossvalidation.
    This does not cover learning rates or momentum parameters in optimizers. 
    This class is targeted more at regularization parameters. 
    The hyperparameters are going to be passed into the graph through a feed_dict for fast training
    of an entire regularization trajectory. 

    ATTRIBUTES
    ----------
    variables : dict
        Dictionary of {name: tensor} for variables with respect to which gradients are computed during optimization. 
        Only entries in this dictionary will be returned from the trained model. 
    loss : tensor
        Tensorflow tensor with shape 0 which will be minimized
    logvars : dict
        Dictionary of {name: tensor} for scalar variables which will be recorded at each optimization step and returned in the trace. 
        One of these must be called "loss". It will be used to determine the node to be minimized. 
    hyperparameters : dict
        Dictionary of {name: tensor} for hyperparameters which will be established from a feed_dict during optimization. 
    log : pd.Dataframe
        Dataframe with the values of logvars during training. Populated by self.train
    result : pd.Dataframe
        Dataframe with the values of variables after training. Populated by self.train
    """
#The user should set up the following dictionaries
    variables = {}
    hyperparameters = {}
    logvars = {}
#The results and associated trajectories are stored in these things
    log = None
    result = {}
#The class uses these attrs to pass around info about the graph
#Ideally the user never needs to look in here
    _loss = None
    _optim = None 
    _sess = None #Current tf.Session
    _feed_dict = None #Use this to pass around current hyperparameter settings {tf.placeholder: value}
    _iternumber = None
    _tolerance = None
        

    def _build_graph(self):
        """
        Extensions of this class must implement this function to populate 
            self.variables
            self.loss
            self.hyperparameters
            self.variables
        """
        raise NotImplementedError

    def _update_log(self):
        step_log = pd.DataFrame()
        for k,v in self.logvars.items():
            val = self._sess.run(v, feed_dict=self._feed_dict)
            #print(f'{k} : {val}')
            step_log[k] = [val]

        step_log['Step'] = self.iternumber
        for k,v in self.hyperparameters.items():
            step_log[k] = self._feed_dict[v]
        #print("step log:\n{}".format(step_log))

        self.log = pd.concat((self.log, step_log))

    def _step(self):
        self._sess.run(self._optim, feed_dict=self._feed_dict)
        self.iternumber += 1
        self._update_log()

    def _converged(self):
        #print(self.log)
        loss,loss_ = self.log['Loss'][-2:]
        if np.abs(loss - loss_)/loss <= self._tolerance:
            return True
        else:
            return False

    def _update_result(self):
        for k,v in self.variables.items():
            result = pd.DataFrame()
            result[k] = np.array(self._sess.run(v, feed_dict=self._feed_dict)).flatten()
            for key,value in self.hyperparameters.items():
                result[key] = self._feed_dict[value]
            self.result[k] = pd.concat((self.result[k], result))

    def _train(self, maxiter=1000):
        """
        Helper function that trains the model for an single combination of hyperparameters.
        """
        intensities = pd.DataFrame()
        self._sess.run(tf.global_variables_initializer(), self._feed_dict)
        self.iternumber = 0

        #Populate initial loss value
        self._update_log()
        self._step()
        for i in range(maxiter):
            self._step()
            if self._converged():
                break
        self._update_result()

    def train(self, hyperparameters, optimizer=None, config=None, maxiter=1000, tolerance=1e-5):
        """
        Train the model for a number of hyperperameter values. 

        PARAMETERS
        ----------
        hyperparameters : dict
            Dictionary of {key: iterable or float} where the keys correspond to entries in self.hyperparameters.
        """
        self._tolerance = tolerance
        self._loss = self.logvars['Loss']
        self.result = {k:None for k in self.variables}

        #Convert values to an iterable if it isn't already
        hyperparameters = {k: np.array(v).flatten() for k,v in hyperparameters.items()}
        length = len(next(iter(hyperparameters.values())))
        self._sess = tf.Session(config=config)
        self._optim = optimizer if optimizer is not None else tf.train.AdamOptimizer(0.05)
        #self._optim = self._optim.minimize(self._loss, var_list=list(self.variables.values()))
        self._optim = self._optim.minimize(self._loss)

        for i in tqdm(range(length)):
            self._feed_dict = {self.hyperparameters[k]:v[i] for k,v in hyperparameters.items()}
            self._train(maxiter)

        self._sess.close()
        return self

class hkl_model(base_model):
    """
    This is a model for crystallographic inference wherein a subset of the log variables have associated miller indices. 
    """
    H = None
    K = None
    L = None
    miller_vars = None

    def _update_result(self):
        if 'Miller' not in self.result:
            self.result['Miller'] = None
        miller_result = pd.DataFrame()
        miller_result['H'], miller_result['K'], miller_result['L'] = self.H, self.K, self.L
        for k,v in self.hyperparameters.items():
            miller_result[k] = self._feed_dict[v]
        for k,v in self.variables.items():
            result = pd.DataFrame()
            result[k] = np.array(self._sess.run(v, feed_dict=self._feed_dict)).flatten()
            if k in self.miller_vars:
                miller_result[k] = result[k]
            for key,value in self.hyperparameters.items():
                result[key] = self._feed_dict[value]
            self.result[k] = pd.concat((self.result[k], result))
        self.result['Miller'] = pd.concat((self.result['Miller'], miller_result))

class deltafestimator(hkl_model):
    def __init__(self, df):
        self._build_graph(df)

    def _build_graph(self, df, referencekey='FCALC', intensitykey='ipm2'):
        tf.reset_default_graph()
        rhop, lp = tf.placeholder(tf.float32),tf.placeholder(tf.float32)
        self.hyperparameters['RHO'], self.hyperparameters['LAMBDA'] = rhop, lp

        indices = {
            'GAMMAINDEX'        : ['MERGEDH', 'MERGEDK', 'MERGEDL'],
            'RUNINDEX'          : 'RUN',
            'IMAGEINDEX'        : ['RUN', 'PHINUMBER', 'SERIES'],
            'PHIINDEX'          : ['RUN', 'PHINUMBER'],
        }

        for k,v in indices.items():
            df.loc[:,k] = df.groupby(v).ngroup()
        df

        #Prepare the per image metadata
        k = [i for i in df if 'ipm' in i.lower()]
        k += ['RUNINDEX']
        imagemetadata = df[k + ['IMAGEINDEX']].groupby('IMAGEINDEX').first()

        #Construct pivot tables of intensities, errors, metadatakeys
        iobs        = df.pivot_table(values='IOBS', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'], columns='SERIES', fill_value=0)
        imagenumber = df.pivot_table(values='IMAGEINDEX', index=['H', 'K', 'L', 'RUNINDEX', 'PHIINDEX'], columns='SERIES', fill_value=0)
        sigma       = df.pivot_table(values='SIGMA(IOBS)', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'], columns='SERIES', fill_value=0)

        gammaidx = df.pivot_table(values='GAMMAINDEX', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'])
        gammaidx = np.array(gammaidx).flatten()
        Foff = df[['GAMMAINDEX', referencekey]].groupby('GAMMAINDEX').first().values.flatten()
        Foff_per_loss_term = Foff[gammaidx]
        Foff,Foff_per_loss_term = tf.constant(Foff, dtype=tf.float32), tf.constant(Foff_per_loss_term, dtype=tf.float32)

        ion    = iobs[[i for i in iobs if  'on' in i]].values
        ioff   = iobs[[i for i in iobs if 'off' in i]].values
        sigon  = sigma[[i for i in sigma if  'on' in i]].values**2
        sigoff = sigma[[i for i in sigma if 'off' in i]].values**2
        imon   = imagenumber[[i for i in imagenumber if  'on' in i]].values
        imoff  = imagenumber[[i for i in imagenumber if 'off' in i]].values

        #Compute inverse variance weights and normalize them to have rows sum to one
        invaron,invaroff = sigon.copy(),sigoff.copy()
        invaron[invaron > 0]    =  1/invaron[invaron > 0]
        invaroff[invaroff > 0]  = 1/invaroff[invaroff > 0]
        invaron  = invaron/invaron.sum(1)[:,None]
        invaroff = invaroff/invaroff.sum(1)[:,None]

        ion  = (ion*invaron).sum(1)
        ioff = (ioff*invaroff).sum(1)
        sigon  = (invaron**2*sigon).sum(1)
        sigoff = (invaroff**2*sigoff).sum(1)

        invaron  = tf.constant(invaron, dtype=tf.float32)
        invaroff = tf.constant(invaroff, dtype=tf.float32)

        ion     = tf.convert_to_tensor(ion, dtype=tf.float32, name='ion')
        ioff    = tf.convert_to_tensor(ioff, dtype=tf.float32, name='ioff')
        sigon   = tf.convert_to_tensor(sigon, dtype=tf.float32, name='sigon')
        sigoff  = tf.convert_to_tensor(sigoff, dtype=tf.float32, name='sigoff')

        #Problem Variables
        h = gammaidx.max() + 1
        ipm = np.float32(imagemetadata[intensitykey])
        variables = tf.Variable(np.concatenate((np.ones(h, dtype=np.float32), ipm)))
        ipm = tf.constant(ipm)
        deltaF = variables[:h]
        gammas = deltaF/Foff + 1 #This is only here for backward compatibility. We will not use it directly
        DF = tf.gather(deltaF, gammaidx)
        Icryst = tf.nn.softplus(variables[h:])

        Bon  = tf.reduce_sum(invaron*tf.gather(Icryst, imon), 1)
        Boff = tf.reduce_sum(invaroff*tf.gather(Icryst, imoff), 1)

        variance = ((DF/Foff_per_loss_term+ 1)*(Bon/Boff))**2*sigoff + sigon
        weights = variance**-1
        weights = weights/tf.reduce_sum(weights) #normalize the weights to one to stabilize regularizer tuning


        likelihood = tf.losses.mean_squared_error(
            ion,
            (DF/Foff_per_loss_term + 1)*(Bon/Boff)*ioff,
            weights=weights
        )

        regularizer = rhop*tf.losses.mean_squared_error(ipm, Icryst)
        sparsifier  = lp*tf.losses.mean_squared_error(tf.zeros(deltaF.shape), deltaF)

        loss = likelihood + regularizer + sparsifier

        #print("6: {}".format(time() - start))
        self.H = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDH'], dtype=int)
        self.K = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDK'], dtype=int)
        self.L = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDL'], dtype=int)

        # Declare an iterator and tensor array loop variables for the gradients.
        n = array_ops.size(gammas)
        loop_vars = [
            array_ops.constant(0, dtypes.int32),
            tensor_array_ops.TensorArray(gammas.dtype, n)
        ]    
        # Iterate over all elements of the gradient and compute second order
        # derivatives.
        gradients = tf.gradients(likelihood, deltaF)[0]
        gradients = array_ops.reshape(gradients, [-1])
        _, diag_A = control_flow_ops.while_loop(
            lambda j, _: j < n, 
            lambda j, result: (j + 1, 
                               result.write(j, tf.gather(tf.gradients(gradients[j], deltaF)[0], j))),
            loop_vars
        )    
        epsilon = 1e-32
        diag_A = array_ops.reshape(diag_A.stack(), [n])
        self.variables['DeltaF'] = deltaF
        self.variables['SIGMA(DeltaF)'] = 1./(diag_A + epsilon)
        self.variables[referencekey] = Foff
        self.variables['Icryst'] = Icryst
        self.variables[intensitykey] = ipm
        self.logvars['Loss'] = loss
        self.logvars['Regularizer'] = regularizer
        self.logvars['Sparsifier'] = sparsifier
        self.logvars['Likelihood'] = likelihood
        self.miller_vars = ['DeltaF', 'SIGMA(DeltaF)', referencekey]

class deltafestimator_laplacian(hkl_model):
    def __init__(self, df):
        self._build_graph(df)

    def _build_graph(self, df, referencekey='FCALC', intensitykey='ipm2'):
        tf.reset_default_graph()
        rhop, lp = tf.placeholder(tf.float32),tf.placeholder(tf.float32)
        self.hyperparameters['RHO'], self.hyperparameters['LAMBDA'] = rhop, lp

        indices = {
            'GAMMAINDEX'        : ['MERGEDH', 'MERGEDK', 'MERGEDL'],
            'RUNINDEX'          : 'RUN',
            'IMAGEINDEX'        : ['RUN', 'PHINUMBER', 'SERIES'],
            'PHIINDEX'          : ['RUN', 'PHINUMBER'],
        }

        for k,v in indices.items():
            df.loc[:,k] = df.groupby(v).ngroup()
        df

        #Prepare the per image metadata
        k = [i for i in df if 'ipm' in i.lower()]
        k += ['RUNINDEX']
        imagemetadata = df[k + ['IMAGEINDEX']].groupby('IMAGEINDEX').first()

        #Construct pivot tables of intensities, errors, metadatakeys
        iobs        = df.pivot_table(values='IOBS', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'], columns='SERIES', fill_value=0)
        imagenumber = df.pivot_table(values='IMAGEINDEX', index=['H', 'K', 'L', 'RUNINDEX', 'PHIINDEX'], columns='SERIES', fill_value=0)
        sigma       = df.pivot_table(values='SIGMA(IOBS)', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'], columns='SERIES', fill_value=0)

        gammaidx = df.pivot_table(values='GAMMAINDEX', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'])
        gammaidx = np.array(gammaidx).flatten()
        Foff = df[['GAMMAINDEX', referencekey]].groupby('GAMMAINDEX').first().values.flatten()
        Foff_per_loss_term = Foff[gammaidx]
        Foff,Foff_per_loss_term = tf.constant(Foff, dtype=tf.float32), tf.constant(Foff_per_loss_term, dtype=tf.float32)

        ion    = iobs[[i for i in iobs if  'on' in i]].values
        ioff   = iobs[[i for i in iobs if 'off' in i]].values
        sigon  = sigma[[i for i in sigma if  'on' in i]].values**2
        sigoff = sigma[[i for i in sigma if 'off' in i]].values**2
        imon   = imagenumber[[i for i in imagenumber if  'on' in i]].values
        imoff  = imagenumber[[i for i in imagenumber if 'off' in i]].values

        #Compute inverse variance weights and normalize them to have rows sum to one
        invaron,invaroff = sigon.copy(),sigoff.copy()
        invaron[invaron > 0]    =  1/invaron[invaron > 0]
        invaroff[invaroff > 0]  = 1/invaroff[invaroff > 0]
        invaron  = invaron/invaron.sum(1)[:,None]
        invaroff = invaroff/invaroff.sum(1)[:,None]

        ion  = (ion*invaron).sum(1)
        ioff = (ioff*invaroff).sum(1)
        sigon  = (invaron**2*sigon).sum(1)
        sigoff = (invaroff**2*sigoff).sum(1)

        invaron  = tf.constant(invaron, dtype=tf.float32)
        invaroff = tf.constant(invaroff, dtype=tf.float32)

        ion     = tf.convert_to_tensor(ion, dtype=tf.float32, name='ion')
        ioff    = tf.convert_to_tensor(ioff, dtype=tf.float32, name='ioff')
        sigon   = tf.convert_to_tensor(sigon, dtype=tf.float32, name='sigon')
        sigoff  = tf.convert_to_tensor(sigoff, dtype=tf.float32, name='sigoff')

        #Problem Variables
        h = gammaidx.max() + 1
        ipm = np.float32(imagemetadata[intensitykey])
        variables = tf.Variable(np.concatenate((np.ones(h, dtype=np.float32), ipm)))
        ipm = tf.constant(ipm)
        deltaF = variables[:h]
        gammas = deltaF/Foff + 1 #This is only here for backward compatibility. We will not use it directly
        DF = tf.gather(deltaF, gammaidx)
        Icryst = tf.nn.softplus(variables[h:])

        Bon  = tf.reduce_sum(invaron*tf.gather(Icryst, imon), 1)
        Boff = tf.reduce_sum(invaroff*tf.gather(Icryst, imoff), 1)

        variance = ((DF/Foff_per_loss_term+ 1)*(Bon/Boff))**2*sigoff + sigon
        weights = variance**-1
        weights = weights/tf.reduce_sum(weights) #normalize the weights to one to stabilize regularizer tuning


        likelihood = tf.losses.mean_squared_error(
            ion,
            (DF/Foff_per_loss_term + 1)*(Bon/Boff)*ioff,
            weights=weights
        )

        regularizer = rhop*tf.losses.mean_squared_error(ipm, Icryst)
        sparsifier  = lp*tf.linalg.norm(deltaF, 1)/tf.size(deltaF, out_type=tf.float32)

        loss = likelihood + regularizer + sparsifier

        #print("6: {}".format(time() - start))
        self.H = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDH'], dtype=int)
        self.K = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDK'], dtype=int)
        self.L = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDL'], dtype=int)

        # Declare an iterator and tensor array loop variables for the gradients.
        n = array_ops.size(gammas)
        loop_vars = [
            array_ops.constant(0, dtypes.int32),
            tensor_array_ops.TensorArray(gammas.dtype, n)
        ]    
        # Iterate over all elements of the gradient and compute second order
        # derivatives.
        gradients = tf.gradients(likelihood, deltaF)[0]
        gradients = array_ops.reshape(gradients, [-1])
        _, diag_A = control_flow_ops.while_loop(
            lambda j, _: j < n, 
            lambda j, result: (j + 1, 
                               result.write(j, tf.gather(tf.gradients(gradients[j], deltaF)[0], j))),
            loop_vars
        )    
        epsilon = 1e-32
        diag_A = array_ops.reshape(diag_A.stack(), [n])
        self.variables['DeltaF'] = deltaF
        self.variables['Hessian Diagonal'] = diag_A
        self.variables['SIGMA(DeltaF)'] = 1./(diag_A + epsilon)
        self.variables[referencekey] = Foff
        self.variables['Icryst'] = Icryst
        self.variables[intensitykey] = ipm
        self.logvars['Loss'] = loss
        self.logvars['Regularizer'] = regularizer
        self.logvars['Sparsifier'] = sparsifier
        self.logvars['Likelihood'] = likelihood
        self.miller_vars = ['DeltaF', 'SIGMA(DeltaF)', referencekey]

class deltafestimator_physical_laplacian(hkl_model):
    def __init__(self, df):
        self._build_graph(df)

    def _build_graph(self, df, referencekey='FCALC', intensitykey='ipm2'):
        tf.reset_default_graph()
        lp = tf.placeholder(tf.float32)
        self.hyperparameters['LAMBDA'] = lp

        indices = {
            'GAMMAINDEX'        : ['MERGEDH', 'MERGEDK', 'MERGEDL'],
            'RUNINDEX'          : 'RUN',
            'IMAGEINDEX'        : ['RUN', 'PHINUMBER', 'SERIES'],
            'PHIINDEX'          : ['RUN', 'PHINUMBER'],
        }

        for k,v in indices.items():
            df.loc[:,k] = df.groupby(v).ngroup()
        df

        #Prepare the per image metadata
        k = [i for i in df if 'ipm' in i.lower()]
        k += ['RUNINDEX']
        imagemetadata = df[k + ['IMAGEINDEX']].groupby('IMAGEINDEX').first()
        runidx = imagemetadata['RUNINDEX'].values

        #Construct pivot tables of intensities, errors, metadatakeys
        iobs        = df.pivot_table(values='IOBS', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'], columns='SERIES', fill_value=0)
        imagenumber = df.pivot_table(values='IMAGEINDEX', index=['H', 'K', 'L', 'RUNINDEX', 'PHIINDEX'], columns='SERIES', fill_value=0)
        sigma       = df.pivot_table(values='SIGMA(IOBS)', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'], columns='SERIES', fill_value=0)

        gammaidx = df.pivot_table(values='GAMMAINDEX', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'])
        gammaidx = np.array(gammaidx).flatten()
        Foff = df[['GAMMAINDEX', referencekey]].groupby('GAMMAINDEX').first().values.flatten()
        Foff_per_loss_term = Foff[gammaidx]
        Foff,Foff_per_loss_term = tf.constant(Foff, dtype=tf.float32), tf.constant(Foff_per_loss_term, dtype=tf.float32)

        ion    = iobs[[i for i in iobs if  'on' in i]].values
        ioff   = iobs[[i for i in iobs if 'off' in i]].values
        sigon  = sigma[[i for i in sigma if  'on' in i]].values**2
        sigoff = sigma[[i for i in sigma if 'off' in i]].values**2
        imon   = imagenumber[[i for i in imagenumber if  'on' in i]].values
        imoff  = imagenumber[[i for i in imagenumber if 'off' in i]].values

        #Compute inverse variance weights and normalize them to have rows sum to one
        invaron,invaroff = sigon.copy(),sigoff.copy()
        invaron[invaron > 0]    =  1/invaron[invaron > 0]
        invaroff[invaroff > 0]  = 1/invaroff[invaroff > 0]
        invaron  = invaron/invaron.sum(1)[:,None]
        invaroff = invaroff/invaroff.sum(1)[:,None]

        ion  = (ion*invaron).sum(1)
        ioff = (ioff*invaroff).sum(1)
        sigon  = (invaron**2*sigon).sum(1)
        sigoff = (invaroff**2*sigoff).sum(1)

        invaron  = tf.constant(invaron, dtype=tf.float32)
        invaroff = tf.constant(invaroff, dtype=tf.float32)

        ion     = tf.convert_to_tensor(ion, dtype=tf.float32, name='ion')
        ioff    = tf.convert_to_tensor(ioff, dtype=tf.float32, name='ioff')
        sigon   = tf.convert_to_tensor(sigon, dtype=tf.float32, name='sigon')
        sigoff  = tf.convert_to_tensor(sigoff, dtype=tf.float32, name='sigoff')

        #Problem Variables
        h = gammaidx.max() + 1
        r = runidx.max() + 1
        ipm  = tf.constant(np.float32(imagemetadata[intensitykey]))
        ipmx = tf.constant(np.float32(imagemetadata[intensitykey+'_xpos']))
        ipmy = tf.constant(np.float32(imagemetadata[intensitykey+'_ypos']))
        ipmy = tf.constant(np.float32(imagemetadata[intensitykey+'_ypos']))
        xstd = np.std(imagemetadata[intensitykey+'_xpos'])
        ystd = np.std(imagemetadata[intensitykey+'_ypos'])
        variables = tf.Variable(np.concatenate((
            np.ones(h, dtype=np.float32),
            -2.*xstd*np.ones(r, dtype=np.float32),
             2.*xstd*np.ones(r, dtype=np.float32),
            -2.*ystd*np.ones(r, dtype=np.float32),
             2.*ystd*np.ones(r, dtype=np.float32),
             np.ones(1, dtype=np.float32),
        )))
        deltaF = variables[:h]
        xmin = variables[h:h+r]
        xmax = variables[h:h+r]
        ymin = variables[h+r:h+2*r]
        ymax = variables[h+2*r:h+3*r]
        ipm_zp = variables[-1]
        gammas = deltaF/Foff + 1 #This is only here for backward compatibility. We will not use it directly
        DF = tf.gather(deltaF, gammaidx)
        Icryst = (ipm + ipm_zp)*tf.nn.softplus(ipm * (
            tf.erf(tf.gather(xmax, runidx) - ipmx) - 
            tf.erf(tf.gather(xmin, runidx) - ipmx)
            ) * (
            tf.erf(tf.gather(ymax, runidx) - ipmy) - 
            tf.erf(tf.gather(ymin, runidx) - ipmy)
        ))

        Bon  = tf.reduce_sum(invaron*tf.gather(Icryst, imon), 1)
        Boff = tf.reduce_sum(invaroff*tf.gather(Icryst, imoff), 1)

        variance = ((DF/Foff_per_loss_term+ 1)*(Bon/Boff))**2*sigoff + sigon
        weights = variance**-1
        weights = weights/tf.reduce_sum(weights) #normalize the weights to one to stabilize regularizer tuning


        likelihood = tf.losses.mean_squared_error(
            ion,
            (DF/Foff_per_loss_term + 1)*(Bon/Boff)*ioff,
            weights=weights
        )

        sparsifier  = tf.linalg.norm(deltaF, 1)/tf.size(deltaF, out_type=tf.float32)

        loss = (1. - lp)*likelihood + lp*sparsifier

        #print("6: {}".format(time() - start))
        self.H = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDH'], dtype=int)
        self.K = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDK'], dtype=int)
        self.L = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDL'], dtype=int)

        # Declare an iterator and tensor array loop variables for the gradients.
        n = array_ops.size(deltaF)
        loop_vars = [
            array_ops.constant(0, dtypes.int32),
            tensor_array_ops.TensorArray(gammas.dtype, n)
        ]    
        # Iterate over all elements of the gradient and compute second order
        # derivatives.
        gradients = tf.gradients(loss, deltaF)[0]
        gradients = array_ops.reshape(gradients, [-1])
        _, diag_A = control_flow_ops.while_loop(
            lambda j, _: j < n, 
            lambda j, result: (j + 1, 
                               result.write(j, tf.gather(tf.gradients(gradients[j], deltaF)[0], j))),
            loop_vars
        )    
        epsilon = 1e-32
        diag_A = array_ops.reshape(diag_A.stack(), [n])
        self.variables['DeltaF'] = deltaF
        self.variables['Hessian Diagonal'] = diag_A
        self.variables['SIGMA(DeltaF)'] = 1./(diag_A + epsilon)
        self.variables[referencekey] = Foff
        self.variables['Icryst'] = Icryst
        self.variables['xmin'] = xmin
        self.variables['xmax'] = xmax
        self.variables['ymin'] = ymin
        self.variables['ymax'] = ymax
        self.variables[intensitykey] = ipm
        self.variables['ipm-zero'] = ipm_zp
        self.logvars['Loss'] = loss
        self.logvars['Sparsifier'] = sparsifier
        self.logvars['Likelihood'] = likelihood
        self.miller_vars = ['DeltaF', 'SIGMA(DeltaF)', 'Hessian Diagonal', referencekey]

class deltafestimator_physical_gaussian(hkl_model):
    def __init__(self, df):
        self._build_graph(df)

    def _build_graph(self, df, referencekey='FCALC', intensitykey='ipm2'):
        tf.reset_default_graph()
        lp = tf.placeholder(tf.float32)
        self.hyperparameters['LAMBDA'] = lp

        indices = {
            'GAMMAINDEX'        : ['MERGEDH', 'MERGEDK', 'MERGEDL'],
            'RUNINDEX'          : 'RUN',
            'IMAGEINDEX'        : ['RUN', 'PHINUMBER', 'SERIES'],
            'PHIINDEX'          : ['RUN', 'PHINUMBER'],
        }

        for k,v in indices.items():
            df.loc[:,k] = df.groupby(v).ngroup()
        df

        #Prepare the per image metadata
        k = [i for i in df if 'ipm' in i.lower()]
        k += ['RUNINDEX']
        imagemetadata = df[k + ['IMAGEINDEX']].groupby('IMAGEINDEX').first()
        runidx = imagemetadata['RUNINDEX'].values

        #Construct pivot tables of intensities, errors, metadatakeys
        iobs        = df.pivot_table(values='IOBS', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'], columns='SERIES', fill_value=0)
        imagenumber = df.pivot_table(values='IMAGEINDEX', index=['H', 'K', 'L', 'RUNINDEX', 'PHIINDEX'], columns='SERIES', fill_value=0)
        sigma       = df.pivot_table(values='SIGMA(IOBS)', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'], columns='SERIES', fill_value=0)

        gammaidx = df.pivot_table(values='GAMMAINDEX', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'])
        gammaidx = np.array(gammaidx).flatten()
        Foff = df[['GAMMAINDEX', referencekey]].groupby('GAMMAINDEX').first().values.flatten()
        Foff_per_loss_term = Foff[gammaidx]
        Foff,Foff_per_loss_term = tf.constant(Foff, dtype=tf.float32), tf.constant(Foff_per_loss_term, dtype=tf.float32)

        ion    = iobs[[i for i in iobs if  'on' in i]].values
        ioff   = iobs[[i for i in iobs if 'off' in i]].values
        sigon  = sigma[[i for i in sigma if  'on' in i]].values**2
        sigoff = sigma[[i for i in sigma if 'off' in i]].values**2
        imon   = imagenumber[[i for i in imagenumber if  'on' in i]].values
        imoff  = imagenumber[[i for i in imagenumber if 'off' in i]].values

        #Compute inverse variance weights and normalize them to have rows sum to one
        invaron,invaroff = sigon.copy(),sigoff.copy()
        invaron[invaron > 0]    =  1/invaron[invaron > 0]
        invaroff[invaroff > 0]  = 1/invaroff[invaroff > 0]
        invaron  = invaron/invaron.sum(1)[:,None]
        invaroff = invaroff/invaroff.sum(1)[:,None]

        ion  = (ion*invaron).sum(1)
        ioff = (ioff*invaroff).sum(1)
        sigon  = (invaron**2*sigon).sum(1)
        sigoff = (invaroff**2*sigoff).sum(1)

        invaron  = tf.constant(invaron, dtype=tf.float32)
        invaroff = tf.constant(invaroff, dtype=tf.float32)

        ion     = tf.convert_to_tensor(ion, dtype=tf.float32, name='ion')
        ioff    = tf.convert_to_tensor(ioff, dtype=tf.float32, name='ioff')
        sigon   = tf.convert_to_tensor(sigon, dtype=tf.float32, name='sigon')
        sigoff  = tf.convert_to_tensor(sigoff, dtype=tf.float32, name='sigoff')

        #Problem Variables
        h = gammaidx.max() + 1
        r = runidx.max() + 1
        ipm  = tf.constant(np.float32(imagemetadata[intensitykey]))
        ipmx = tf.constant(np.float32(imagemetadata[intensitykey+'_xpos']))
        ipmy = tf.constant(np.float32(imagemetadata[intensitykey+'_ypos']))
        ipmy = tf.constant(np.float32(imagemetadata[intensitykey+'_ypos']))
        xstd = np.std(imagemetadata[intensitykey+'_xpos'])
        ystd = np.std(imagemetadata[intensitykey+'_ypos'])
        variables = tf.Variable(np.concatenate((
            np.ones(h, dtype=np.float32),
            -2*xstd*np.ones(r, dtype=np.float32),
             2*xstd*np.ones(r, dtype=np.float32),
            -2*ystd*np.ones(r, dtype=np.float32),
             2*ystd*np.ones(r, dtype=np.float32),
             np.zeros(1, dtype=np.float32),
        )))
        deltaF = variables[:h]
        xmin = variables[h:h+r]
        xmax = variables[h:h+r]
        ymin = variables[h+r:h+2*r]
        ymax = variables[h+2*r:h+3*r]
        ipm_zp = variables[-1]
        gammas = deltaF/Foff + 1 #This is only here for backward compatibility. We will not use it directly
        DF = tf.gather(deltaF, gammaidx)
        Icryst = (ipm + ipm_zp)*tf.nn.softplus(ipm * (
            tf.erf(tf.gather(xmax, runidx) - ipmx) - 
            tf.erf(tf.gather(xmin, runidx) - ipmx)
            ) * (
            tf.erf(tf.gather(ymax, runidx) - ipmy) - 
            tf.erf(tf.gather(ymin, runidx) - ipmy)
        ))

        Bon  = tf.reduce_sum(invaron*tf.gather(Icryst, imon), 1)
        Boff = tf.reduce_sum(invaroff*tf.gather(Icryst, imoff), 1)

        variance = ((DF/Foff_per_loss_term+ 1)*(Bon/Boff))**2*sigoff + sigon
        weights = variance**-1
        weights = weights/tf.reduce_sum(weights) #normalize the weights to one to stabilize regularizer tuning


        likelihood = tf.losses.mean_squared_error(
            ion,
            (DF/Foff_per_loss_term + 1)*(Bon/Boff)*ioff,
            weights=weights
        )

        sparsifier  = tf.losses.mean_squared_error(tf.zeros(deltaF.shape), deltaF)

        loss = (1. - lp)*likelihood + lp*sparsifier

        #print("6: {}".format(time() - start))
        self.H = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDH'], dtype=int)
        self.K = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDK'], dtype=int)
        self.L = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDL'], dtype=int)

        # Declare an iterator and tensor array loop variables for the gradients.
        n = array_ops.size(deltaF)
        loop_vars = [
            array_ops.constant(0, dtypes.int32),
            tensor_array_ops.TensorArray(gammas.dtype, n)
        ]    
        # Iterate over all elements of the gradient and compute second order
        # derivatives.
        gradients = tf.gradients(loss, deltaF)[0]
        gradients = array_ops.reshape(gradients, [-1])
        _, diag_A = control_flow_ops.while_loop(
            lambda j, _: j < n, 
            lambda j, result: (j + 1, 
                               result.write(j, tf.gather(tf.gradients(gradients[j], deltaF)[0], j))),
            loop_vars
        )    
        epsilon = 1e-32
        diag_A = array_ops.reshape(diag_A.stack(), [n])
        self.variables['DeltaF'] = deltaF
        self.variables['Hessian Diagonal'] = diag_A
        self.variables['SIGMA(DeltaF)'] = 1./(diag_A + epsilon)
        self.variables[referencekey] = Foff
        self.variables['Icryst'] = Icryst
        self.variables['xmin'] = xmin
        self.variables['xmax'] = xmax
        self.variables['ymin'] = ymin
        self.variables['ymax'] = ymax
        self.variables[intensitykey] = ipm
        self.variables['ipm-zero'] = ipm_zp
        self.logvars['Loss'] = loss
        self.logvars['Sparsifier'] = sparsifier
        self.logvars['Likelihood'] = likelihood
        self.miller_vars = ['DeltaF', 'SIGMA(DeltaF)', 'Hessian Diagonal', referencekey]

