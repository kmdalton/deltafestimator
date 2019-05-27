import tensorflow as tf
import pandas as pd
import numpy as np
from sys import stdout
import tensorflow_probability as tfp
from scipy import sparse
from tensorflow.python.ops.parallel_for.gradients import jacobian
from tensorflow.contrib.layers import dense_to_sparse
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow import dtypes

pd.options.mode.chained_assignment = None


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
        loss,loss_ = self.log['loss'][-2:]
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
        self._loss = self.logvars['loss']
        self.result = {k:None for k in self.variables}

        #Convert values to an iterable if it isn't already
        hyperparameters = {k: np.array(v).flatten() for k,v in hyperparameters.items()}
        length = len(next(iter(hyperparameters.values())))
        self._sess = tf.Session(config=config)
        self._optim = optimizer if optimizer is not None else tf.train.AdamOptimizer(0.05)
        #self._optim = self._optim.minimize(self._loss, var_list=list(self.variables.values()))
        self._optim = self._optim.minimize(self._loss)

        for i in range(length):
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
        sparsifier  = lp*tf.losses.mean_squared_error(tf.zeros(DF.shape), DF)

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
        gradients = tf.gradients(loss, DF)[0]
        gradients = array_ops.reshape(gradients, [-1])
        _, diag_A = control_flow_ops.while_loop(
            lambda j, _: j < n, 
            lambda j, result: (j + 1, 
                               result.write(j, tf.gather(tf.gradients(gradients[j], DF)[0], j))),
            loop_vars
        )    
        diag_A = array_ops.reshape(diag_A.stack(), [n])
        epsilon = 1e-15
        self.variables['DeltaF'] = deltaF
        self.variables['SIGMA(DeltaF)'] = 1./(diag_A + epsilon)
        self.logvars['loss'] = loss
        self.logvars['Icryst'] = Icryst
        self.logvars['regularizer'] = regularizer
        self.logvars['sparsifier'] = sparsifier
        self.miller_vars = ['DeltaF', 'SIGMA(DeltaF)']


"""


def fit_model(df, rho, l, intensitykey='ipm2', referencekey='FCALC', optimizer=None, config=None, maxiter=1000, tolerance=1e-5, verbose=False):
    tf.reset_default_graph()

    rho = np.array(rho).flatten()
    l = np.array(l).flatten()

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

    ion    = iobs[[i for i in iobs if  'on' in i]].values
    ioff   = iobs[[i for i in iobs if 'off' in i]].values
    sigon  = sigma[[i for i in sigma if  'on' in i]].values
    sigoff = sigma[[i for i in sigma if 'off' in i]].values
    imon   = imagenumber[[i for i in imagenumber if  'on' in i]].values
    imoff  = imagenumber[[i for i in imagenumber if 'off' in i]].values
    cardon  = (ion  > 0).sum(1)
    cardoff = (ioff > 0).sum(1)

    ion     = tf.convert_to_tensor(ion, dtype=tf.float32, name='ion')
    ioff    = tf.convert_to_tensor(ioff, dtype=tf.float32, name='ioff')
    sigon   = tf.convert_to_tensor(sigon, dtype=tf.float32, name='sigon')
    sigoff  = tf.convert_to_tensor(sigoff, dtype=tf.float32, name='sigoff')
    cardon  = tf.convert_to_tensor(cardon, dtype=tf.float32, name='cardon')
    cardoff = tf.convert_to_tensor(cardoff, dtype=tf.float32, name='cardoff')

    #Problem Variables
    gammaidx = df.pivot_table(values='GAMMAINDEX', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'])
    gammaidx = np.array(gammaidx).flatten()
    Foff = tf.constant(df[['GAMMAINDEX', referencekey]].groupby('GAMMAINDEX').first().values.flatten(), tf.float32)
    h = gammaidx.max() + 1
    ipm = np.float32(imagemetadata[intensitykey])
    variables = tf.Variable(np.concatenate((np.ones(h, dtype=np.float32), ipm)))
    ipm = tf.constant(ipm)
    gammas = tf.nn.softplus(variables[:h])
    Icryst = tf.nn.softplus(variables[h:])

    Bon  = tf.gather(Icryst, imon)
    Boff = tf.gather(Icryst, imoff)

    n = int(Icryst.shape[0]) #Is this variable ever referenced? 

    variance = tf.math.reduce_sum((tf.gather(gammas, gammaidx)[:,None]*sigoff/Boff/cardoff[:,None])**2, 1) + \
               tf.math.reduce_sum((sigon/Bon/cardon[:,None])**2, 1) 


    likelihood = tf.losses.mean_squared_error(
        tf.reduce_sum(ioff/Boff/cardoff[:,None], 1)*tf.gather(gammas, gammaidx), 
        tf.reduce_sum(ion/Bon/cardon[:,None], 1), 
        weights=variance**-1
    )

    rhop = tf.placeholder(tf.float32)
    lp   = tf.placeholder(tf.float32)

    regularizer = rhop*tf.losses.mean_squared_error(ipm, Icryst)
    sparsifier  = lp*tf.losses.mean_squared_error(tf.zeros(gammas.shape), Foff*(gammas - 1))

    loss = likelihood + regularizer + sparsifier
    #Foff = np.ones(len(gammaidx))


    #print("6: {}".format(time() - start))
    H = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDH'], dtype=int)
    K = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDK'], dtype=int)
    L = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDL'], dtype=int)

    # Declare an iterator and tensor array loop variables for the gradients.
    n = array_ops.size(gammas)
    loop_vars = [
        array_ops.constant(0, dtypes.int32),
        tensor_array_ops.TensorArray(gammas.dtype, n)
    ]    
    # Iterate over all elements of the gradient and compute second order
    # derivatives.
    gradients = tf.gradients(loss, gammas)[0]
    gradients = array_ops.reshape(gradients, [-1])
    _, diag_A = control_flow_ops.while_loop(
        lambda j, _: j < n, 
        lambda j, result: (j + 1, 
                           result.write(j, tf.gather(tf.gradients(gradients[j], gammas)[0], j))),
        loop_vars
    )    
    diag_A = array_ops.reshape(diag_A.stack(), [n])

    #diag_A = tf.diag_part(tf.hessians(loss, gammas)[0])
    #diag_A = tfp.math.diag_jacobian(gammas, tfp.math.diag_jacobian(gammas, loss))
    #diag_A = [tf.gradients(tf.gradients(loss, gammas[i]), gammas[i]) for i in range(h)]
    #B = dense_to_sparse(jacobian(tf.gradients(loss, gammas)[0], Icryst, use_pfor=False))
    #C = dense_to_sparse(tf.hessians(loss, Icryst)[0])

    result = pd.DataFrame()
    intensities = pd.DataFrame()
    lossdf = pd.DataFrame()
    optimizer = tf.train.AdamOptimizer(0.05) if optimizer is None else optimizer
    optimizer = optimizer.minimize(loss)
    for iternumber,(rho_,l_) in enumerate(zip(rho, l)):
        losses = [[], [], [], []]
        feed_dict = {lp: l_, rhop: rho_}
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

            # Loop and a half problem, thanks mehran
            _, loss__, like, reg, sp = sess.run((optimizer, loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict)
            losses[0].append(loss__)
            losses[1].append(like)
            losses[2].append(reg)
            losses[3].append(sp)
            if np.isnan(loss__) or np.isinf(loss__):
                print("Initial Loss is NaN!!")
                break

            for i in range(maxiter):
                _, loss_ = sess.run((optimizer, loss), feed_dict=feed_dict)
                _, loss_, like, reg, sp = sess.run((optimizer, loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict)
                losses[0].append(loss_)
                losses[1].append(like)
                losses[2].append(reg)
                losses[3].append(sp)

                if np.isnan(loss__) or np.isinf(loss__):
                    print("Desired error not achieved due to precision loss. Exiting after {} iterations".format(i+1))
                    break

                #Absolute fractional change
                if np.abs(loss_ - loss__)/loss_ < tolerance:
                    if verbose:
                        percent = 100*(iternumber+1)/len(rho)
                        message = "Converged to tol={} after {} iterations. {} % complete...".format(tolerance, i, percent)
                        print(message, end='\r')
                    break

                loss__ = loss_

            #print(sess.run((loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict))
            gammas_, Icryst_, ipm_ = sess.run((gammas, Icryst, ipm), feed_dict=feed_dict)

            # These are the parts of the Hessian needed to compute error estimates
            diag_A_ = sess.run(diag_A, feed_dict=feed_dict)
        F = pd.DataFrame()
        F.loc[:,'H'] = H
        F.loc[:,'K'] = K
        F.loc[:,'L'] = L
        F.loc[:,'GAMMA'] = gammas_
        F = F.set_index(['H', 'K', 'L'])

        F.loc[:,'FCALC'] = np.array(df.groupby('GAMMAINDEX').first()['FCALC'], dtype=float)
        F.loc[:,'PHIC'] = np.array(df.groupby('GAMMAINDEX').first()['PHIC'], dtype=float)
        F.loc[:,'FOBS'] = np.array(df.groupby('GAMMAINDEX').first()['FOBS'], dtype=float)
        F.loc[:,'SIGMA(FOBS)'] = np.array(df.groupby('GAMMAINDEX').first()['SIGMA(FOBS)'], dtype=float)
        F.loc[:,'D'] = np.array(df.groupby('GAMMAINDEX').first()['D'], dtype=float)
        F.loc[:,'DeltaF'] = F[referencekey]*(F['GAMMA'] - 1)

        I = pd.DataFrame()
        I.loc[:,intensitykey] = ipm_
        I.loc[:,'Icryst'] = Icryst_

        sigmagamma = np.array(diag_A_)
        sigmagamma[sigmagamma > 0] = 1./sigmagamma[sigmagamma > 0]
        F.loc[:,'SIGMA(GAMMA)'] = sigmagamma
        F.loc[:,'SIGMA(DeltaF)'] = np.abs(0.5*sigmagamma*F[referencekey]*F['GAMMA']**-0.5)

        F.loc[:,'LAMBDA'] = l_
        F.loc[:,'RHO'] = rho_
        I.loc[:,'LAMBDA'] = l_
        I.loc[:,'RHO'] = rho_

        lossdf_ = pd.DataFrame()
        lossdf_.loc[:,'STEP']        = range(len(losses[0]))
        lossdf_.loc[:,'LOSS']        = losses[0]
        lossdf_.loc[:,'LIKELIHOOD']  = losses[1]
        lossdf_.loc[:,'REGULARIZER'] = losses[2]
        lossdf_.loc[:,'SPARSIFIER']  = losses[3]
        lossdf_.loc[:,'RHO']         = rho_
        lossdf_.loc[:,'LAMBDA']      = l_

        result = pd.concat((result, F))
        intensities = pd.concat((intensities, I))
        lossdf = pd.concat((lossdf, lossdf_))

    return result, intensities, lossdf




#TM
def gammastimator_classic(df, rho, l, intensitykey='ipm2', referencekey='FCALC', optimizer=None, config=None, maxiter=1000, tolerance=1e-5, verbose=False):
    tf.reset_default_graph()

    rho = np.array(rho).flatten()
    l = np.array(l).flatten()

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

    ion    = iobs[[i for i in iobs if  'on' in i]].values
    ioff   = iobs[[i for i in iobs if 'off' in i]].values
    sigon  = sigma[[i for i in sigma if  'on' in i]].values**2
    sigoff = sigma[[i for i in sigma if 'off' in i]].values**2
    imon   = imagenumber[[i for i in imagenumber if  'on' in i]].values
    imoff  = imagenumber[[i for i in imagenumber if 'off' in i]].values

    invaron,invaroff = sigon.copy(),sigoff.copy()
    invaron[invaron > 0]    =  1/invaron[invaron > 0]
    invaroff[invaroff > 0]  = 1/invaroff[invaroff > 0]
    ion  = np.average(ion, 1, invaron)
    ioff = np.average(ioff, 1, invaroff)
    sigon  = (invaron**2*sigon).sum(1)
    sigoff = (invaroff**2*sigoff).sum(1)


    ion     = tf.convert_to_tensor(ion, dtype=tf.float32, name='ion')
    ioff    = tf.convert_to_tensor(ioff, dtype=tf.float32, name='ioff')
    sigon   = tf.convert_to_tensor(sigon, dtype=tf.float32, name='sigon')
    sigoff  = tf.convert_to_tensor(sigoff, dtype=tf.float32, name='sigoff')

    #Problem Variables
    gammaidx = df.pivot_table(values='GAMMAINDEX', index=['H', 'K', 'L', 'RUNINDEX','PHIINDEX'])
    gammaidx = np.array(gammaidx).flatten()
    Foff = tf.constant(df[['GAMMAINDEX', referencekey]].groupby('GAMMAINDEX').first().values.flatten(), tf.float32)
    h = gammaidx.max() + 1
    ipm = np.float32(imagemetadata[intensitykey])
    variables = tf.Variable(np.concatenate((np.ones(h, dtype=np.float32), ipm)))
    ipm = tf.constant(ipm)
    gammas = tf.nn.softplus(variables[:h])
    G = tf.gather(gammas, gammaidx)
    Icryst = tf.nn.softplus(variables[h:])

    Bon  = tf.reduce_sum(tf.gather(Icryst, imon), 1)
    Boff = tf.reduce_sum(tf.gather(Icryst, imoff), 1)

    n = int(Icryst.shape[0]) #Is this variable ever referenced? 

    variance = (G*Bon/Boff)**2*sigoff + sigon
    weights = variance**-1
    weights = weights/tf.reduce_sum(weights)

    likelihood = tf.losses.mean_squared_error(
        G*Bon/Boff*ioff,
        ion,
        weights=weights
    )

    rhop = tf.placeholder(tf.float32)
    lp   = tf.placeholder(tf.float32)

    regularizer = rhop*tf.losses.mean_squared_error(ipm, Icryst)
    sparsifier  = lp*tf.losses.mean_squared_error(tf.zeros(gammas.shape), Foff*(gammas - 1))

    loss = likelihood + regularizer + sparsifier
    #Foff = np.ones(len(gammaidx))


    #print("6: {}".format(time() - start))
    H = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDH'], dtype=int)
    K = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDK'], dtype=int)
    L = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDL'], dtype=int)

    # Declare an iterator and tensor array loop variables for the gradients.
    n = array_ops.size(gammas)
    loop_vars = [
        array_ops.constant(0, dtypes.int32),
        tensor_array_ops.TensorArray(gammas.dtype, n)
    ]    
    # Iterate over all elements of the gradient and compute second order
    # derivatives.
    gradients = tf.gradients(loss, gammas)[0]
    gradients = array_ops.reshape(gradients, [-1])
    _, diag_A = control_flow_ops.while_loop(
        lambda j, _: j < n, 
        lambda j, result: (j + 1, 
                           result.write(j, tf.gather(tf.gradients(gradients[j], gammas)[0], j))),
        loop_vars
    )    
    diag_A = array_ops.reshape(diag_A.stack(), [n])

    #diag_A = tf.diag_part(tf.hessians(loss, gammas)[0])
    #diag_A = tfp.math.diag_jacobian(gammas, tfp.math.diag_jacobian(gammas, loss))
    #diag_A = [tf.gradients(tf.gradients(loss, gammas[i]), gammas[i]) for i in range(h)]
    #B = dense_to_sparse(jacobian(tf.gradients(loss, gammas)[0], Icryst, use_pfor=False))
    #C = dense_to_sparse(tf.hessians(loss, Icryst)[0])

    result = pd.DataFrame()
    intensities = pd.DataFrame()
    lossdf = pd.DataFrame()
    optimizer = tf.train.AdamOptimizer(0.05) if optimizer is None else optimizer
    optimizer = optimizer.minimize(loss)
    for iternumber,(rho_,l_) in enumerate(zip(rho, l)):
        losses = [[], [], [], []]
        feed_dict = {lp: l_, rhop: rho_}
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

            # Loop and a half problem, thanks mehran
            _, loss__, like, reg, sp = sess.run((optimizer, loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict)
            losses[0].append(loss__)
            losses[1].append(like)
            losses[2].append(reg)
            losses[3].append(sp)
            if np.isnan(loss__) or np.isinf(loss__):
                print("Initial Loss is NaN!!")
                break

            for i in range(maxiter):
                _, loss_ = sess.run((optimizer, loss), feed_dict=feed_dict)
                _, loss_, like, reg, sp = sess.run((optimizer, loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict)
                losses[0].append(loss_)
                losses[1].append(like)
                losses[2].append(reg)
                losses[3].append(sp)

                if np.isnan(loss__) or np.isinf(loss__):
                    print("Desired error not achieved due to precision loss. Exiting after {} iterations".format(i+1))
                    break

                #Absolute fractional change
                if np.abs(loss_ - loss__)/loss_ < tolerance:
                    if verbose:
                        percent = 100*(iternumber+1)/len(rho)
                        message = "Converged to tol={} after {} iterations. {} % complete...".format(tolerance, i, percent)
                        print(message, end='\r')
                    break

                loss__ = loss_

            #print(sess.run((loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict))
            gammas_, Icryst_, ipm_ = sess.run((gammas, Icryst, ipm), feed_dict=feed_dict)

            # These are the parts of the Hessian needed to compute error estimates
            diag_A_ = sess.run(diag_A, feed_dict=feed_dict)
        F = pd.DataFrame()
        F.loc[:,'H'] = H
        F.loc[:,'K'] = K
        F.loc[:,'L'] = L
        F.loc[:,'GAMMA'] = gammas_
        F = F.set_index(['H', 'K', 'L'])

        F.loc[:,'FCALC'] = np.array(df.groupby('GAMMAINDEX').first()['FCALC'], dtype=float)
        F.loc[:,'PHIC'] = np.array(df.groupby('GAMMAINDEX').first()['PHIC'], dtype=float)
        F.loc[:,'FOBS'] = np.array(df.groupby('GAMMAINDEX').first()['FOBS'], dtype=float)
        F.loc[:,'SIGMA(FOBS)'] = np.array(df.groupby('GAMMAINDEX').first()['SIGMA(FOBS)'], dtype=float)
        F.loc[:,'D'] = np.array(df.groupby('GAMMAINDEX').first()['D'], dtype=float)
        F.loc[:,'DeltaF'] = F[referencekey]*(F['GAMMA'] - 1)

        I = pd.DataFrame()
        I.loc[:,intensitykey] = ipm_
        I.loc[:,'Icryst'] = Icryst_

        sigmagamma = np.array(diag_A_)
        sigmagamma[sigmagamma > 0] = 1./sigmagamma[sigmagamma > 0]
        F.loc[:,'SIGMA(GAMMA)'] = sigmagamma
        F.loc[:,'SIGMA(DeltaF)'] = np.abs(0.5*sigmagamma*F[referencekey]*F['GAMMA']**-0.5)

        F.loc[:,'LAMBDA'] = l_
        F.loc[:,'RHO'] = rho_
        I.loc[:,'LAMBDA'] = l_
        I.loc[:,'RHO'] = rho_

        lossdf_ = pd.DataFrame()
        lossdf_.loc[:,'STEP']        = range(len(losses[0]))
        lossdf_.loc[:,'LOSS']        = losses[0]
        lossdf_.loc[:,'LIKELIHOOD']  = losses[1]
        lossdf_.loc[:,'REGULARIZER'] = losses[2]
        lossdf_.loc[:,'SPARSIFIER']  = losses[3]
        lossdf_.loc[:,'RHO']         = rho_
        lossdf_.loc[:,'LAMBDA']      = l_

        result = pd.concat((result, F))
        intensities = pd.concat((intensities, I))
        lossdf = pd.concat((lossdf, lossdf_))

    return result, intensities, lossdf


def deltafestimator(df, rho, l, intensitykey='ipm2', referencekey='FCALC', optimizer=None, config=None, maxiter=1000, tolerance=1e-5, verbose=False):
    tf.reset_default_graph()

    rho = np.array(rho).flatten()
    l = np.array(l).flatten()

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
    #Bon  = tf.reduce_mean(tf.gather(Icryst, imon), 1)
    #Boff = tf.reduce_mean(tf.gather(Icryst, imoff), 1)

    n = int(Icryst.shape[0]) #Is this variable ever referenced? 

    variance = ((DF/Foff_per_loss_term+ 1)*(Bon/Boff))**2*sigoff + sigon
    weights = variance**-1
    weights = weights/tf.reduce_sum(weights) #normalize the weights to one to stabilize regularizer tuning


    likelihood = tf.losses.mean_squared_error(
        ion,
        (DF/Foff_per_loss_term + 1)*(Bon/Boff)*ioff,
        weights=weights
    )

    rhop = tf.placeholder(tf.float32)
    lp   = tf.placeholder(tf.float32)

    regularizer = rhop*tf.losses.mean_squared_error(ipm, Icryst)
    sparsifier  = lp*tf.losses.mean_squared_error(tf.zeros(DF.shape), DF)

    loss = likelihood + regularizer + sparsifier

    #print("6: {}".format(time() - start))
    H = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDH'], dtype=int)
    K = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDK'], dtype=int)
    L = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDL'], dtype=int)

    # Declare an iterator and tensor array loop variables for the gradients.
    n = array_ops.size(gammas)
    loop_vars = [
        array_ops.constant(0, dtypes.int32),
        tensor_array_ops.TensorArray(gammas.dtype, n)
    ]    
    # Iterate over all elements of the gradient and compute second order
    # derivatives.
    gradients = tf.gradients(loss, DF)[0]
    gradients = array_ops.reshape(gradients, [-1])
    _, diag_A = control_flow_ops.while_loop(
        lambda j, _: j < n, 
        lambda j, result: (j + 1, 
                           result.write(j, tf.gather(tf.gradients(gradients[j], DF)[0], j))),
        loop_vars
    )    
    diag_A = array_ops.reshape(diag_A.stack(), [n])

    #diag_A = tf.diag_part(tf.hessians(loss, gammas)[0])
    #diag_A = tfp.math.diag_jacobian(gammas, tfp.math.diag_jacobian(gammas, loss))
    #diag_A = [tf.gradients(tf.gradients(loss, gammas[i]), gammas[i]) for i in range(h)]
    #B = dense_to_sparse(jacobian(tf.gradients(loss, gammas)[0], Icryst, use_pfor=False))
    #C = dense_to_sparse(tf.hessians(loss, Icryst)[0])

    result = pd.DataFrame()
    intensities = pd.DataFrame()
    lossdf = pd.DataFrame()
    optimizer = tf.train.AdamOptimizer(0.05) if optimizer is None else optimizer
    optimizer = optimizer.minimize(loss)
    for iternumber,(rho_,l_) in enumerate(zip(rho, l)):
        losses = [[], [], [], []]
        feed_dict = {lp: l_, rhop: rho_}
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

            # Loop and a half problem, thanks mehran
            _, loss__, like, reg, sp = sess.run((optimizer, loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict)
            losses[0].append(loss__)
            losses[1].append(like)
            losses[2].append(reg)
            losses[3].append(sp)
            if np.isnan(loss__) or np.isinf(loss__):
                print("Initial Loss is NaN!!")
                break

            for i in range(maxiter):
                _, loss_ = sess.run((optimizer, loss), feed_dict=feed_dict)
                _, loss_, like, reg, sp = sess.run((optimizer, loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict)
                losses[0].append(loss_)
                losses[1].append(like)
                losses[2].append(reg)
                losses[3].append(sp)

                if np.isnan(loss__) or np.isinf(loss__):
                    print("Desired error not achieved due to precision loss. Exiting after {} iterations".format(i+1))
                    break

                #Absolute fractional change
                if np.abs(loss_ - loss__)/loss_ < tolerance:
                    if verbose:
                        percent = 100*(iternumber+1)/len(rho)
                        message = "Converged to tol={} after {} iterations. {} % complete...".format(tolerance, i, percent)
                        print(message, end='\r')
                    break

                loss__ = loss_

            #print(sess.run((loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict))
            gammas_, deltaf_, Icryst_, ipm_ = sess.run((gammas, deltaF, Icryst, ipm), feed_dict=feed_dict)

            # These are the parts of the Hessian needed to compute error estimates
            diag_A_ = sess.run(diag_A, feed_dict=feed_dict)
        F = pd.DataFrame()
        F.loc[:,'H'] = H
        F.loc[:,'K'] = K
        F.loc[:,'L'] = L
        F.loc[:,'GAMMA'] = gammas_
        F = F.set_index(['H', 'K', 'L'])

        F.loc[:,'FCALC'] = np.array(df.groupby('GAMMAINDEX').first()['FCALC'], dtype=float)
        F.loc[:,'PHIC'] = np.array(df.groupby('GAMMAINDEX').first()['PHIC'], dtype=float)
        F.loc[:,'FOBS'] = np.array(df.groupby('GAMMAINDEX').first()['FOBS'], dtype=float)
        F.loc[:,'SIGMA(FOBS)'] = np.array(df.groupby('GAMMAINDEX').first()['SIGMA(FOBS)'], dtype=float)
        F.loc[:,'D'] = np.array(df.groupby('GAMMAINDEX').first()['D'], dtype=float)
        F.loc[:,'DeltaF'] = deltaf_

        I = pd.DataFrame()
        I.loc[:,intensitykey] = ipm_
        I.loc[:,'Icryst'] = Icryst_

        sigmadeltaf = np.array(diag_A_)
        sigmadeltaf[sigmadeltaf > 0] = 1./sigmadeltaf[sigmadeltaf > 0]
        F.loc[:,'SIGMA(DeltaF)'] = sigmadeltaf

        F.loc[:,'LAMBDA'] = l_
        F.loc[:,'RHO'] = rho_
        I.loc[:,'LAMBDA'] = l_
        I.loc[:,'RHO'] = rho_

        lossdf_ = pd.DataFrame()
        lossdf_.loc[:,'STEP']        = range(len(losses[0]))
        lossdf_.loc[:,'LOSS']        = losses[0]
        lossdf_.loc[:,'LIKELIHOOD']  = losses[1]
        lossdf_.loc[:,'REGULARIZER'] = losses[2]
        lossdf_.loc[:,'SPARSIFIER']  = losses[3]
        lossdf_.loc[:,'RHO']         = rho_
        lossdf_.loc[:,'LAMBDA']      = l_

        result = pd.concat((result, F))
        intensities = pd.concat((intensities, I))
        lossdf = pd.concat((lossdf, lossdf_))

    return result, intensities, lossdf



def gammastimator2(df, rho, l, intensitykey='ipm2', referencekey='FCALC', optimizer=None, config=None, maxiter=1000, tolerance=1e-5, verbose=False):
    tf.reset_default_graph()

    rho = np.array(rho).flatten()
    l = np.array(l).flatten()

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
    Foff = tf.constant(Foff, dtype=tf.float32)

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
    gammas = tf.nn.softplus(variables[:h])
    deltaF = Foff*(1 - gammas)
    G = tf.gather(gammas, gammaidx)
    Icryst = tf.nn.softplus(variables[h:])

    Bon  = tf.reduce_sum(invaron*tf.gather(Icryst, imon), 1)
    Boff = tf.reduce_sum(invaroff*tf.gather(Icryst, imoff), 1)
    #Bon  = tf.reduce_mean(tf.gather(Icryst, imon), 1)
    #Boff = tf.reduce_mean(tf.gather(Icryst, imoff), 1)

    #n = int(Icryst.shape[0]) #Is this variable ever referenced? 

    variance = sigon + sigoff*(G*Bon/Boff)**2
    weights = variance**-1
    weights = weights/tf.reduce_sum(weights) #normalize the weights to one to stabilize regularizer tuning

    likelihood = tf.losses.mean_squared_error(
        ion,
        G*(Bon/Boff)*ioff,
        weights=weights
    )

    rhop = tf.placeholder(tf.float32)
    lp   = tf.placeholder(tf.float32)

    regularizer = rhop*tf.losses.mean_squared_error(ipm, Icryst)
    sparsifier  = lp*tf.losses.mean_squared_error(tf.ones(G.shape), G)

    loss = likelihood + regularizer + sparsifier

    #print("6: {}".format(time() - start))
    H = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDH'], dtype=int)
    K = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDK'], dtype=int)
    L = np.array(df.groupby('GAMMAINDEX').mean()['MERGEDL'], dtype=int)

    # Declare an iterator and tensor array loop variables for the gradients.
    n = array_ops.size(gammas)
    loop_vars = [
        array_ops.constant(0, dtypes.int32),
        tensor_array_ops.TensorArray(gammas.dtype, n)
    ]    
    # Iterate over all elements of the gradient and compute second order
    # derivatives.
    gradients = tf.gradients(loss, gammas)[0]
    gradients = array_ops.reshape(gradients, [-1])
    _, diag_A = control_flow_ops.while_loop(
        lambda j, _: j < n, 
        lambda j, result: (j + 1, 
                           result.write(j, tf.gather(tf.gradients(gradients[j], gammas)[0], j))),
        loop_vars
    )    
    diag_A = array_ops.reshape(diag_A.stack(), [n])

    #diag_A = tf.diag_part(tf.hessians(loss, gammas)[0])
    #diag_A = tfp.math.diag_jacobian(gammas, tfp.math.diag_jacobian(gammas, loss))
    #diag_A = [tf.gradients(tf.gradients(loss, gammas[i]), gammas[i]) for i in range(h)]
    #B = dense_to_sparse(jacobian(tf.gradients(loss, gammas)[0], Icryst, use_pfor=False))
    #C = dense_to_sparse(tf.hessians(loss, Icryst)[0])

    result = pd.DataFrame()
    intensities = pd.DataFrame()
    lossdf = pd.DataFrame()
    optimizer = tf.train.AdamOptimizer(0.05) if optimizer is None else optimizer
    optimizer = optimizer.minimize(loss)
    for iternumber,(rho_,l_) in enumerate(zip(rho, l)):
        losses = [[], [], [], []]
        feed_dict = {lp: l_, rhop: rho_}
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

            # Loop and a half problem, thanks mehran
            _, loss__, like, reg, sp = sess.run((optimizer, loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict)
            losses[0].append(loss__)
            losses[1].append(like)
            losses[2].append(reg)
            losses[3].append(sp)
            if np.isnan(loss__) or np.isinf(loss__):
                print("Initial Loss is NaN!!")
                break

            for i in range(maxiter):
                _, loss_ = sess.run((optimizer, loss), feed_dict=feed_dict)
                _, loss_, like, reg, sp = sess.run((optimizer, loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict)
                losses[0].append(loss_)
                losses[1].append(like)
                losses[2].append(reg)
                losses[3].append(sp)

                if np.isnan(loss__) or np.isinf(loss__):
                    print("Desired error not achieved due to precision loss. Exiting after {} iterations".format(i+1))
                    break

                #Absolute fractional change
                if np.abs(loss_ - loss__)/loss_ < tolerance:
                    if verbose:
                        percent = 100*(iternumber+1)/len(rho)
                        message = "Converged to tol={} after {} iterations. {} % complete...".format(tolerance, i, percent)
                        print(message, end='\r')
                    break

                loss__ = loss_

            #print(sess.run((loss, likelihood, regularizer, sparsifier), feed_dict=feed_dict))
            gammas_, deltaf_, Icryst_, ipm_ = sess.run((gammas, deltaF, Icryst, ipm), feed_dict=feed_dict)

            # These are the parts of the Hessian needed to compute error estimates
            diag_A_ = sess.run(diag_A, feed_dict=feed_dict)
        F = pd.DataFrame()
        F.loc[:,'H'] = H
        F.loc[:,'K'] = K
        F.loc[:,'L'] = L
        F.loc[:,'GAMMA'] = gammas_
        F = F.set_index(['H', 'K', 'L'])

        F.loc[:,'FCALC'] = np.array(df.groupby('GAMMAINDEX').first()['FCALC'], dtype=float)
        F.loc[:,'PHIC'] = np.array(df.groupby('GAMMAINDEX').first()['PHIC'], dtype=float)
        F.loc[:,'FOBS'] = np.array(df.groupby('GAMMAINDEX').first()['FOBS'], dtype=float)
        F.loc[:,'SIGMA(FOBS)'] = np.array(df.groupby('GAMMAINDEX').first()['SIGMA(FOBS)'], dtype=float)
        F.loc[:,'D'] = np.array(df.groupby('GAMMAINDEX').first()['D'], dtype=float)
        F.loc[:,'DeltaF'] = F[referencekey]*(F['GAMMA']-1)

        I = pd.DataFrame()
        I.loc[:,intensitykey] = ipm_
        I.loc[:,'Icryst'] = Icryst_

        sigmagammas = np.array(diag_A_)
        sigmagammas[sigmagammas > 0] = 1./sigmagammas[sigmagammas > 0]
        F.loc[:,'SIGMA(GAMMA)'] = sigmagammas
        F.loc[:,'SIGMA(DeltaF)'] = F[referencekey]**2*(F['SIGMA(GAMMA)'])

        F.loc[:,'LAMBDA'] = l_
        F.loc[:,'RHO'] = rho_
        I.loc[:,'LAMBDA'] = l_
        I.loc[:,'RHO'] = rho_

        lossdf_ = pd.DataFrame()
        lossdf_.loc[:,'STEP']        = range(len(losses[0]))
        lossdf_.loc[:,'LOSS']        = losses[0]
        lossdf_.loc[:,'LIKELIHOOD']  = losses[1]
        lossdf_.loc[:,'REGULARIZER'] = losses[2]
        lossdf_.loc[:,'SPARSIFIER']  = losses[3]
        lossdf_.loc[:,'RHO']         = rho_
        lossdf_.loc[:,'LAMBDA']      = l_

        result = pd.concat((result, F))
        intensities = pd.concat((intensities, I))
        lossdf = pd.concat((lossdf, lossdf_))

    return result, intensities, lossdf





"""
