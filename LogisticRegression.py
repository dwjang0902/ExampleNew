"""
1. BPSK
2. Linear Equalization / DFE using Softmax Classification
3. Batch Processing 
@author: Jang

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy
import numpy as np 
from scipy import signal
from scipy.signal import lfilter
from matplotlib import pyplot as plt

import theano
import theano.tensor as T



class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.tensor._shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.tensor._shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1
        
        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
                

    def prediction_out(self): 
        return self.y_pred
        

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        #self.y = y
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class radio_tr(object):
    def __init__(self, num_of_data = 1000, sigma = 0.01, m_ary = 2):
        self.n_data = num_of_data
        self.sigma = sigma
        self.m_ary = m_ary
        # print ('number of data = %e' 'sigma = %e' 'M-ary = %d' ) % (self.n_data,self.sigma, self.m_ary = m_ary)
        

    def transmit_signal(self):
    # Signal(Random Binary Data) Generation
        a = np.random.randint(self.m_ary, size = self.n_data) # random integer(0,1)

        #Transmitter  Data transmission & Channel
        tr_signal = a 
        # convert data into transmitted signal(0,1) --> (-1,1)
        for i in range(np.size(a)):
            if tr_signal[i] == 0: tr_signal[i] = -1
        return tr_signal

    def channel_filter_noise(self, tr_signal):
        num_of_data = np.size(tr_signal)  
        #channel filtering block
        ch_filter = [0,0,0.5,1.0,0.5, 0.0] # channel filter
        distorted_signal = lfilter(ch_filter,1.0, tr_signal)
        #noise = np.random.randn(num_of_data)
        num_of_data = np.size(distorted_signal)      
        gau = np.random.normal(0,1,num_of_data) 
        ch_signal = tr_signal+0.1*self.sigma # adding Gaussian Noise
        ch_signal = distorted_signal #+ 0.1*gau # adding Gaussian Noise
        return ch_signal

#-----------------------------------------------------
def load_data(re_signal, tr_signal, num_input, num_output, borrow = True):

    #num_input       = 10    # input feature sizes
    #num_output      = 2     # of output clas; Binary classes
    num_training    =  10000   # of training sample
    num_valid       = 300000   # of validation data (BER)
    num_test        = 300000   # of testing data (BER)                     

    # Data set Generation for Receiver Processing
    num_of_data = np.size(re_signal)
    
    data_set_x = np.zeros((num_of_data, num_input))
    data_set_y = np.zeros(num_of_data)
    #data_set_x = ([[], []])
    #data_set_y = ([])

    x_signal = np.zeros(num_input, dtype=float)
    for ii in range(np.size(re_signal)-num_input):
        x_signal = re_signal[ii:ii+num_input] #Equalizer signal block
        data_set_x[ii] = x_signal

        d_signal = tr_signal[ii+num_input/4] #desired signal (-1, 1)    
        if d_signal == 1 : data_set_y[ii] = 1 
        else : data_set_y[ii] = 0 

    def shared_dataset(data_x, data_y, borrow=True):
       shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
       shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
       return shared_x, T.cast(shared_y, 'int32')
               
    train_set_x, train_set_y = shared_dataset(data_set_x[0:num_training], 
                 data_set_y[0:num_training])
    valid_set_x, valid_set_y = shared_dataset(data_set_x[num_training:num_training+num_valid], 
                 data_set_y[num_training:num_training+num_valid])
    test_set_x, test_set_y = shared_dataset(data_set_x[num_training+num_valid:num_training+num_valid+num_test], 
                 data_set_y[num_training+num_valid:num_training+num_valid+num_test])

    return_val = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return return_val
    

"""
-----------------------------------------------------------------
-----------------------------------------------------------------

"""          
def sgd_optimization_BPSK(learning_rate=0.5, n_epochs=1000, batch_size=10):

    num_of_data     = 1000000
    num_input       = 20    # input feature sizes
    num_output      = 2     # of output clas; Binary classes

    """
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    """
    
    """
    --------------------------------------------------------------------
    Data Transmission(binary dada), Channel Filtering & receiving Signal
    --------------------------------------------------------------------
    """
    
    # Radio object Instance        
    LogReg_Radio = radio_tr(num_of_data, sigma = 0.01, m_ary = 2)
    
    tr_signal = LogReg_Radio.transmit_signal()
    ch_signal = LogReg_Radio.channel_filter_noise(tr_signal)
        
    re_signal = ch_signal # received signal 
        
    #data loading
    datasets = load_data(re_signal, tr_signal, num_input, num_output, borrow = True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y   = datasets[2]
   
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0] // batch_size


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in = num_input, n_out = num_output)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)
    ber  = classifier.errors(y)    

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


if __name__ == '__main__':
    sgd_optimization_BPSK(learning_rate=0.3, n_epochs=1000, batch_size=10)