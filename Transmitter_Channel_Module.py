# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
Transmitter, data_loading
Created on Wed Mar 01 23:00:14 2017
시뮬레이션을 위한 데이터를 생성한다. 
 - 변조(2,4,8PSK Differential Encoding) 방식에 따른 심볼 Generation
 - 채널 Distortion + AWGN
 - frequency and phase offset
 - 수신부 Simulation을 위한 데이터 loading (Training, Validation, Testing)
@author: Jang
--------------------------------------------------------------------------
"""
import numpy
import numpy as np 
from scipy import signal
from scipy.signal import lfilter
from matplotlib import pyplot as plt
#from matplotlib import pyplot as plt
import theano
import theano.tensor as T

# Class for Data generation, Digital Modulation, and Differential Encoding
class radio_transmitter(object):
    def __init__(self, n_data = 10, m_ary = 2 ) :

        self.n_data = n_data
        self.m_ary = m_ary 
        #self.phase_shift = (np.cos(np.pi/(m_ary))+1j*np.sin(np.pi/(m_ary)))       
        self.phase_shift = 1+0j
        # signal constellation
        self.mod_bpsk = np.array([1.0 + 0.0j,-1.0+0j])
        self.mod_bpsk = self.mod_bpsk*self.phase_shift
        self.mod_qpsk = np.array([1.0+0j, 0+1.0j, -1.0+0j, 0-1.0j])
        self.mod_qpsk = self.mod_qpsk*self.phase_shift
        self.mod_8psk = np.array([1.0+0j,1/np.sqrt(2)+1/np.sqrt(2)*1j, 0+1.0j, 
                                    -1/np.sqrt(2)+ 1/np.sqrt(2)*1j, 
                                    -1.0+0j, -1/np.sqrt(2) -1/np.sqrt(2)*1j,0-1.0j, 
                                    1/np.sqrt(2) -1/np.sqrt(2)*1j])
        self.mod_8psk = self.mod_8psk*self.phase_shift

    def trans_data(self, num_data = 0) : # real operation
        # random symbol generation
        a = np.random.randint(self.m_ary, size = self.n_data)
        return a.astype(np.int)
        
    def diff_encoding(self,a): # real operation
        # differential encoding
        b = np.zeros(self.n_data)
        b[0] = 0 # first value
        for i in range(self.n_data-1) : b[i+1] = (b[i]+a[i]) % self.m_ary
        return b.astype(np.int)
            
    def digital_mod(self, tr_symbol, m_ary = 2): # symbol mapping
        #digital modulation, signal constellation generation
        num_of_size = np.size(tr_symbol)
        print( '\n Number of Transmitted M-ary Symbol : %.2e' % num_of_size)
        print( ' Used Modulation : ', m_ary)
        if m_ary == 2 : modulated_symbol = [self.mod_bpsk[tr_symbol[i]] for i in range(num_of_size)]
        if m_ary == 4 : modulated_symbol = [self.mod_qpsk[tr_symbol[i]] for i in range(num_of_size)]
        if m_ary == 8 : modulated_symbol = [self.mod_8psk[tr_symbol[i]] for i in range(num_of_size)]
        return np.asarray(modulated_symbol)
        
    def diff_mod(self, mod_signal):
        b = 1 + 0j
        for i in range(np.size(mod_signal)):
            b = b*mod_signal[i]
            mod_signal[i] = b
        return mod_signal
                               
    def diff_decoding(self, detected_data): # real operation
        # differential decoding
        b = detected_data
        c = np.zeros(self.n_data)
        for i in range(self.n_data-1) : c[i] = (b[i+1]-b[i]) % self.m_ary
        return c.astype(np.int)        

    def diff_dmod(self, re_signal):
        diff = 1 # number of difference
        #differential demodulation
        diff_signal = re_signal[diff]*np.conjugate(re_signal[0])
        for ii in range(np.size(re_signal)-diff):
            diff_signal = np.append(diff_signal,re_signal[ii+diff]*np.conjugate(re_signal[ii]))
        diff_signal = np.delete(diff_signal, [0]) 
        return diff_signal
                                
    
# Channel Filtering and adding AWGN
class radio_channel(object): #complex operation
    def __init__(self, sigma = 0.1, m_ary = 2, freq = 0., phase = 0.,
                         ch_filter = np.array([0.0+0j, 1.0+0j, 0.0+0j]) ):
        self.sigma = sigma
        self.m_ary = m_ary
        self.channel = ch_filter
        self.freq = freq
        self.phase =  phase
 
    def channel_filter_noise(self, tr_signal): # 심볼단위로 처리한다.
        print( '\n Channel Filter : ', self.channel)
        print( ' frequency offset = %.2e %%' % (self.freq*100) )
        print( ' phase offset = %.2e (degree)' % (self.phase*180) )
        print( ' Signal to Noise Ratio : ', 10*np.log10(1./(self.sigma**2)), '\n')

        distorted_signal = np.convolve(tr_signal, self.channel) # channel filtering

        num_of_data = np.size(distorted_signal)
        # frequence & phase offset
        t = np.linspace(0, num_of_data,num_of_data)
        freq_phase_offset = np.cos(2*np.pi*self.freq*t+self.phase) \
            + 1j*np.sin(2*np.pi*self.freq*t+self.phase)   
        distorted_signal = distorted_signal*freq_phase_offset
        #add noise = np.random.randn(num_of_data)
        re_gau = np.random.normal(0,1,num_of_data)
        im_gau = np.random.normal(0,1,num_of_data)
        
        gau = (1./np.sqrt(2))*self.sigma*(re_gau + im_gau*1j)
        ch_signal = distorted_signal + gau
        #ch_signal = distorted_signal
        ch_signal = np.delete( ch_signal, range( np.int(np.size(self.channel/2))) )
        
        snr = 10*np.log10( (np.sum(distorted_signal.real**2 +distorted_signal.imag**2 ))
               / (np.sum(gau.real**2 +gau.imag**2)) )
        print( " Calculated Signal to Noise Ratio = ", snr)
        return ch_signal

# Data generation for receiver simulation (Training, Validating, and Testing)
class data_generating(object):
    def __init__(self, num_training=5, num_valid=5, num_test=5):

        self.num_training    = num_training   # of training sample
        self.num_valid       = num_valid   # of validation data (BER)
        self.num_test        = num_test   # of testing data (BER)
        
       
    def load_data(self, re_signal, tr_signal, num_input, num_output, borrow = True): # 1D array
        # num_input : number of Network input symbol
        # num_out : number of network output; number of class
        # Data set Generation for Receiver Processing
        
        print( '\n Number of Training data = %.1e'%  self.num_training )
        print( ' Number of validation data = %.1e'%  self.num_valid)
        print( ' Number of testing data = %.1e'% self.num_test, '\n')
        print( ' Length of input feature vector : ', num_input*2)
        print( ' Number of output class : ', num_output, '\n')          
        
        num_of_data = np.shape(re_signal)[0]
        data_set_x = np.zeros((num_of_data, 2*num_input), dtype=float)
        data_set_y = np.zeros(num_of_data)
        #x_signal = np.zeros((num_input,2), dtype=float)
        
        for ii in range(np.shape(re_signal)[0]-num_input*2): 
            x_signal = re_signal[ii:ii+num_input] #Equalizer signal block
            # 1. 블럭단위로 스택
            data_set_x[ii] = np.append(x_signal.real,x_signal.imag) # column-wise stacking
                        
           # 2. 심볼 단위로 데이터를 스택.. 그리고 .differentially detected signal 추가
           # for jj in range(num_input):
           #     data_set_x[ii,2*jj]   = x_signal.real[jj]
           #     data_set_x[ii,2*jj+1] = x_signal.imag[jj]
                
            data_set_y[ii] = tr_signal[np.int(ii+num_input/2)] #desired signal {0,..., m_ary-1}

        def shared_dataset(data_x, data_y, borrow=True):
            #print data_y
            shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')

        def non_shared_dataset(data_x, data_y):
            #print data_y
            non_shared_x = numpy.asarray(data_x, dtype=theano.config.floatX)
            non_shared_y = numpy.asarray(data_y, dtype=theano.config.floatX)
            return non_shared_x, non_shared_y.astype(int)

               
        train_set_x, train_set_y = non_shared_dataset(data_set_x[0:self.num_training], 
                 data_set_y[0:self.num_training])
        #print " testing"
        #print data_set_y[0], num_input/2
        #print train_set_x.get_value()[0,num_input/2]

        valid_set_x, valid_set_y = non_shared_dataset(data_set_x[self.num_training:self.num_training+self.num_valid], 
                 data_set_y[self.num_training:self.num_training+self.num_valid])
        #print data_set_y[self.num_training:self.num_training+self.num_valid]
        test_set_x, test_set_y = non_shared_dataset(data_set_x[self.num_training+self.num_valid:self.num_training+self.num_valid+self.num_test], 
                 data_set_y[self.num_training+self.num_valid:self.num_training+self.num_valid+self.num_test])
        #print data_set_y[self.num_training+self.num_valid:self.num_training+self.num_valid+self.num_test] 
        return_val = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                          (test_set_x, test_set_y)]
        return return_val
            

    def load_data_2d(self, re_signal, tr_signal, num_input, num_output, borrow = True): # for CNN input; 2D array
        # num_input : number of Network input symbol
        # num_out : number of network output; number of class
        # Data set Generation for Receiver Processing
        
        print( '\n Number of Training data = %.1e'%  self.num_training)
        print( ' Number of validation data = %.1e'%  self.num_valid)
        print( ' Number of testing data = %.1e'% self.num_test, '\n') 
        print( ' Length of input feature vector : ', num_input*2)
        print( ' Number of output class : ', num_output, '\n')          
        
        num_of_data = np.shape(re_signal)[0]
        data_set_x = np.zeros((num_of_data, 2, num_input), dtype=float)
        data_set_y = np.zeros(num_of_data)
        #x_signal = np.zeros((num_input,2), dtype=float)
        
        #print " d = ", tr_signal[0:5]
        #print " r = ",  re_signal[0:5]

        for ii in range(np.shape(re_signal)[0]-num_input): #시간이 많이 걸린다...
            x_signal = re_signal[ii:ii+num_input] #Equalizer signal block
            # 1. 블럭단위로 스택
            #data_set_x[ii] = np.append([x_signal.real,x_signal.imag]) # column-wise stacking
            data_set_x[ii, 0, :] = x_signal.real
            data_set_x[ii, 1, :] = x_signal.imag

            data_set_y[ii] = tr_signal[ii+num_input/2] #desired signal {0,..., m_ary-1}

        def shared_dataset(data_x, data_y, borrow=True):
            #print data_y
            shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')

        def non_shared_dataset(data_x, data_y):
            #print data_y
            non_shared_x = numpy.asarray(data_x, dtype=theano.config.floatX)
            non_shared_y = numpy.asarray(data_y, dtype=theano.config.floatX)
            return non_shared_x, non_shared_y.astype(int)

               
        train_set_x, train_set_y = non_shared_dataset(data_set_x[0:self.num_training], 
                 data_set_y[0:self.num_training])
        #print " testing"
        #print data_set_y[0], num_input/2
        #print train_set_x.get_value()[0,num_input/2]

        valid_set_x, valid_set_y = non_shared_dataset(data_set_x[self.num_training:self.num_training+self.num_valid], 
                 data_set_y[self.num_training:self.num_training+self.num_valid])
        #print data_set_y[self.num_training:self.num_training+self.num_valid]
        test_set_x, test_set_y = non_shared_dataset(data_set_x[self.num_training+self.num_valid:self.num_training+self.num_valid+self.num_test], 
                 data_set_y[self.num_training+self.num_valid:self.num_training+self.num_valid+self.num_test])
        #print data_set_y[self.num_training+self.num_valid:self.num_training+self.num_valid+self.num_test] 
        return_val = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                          (test_set_x, test_set_y)]
        return return_val
        

    def load_data_complex(self, re_signal, tr_signal, num_input, num_output, borrow = True): # for CNN input; 2D array
        # num_input : number of Network input symbol
        # num_out : number of network output; number of class
        # Data set Generation for Receiver Processing
        
        print( '\n Number of Training data = %.1e'%  self.num_training)
        print( ' Number of validation data = %.1e'%  self.num_valid)
        print( ' Number of testing data = %.1e'% self.num_test, '\n') 
        print( ' Length of input feature vector : ', num_input)
        print( ' Number of output class : ', num_output, '\n')          
        
        num_of_data = np.shape(re_signal)[0]
        data_set_x = np.zeros((num_of_data,num_input), dtype = complex)
        data_set_y = np.zeros(num_of_data)
        #x_signal = np.zeros((num_input,2), dtype=float)
        
        #print " d = ", tr_signal[0:5]
        #print " r = ",  re_signal[0:5]
        #print np.shape(tr_signal)[0]
        #print np.shape(re_signal)[0]

        for ii in range(np.shape(tr_signal)[0]-num_input): #시간이 많이 걸린다...
            x_signal = re_signal[ii:ii+num_input] #Equalizer signal block
            # 1. 블럭단위로 스택
            #data_set_x[ii] = np.append([x_signal.real,x_signal.imag]) # column-wise stacking
            data_set_x[ii] = x_signal
            #print data_set_x[ii]
            data_set_y[ii] = tr_signal[ii+np.int(num_input/2)] #desired signal {0,..., m_ary-1}


        def non_shared_dataset(data_x, data_y):
            #print data_y
            non_shared_x = numpy.asarray(data_x)
            non_shared_y = numpy.asarray(data_y)
            return non_shared_x, non_shared_y.astype(int)

               
        train_set_x, train_set_y = non_shared_dataset(data_set_x[0:self.num_training], 
                 data_set_y[0:self.num_training])
        #print " testing"
        #print data_set_y[0], num_input/2
        #print train_set_x.get_value()[0,num_input/2]

        valid_set_x, valid_set_y = non_shared_dataset(data_set_x[self.num_training:self.num_training+self.num_valid], 
                 data_set_y[self.num_training:self.num_training+self.num_valid])
        #print data_set_y[self.num_training:self.num_training+self.num_valid]
        test_set_x, test_set_y = non_shared_dataset(data_set_x[self.num_training+self.num_valid:self.num_training+self.num_valid+self.num_test], 
                 data_set_y[self.num_training+self.num_valid:self.num_training+self.num_valid+self.num_test])
        #print data_set_y[self.num_training+self.num_valid:self.num_training+self.num_valid+self.num_test] 
        return_val = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                          (test_set_x, test_set_y)]
        return return_val
        

        
class radio_receiver(object):

    def __init__(self, n_data = 10, m_ary = 2 ) :

        self.n_data = n_data
        self.m_ary = m_ary 
        #self.phase_shift = (np.cos(np.pi/(m_ary))+1j*np.sin(np.pi/(m_ary)))       
        self.phase_shift = 1+0j
        # signal constellation
        self.mod_bpsk = np.array([1.0 + 0.0j,-1.0+0j])
        self.mod_bpsk = self.mod_bpsk*self.phase_shift
        self.mod_qpsk = np.array([1.0+0j, 0.0+1.0j, -1.0+0j, 0.0-1.0j])
        #self.mod_qpsk = self.mod_qpsk*self.phase_shift
        self.mod_8psk = np.array([1.0+0j,1/np.sqrt(2)+1/np.sqrt(2)*1j, 0+1.0j, 
                                 -1/np.sqrt(2)+ 1/np.sqrt(2)*1j, 
                                 -1.0+0j, -1/np.sqrt(2) -1/np.sqrt(2)*1j,0-1.0j, 
                                 1/np.sqrt(2) -1/np.sqrt(2)*1j])
        self.mod_8psk = self.mod_8psk*self.phase_shift        
        
    def diff_decoding(self, detected_data): # real operation
        # differential decoding
        b = detected_data
        c = np.zeros(self.n_data)
        for i in range(self.n_data-1) : c[i] = (b[i+1]-b[i]) % self.m_ary
        return c.astype(np.int)        

    def diff_dmod(self, re_signal):
        diff = 1 # number of difference
        #differential demodulation
        diff_signal = re_signal[diff]*np.conjugate(re_signal[0])
        for ii in range(np.size(re_signal)-diff):
            diff_signal = np.append(diff_signal,re_signal[ii+diff]*np.conjugate(re_signal[ii]))
        diff_signal = np.delete(diff_signal, [0]) 
        return diff_signal
    
        
    def equ_input_signal(self, ff_signal, y_label, m_ary):
        num_label = np.size(y_label)
        #print('inside = ', y_label)
        y_label = y_label.astype(int) # convert non-integer into integer
        fb_signal = np.zeros(num_label*2)
        for ii in range(num_label):
            if m_ary == 2 :
                fb_signal[ii] = self.mod_bpsk[y_label[ii]].real
                fb_signal[num_label + ii] = self.mod_bpsk[y_label[ii]].imag
            if m_ary == 4 :
                fb_signal[ii] = self.mod_qpsk[y_label[ii]].real
                fb_signal[num_label + ii] = self.mod_qpsk[y_label[ii]].imag
            if m_ary == 8 :
                fb_signal[ii] = self.mod_8psk[y_label[ii]].real
                fb_signal[num_label + ii] = self.mod_8psk[y_label[ii]].imag
        ff_signal = np.append(ff_signal, fb_signal)
        num = np.size(ff_signal)
        ff_signal = ff_signal.reshape(1,num) # 디멘젼 관련해서 좀 더 따져 볼 것...
        return ff_signal

    # symbol decision; get label from estimated signal
    def sym_decision(self, est, m_ary, batch_size):        
        y = np.zeros(batch_size)
        if m_ary == 2: # for binary
            for ii in range(batch_size):
                if est[ii,0] < 0 : y[ii] = 1 
        
        if m_ary == 4: # for qpsk
           for ii in range(batch_size):
               c = (est[ii,0] + 1j*est[ii,1])*(np.cos(np.pi/4.) + 1j*np.cos(np.pi/4.))
               if (c.real > 0.) & (c.imag > 0.) : y[ii] = 0                   
               if (c.real > 0.) & (c.imag < 0.) : y[ii] = 3                   
               if (c.real < 0.) & (c.imag < 0.) : y[ii] = 2                   
               if (c.real < 0.) & (c.imag > 0.) : y[ii] = 1
        return y

    def sym_decision_complex(self, est, m_ary): 
        y = 0             
        if m_ary == 2: # for binary
            if est.real < 0 : y = 1 
            else : y = 0
        
        if m_ary == 4: # for qpsk
            c = est*(np.cos(np.pi/4.) + 1j*np.cos(np.pi/4.))
            if (c.real > 0.) & (c.imag > 0.) : y = 0                   
            if (c.real > 0.) & (c.imag < 0.) : y = 3                   
            if (c.real < 0.) & (c.imag < 0.) : y = 2                   
            if (c.real < 0.) & (c.imag > 0.) : y = 1
        return y        
        
    def label_to_constellation(self, label, m_ary):
        if m_ary == 2: return self.mod_bpsk[label]
        if m_ary == 4: return self.mod_qpsk[label]
        if m_ary == 8: return self.mod_8psk[label] 

        
        
def Data_Transmitting(num_of_data = 3000):
    print ("\n------------------------------------------------------------------- ")
    print ("\t\t Data Transmission & Data loading   ")
    print ("------------------------------------------------------------------- \n")
    #---------------------------------
    # initializing value (parameters)
    #---------------------------------
    n_data          = num_of_data # number of all data(training + validating + testing)
    num_training    = 300
    num_valid       = 300 
    num_test        = 300

    m_ary           = 4 # signal constellation
    n_ff            = 2 # feed forward; minimu m= 3
    n_fb            = 0 # of feedback; minimum = 0

    
    snr             = 10
    sigma = np.sqrt(np.power(10, -snr/10.))
    # sigma           = 0.01 # standard deviation
    #ch_filter = np.array([0.0+0j, 0.340+0.0j, 0.876+0j, 0.340+0.0j, 0.0+0j])
    ch_filter = np.array([0.0+0j, 0.0+0.0j, 1+0j, 0.0+0.0j, 0.0+0j])
    freq            = 0./2400 # frequency offset
    phase           = 0*np.pi # phase offset

    
    #Example of data generation
    #-------------------------------------------------------------------
    my_radio   = radio_transmitter(n_data, m_ary)
    my_channel = radio_channel(sigma, m_ary, freq, phase, ch_filter)
    my_load    = data_generating(num_training, num_valid, num_test )
    my_receiver = radio_receiver()
    #-------------------------------------------------------------------

    transmitted_data    = my_radio.trans_data(n_data) 
    #diff_encoded_data   = my_radio.diff_encoding(transmitted_data)
    modulated_signal    = my_radio.digital_mod(transmitted_data, m_ary)
    diff_modulated_symbol = my_radio.diff_mod(modulated_signal)
    
    #print " t = ", transmitted_data[0:5]
    #print " m = ",  modulated_symbol[0:5]

    ch_signal           = my_channel.channel_filter_noise(diff_modulated_symbol)
    diff_signal         = my_radio.diff_dmod(ch_signal)

    #print "d = ", diff_signal[0:5]
    #decoded_data        = my_radio.diff_decoding(diff_encoded_data)
    #-------------------------------------------------------------------

    tr_signal =  transmitted_data # transmitted symbol
    
    re_signal = ch_signal # received signal

    num_input = n_ff + n_fb # Neural Network input vector (심볼의 갯수, 복소수의 갯수)
    num_output = m_ary # Neural Network output vector
    
    # 데이터 로딩; 여기는 feedford 파트만 로딩...
    datasets = my_load.load_data(re_signal, tr_signal, n_ff, 
                                 m_ary, borrow = True)
    train_set_x, train_set_y = datasets[0]

    #---------------------------------
    # equalizer input vector 만들기
    x_in = train_set_x[0:3]
    re = x_in.reshape(x_in.shape[0], 1, 2, -1)
    print( x_in.shape, re.shape)
    
    datasets2 = my_load.load_data_complex(re_signal, tr_signal, n_ff, 
                                 m_ary, borrow = True)
    train_set_x2, train_set_y2 = datasets2[0]
    print (train_set_x2[0])
    
    #y_in = train_set_y[0:n_fb]
    #y = y_in
    #y = y_in.eval()
    #y = y.astype(int)
    #equ_input = my_receiver.equ_input_signal(x_in.eval(),y, m_ary)
    #equ_input = my_receiver.equ_input_signal(x_in,y, m_ary)
    #print(y)
    #print('equ input = ', equ_input)
    #---------------------------------------
    
    #print '\n desired label = ', train_set_y.eval()
    #print ' corresponding one train vector / input vector = \n\t', train_set_x.get_value() # T.cast를 사용했을 때는 값을 프린트 할 수 가 없다.
    #-------------------------------------------------------------------

    # ploting results
    plt_signal = ch_signal  # signal selection for plotting
    
    #signal constellation
    plt.figure(1)
    plt.title('Received Signal Constellation')
    plt.ylabel('Quadrature-phase')
    plt.xlabel('In-phase')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.grid()

    num_point =200
    start =300
    end = start + num_point
    y=np.zeros(num_point)
    plt.scatter(plt_signal.real[start:end],plt_signal.imag[start:end],10)
    print (np.shape(ch_signal) )

    plt.show(1)

    plt.figure(2)
    plt.title('channel signal')
    plt.ylabel('Value')
    plt.xlabel('# of sample')
    #plt.xlim(-2,2)
    plt.ylim(-2,2)
    
    num_sample = 120
    #x = np.linspace(1, num_sample, num_sample)
    x_axis = np.arange(np.size(plt_signal[100:num_sample]))
    plt.plot(x_axis,plt_signal.real[100:num_sample], marker='o', label ="I_phase" )
    #plt.bar(x_axis,ch_signal[100:num_sample,0], 0.02 )
    plt.plot(x_axis,plt_signal.imag[100:num_sample],marker='*', label ="Q_phase" )   
    #plt.bar(x_axis,ch_signal[100:num_sample,1], 0.02 )
    plt.legend(loc='lower right')
    plt.grid()
    plt.show(2)
    
    plt.figure(3)
    plt.title('channel signal')
    plt.ylabel('Value')
    plt.xlabel('# of sample')
    #plt.xlim(-2,2)
    plt.ylim(-2,2)
    
    num_sample = 120
    #x = np.linspace(1, num_sample, num_sample)
    x_axis = np.arange(np.size(plt_signal[100:num_sample]))
    plt.plot(x_axis,plt_signal.real[100:num_sample],linestyle="None",
             color = 'yellow', marker='o', label ="I_phase" )
    plt.plot(x_axis,plt_signal.imag[100:num_sample],linestyle="None",
             marker='*', label ="Q_phase" )
    plt.bar(x_axis,plt_signal.real[100:num_sample], 0.3, color = 'green' )
    plt.bar(x_axis,plt_signal.imag[100:num_sample], 0.3, color = 'yellow' )
    plt.legend(loc='lower right')
    plt.grid()
    plt.show(3)
    
    return ch_signal
    return equ_input
   
#------------------------------------------------------------------- 
if __name__ == '__main__':
    out = Data_Transmitting(1000)
    
    #my_receiver = radio_receiver()
    
    #x = np.array([[0,1], [0,-1]])
    #print( my_receiver.sym_decision(x,np.shape(x)[0], 4) )
    








