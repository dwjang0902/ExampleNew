# -*- coding: utf-8 -*-
"""
Created on Wed Apr. 25 13:56:00 2017
1. BPSK
2. Linear Equalization
3. Complex Softmax Regression / Classification 사용
@author: Jang
"""

from __future__ import print_function
__docformat__ = 'restructedtext en'

import os
import sys
import gc

gc.enable()
gc.collect()
sys.setrecursionlimit(300)

sys.path.insert(0, 'E:\\SABATICS-GA TECH\\REPORT\\PythonPrograms\\1.Signal Generation')

import timeit
from scipy.signal import lfilter
from matplotlib import pyplot as plt
import numpy
import numpy as np
import theano
import theano.tensor as T
#from Transmitter_Channel import radio_transmitter, radio_channel, data_generating
from Transmitter_Channel_Module import radio_transmitter, radio_channel
from Transmitter_Channel_Module import data_generating, radio_receiver

#----------------------------------------------------------------
# Signal generation
#----------------------------------------------------------------
def sim_data_gen(snr_in, n_ff, n_fb, m_ary, num_of_data = 100000):
    #---------------------------------
    # 1. initializing value
    #---------------------------------
    n_data          = num_of_data # number of all data(training + validating + testing)
    num_train       = 90000
    num_valid       = 100
    num_test        = 100
    if(num_of_data < num_train + num_valid + num_test):
        print (" Error : too small data generation ---- ")
    m_ary           = m_ary # signal constellation
    n_ff            = n_ff # feed forward, input feature vector / 2 
    n_fb            = 0
    num_input       = (n_ff + n_fb) # Neural Network input vector 
    num_output      = m_ary # Neural Network output vector
    snr             = snr_in
    sigma = np.sqrt(np.power(10, -snr/10.))
    #sigma           = 0.1 # standard deviation
    #ch_filter = np.array([0.0+0j, 0.340+0.0j, 0.876+0j, 0.340+0.0j, 0.0+0j])
    #ch_filter = np.array([0.0+0j, 0.10+0.0j, 0.876+0j, 0.10+0.0j, 0.0+0j])
    ch_filter = np.array([0.0+0j, 0.0+0.0j, 1.+0j, 0.0+0.0j, 0.0+0j])
    freq            = 0./2400 # frequency offset
    phase           = np.pi*(0./4) # phase offset
    #n_fb           = 10 # feedback

    #-------------------------------------------------------------------
    # 2. Object Instance
    #-------------------------------------------------------------------
    print(" System construction for Data Generation")
    my_radio   = radio_transmitter(n_data, m_ary)
    my_channel = radio_channel(sigma, m_ary, ch_filter)
    my_channel = radio_channel(sigma, m_ary, freq, phase, ch_filter)
    my_load    = data_generating(num_train, num_valid, num_test)
    my_receiver = radio_receiver()

        
    #-------------------------------------------------------------------
    # 3. Data....Generating
    #-------------------------------------------------------------------
    print(" Data generation - Start")
    transmitted_data    = my_radio.trans_data(n_data) 
    diff_encoded_data   = my_radio.diff_encoding(transmitted_data)
    modulated_signal    = my_radio.digital_mod(transmitted_data, m_ary)
    #diff_modulated_symbol = my_radio.diff_mod(modulated_signal)
    ch_signal           = my_channel.channel_filter_noise(modulated_signal)
    diff_signal         = my_radio.diff_dmod(ch_signal)
    #decoded_data        = my_radio.diff_decoding(diff_encoded_data)

    tr_signal = transmitted_data
    re_signal =  ch_signal 
    
    datasets = my_load.load_data_complex(re_signal, tr_signal, n_ff, 
                    num_output, borrow = True)
    print( " Data Generation End ")
    return datasets 

#------------------------------
# Adaptive Equalization Block by Logistic Regression
#------------------------------

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    e_x = np.exp(x)
    return e_x / e_x.sum()

def softmax_equ(datasets, m_ary , n_ff, n_fb, 
            learning_rate, num_training_mode ):
    print(" Equalization by softmax --- Start ")
    num_output = m_ary
    num_input = n_ff + n_fb
    num_training = num_training_mode
    num_weights = (num_output, num_input)
    step_size = learning_rate

    x_signal = np.zeros(num_input, dtype = complex)
    tap_weights = np.zeros(num_weights, dtype = complex)
    w = np.zeros(num_weights, dtype = complex)
    del_w = np.zeros(num_weights, dtype = complex)
    z = np.zeros(num_output, dtype = complex)
    p = np.zeros(num_output, dtype = complex)
    t_m_p = np.zeros(num_output, dtype = complex)
    t = np.zeros((num_output, num_output), dtype = complex)    
    for ii in range(num_output) : t[ii,ii] = 1+1j #??????

    est_error = np.zeros(1, dtype = complex)
    #desired_label = np.zeros(1)
    
    detected_signal = []
    est_error_signal =[]

    my_receiver = radio_receiver()
        
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y   = datasets[2]

    print ("Received Signal Constellation")
    plt.figure(1)
    plt.title('Signal Constellation')
    plt.ylabel('quadrature phase')
    plt.xlabel('inphase')
    plt.ylim(-2,2)
    plt.xlim(-2,2)

    #--------------------------------- Equalization
    num_point = 500
    start =5000
    end = start + num_point
    y=np.zeros(num_point)
    plt.scatter(train_set_x.real[start:end],train_set_x.imag[start:end],20)
    plt.show(1) 
    
    pre_d_signal = 0+0j 
    del_t = 0+0j
    error = 0
    
    for ii in range(np.size(train_set_y)):
        x_signal = train_set_x[ii] # feedforward TDL
        x_signal[0] = 1+1j #decision feedback
        x_signal[-1] = pre_d_signal #decision feedback
        desired_label = train_set_y[ii] # training mode, desired_label
        # label to signal
        d_signal = my_receiver.label_to_constellation(train_set_y[ii], m_ary)

        z = w.dot(x_signal) # linear output
        #z = T.nnet.sigmoid(z.real) + 1j*T.nnet.sigmoid(z.imag)
        #z = np.tanh(z.real) + 1j*np.tanh(z.imag)

        p = softmax(z.real) +1j*softmax(z.imag) # softmax output
        p = p
        
        detected_signal =  np.append(detected_signal, p) #just for plotting 
        
        detected_data = np.argmax(p.real)   # decision making
        #detected_data = np.argmax(p.real + p.imag)  # decision making

        if ii > num_training: # decision directed mode 
            # label to signal           
            d_signal  = my_receiver.label_to_constellation(detected_data, m_ary)
            desired_label = detected_data

        del_t = t[desired_label] - p # t-p
        #del_w = np.outer(del_t, np.conjugate(x_signal))
        for jj in range(num_output):
            for kk in range(num_input) :
                del_w[jj,kk] = del_t[jj]*np.conjugate(x_signal[kk])
        w = w +  step_size*del_w
        
        pre_d_signal = d_signal # feedback signal
        
        est_error = d_signal - z[desired_label] / np.sqrt(z.real[desired_label]**2 +z.imag[desired_label]**2)  # calculate error signal        
        est_error_signal = np.append(est_error_signal, \
                                           est_error*np.conjugate(est_error))
        if ii > num_training: # decision directed mode 
            if train_set_y[ii] != detected_data:
                error += 1
        #------------------------------------- Equalization
    print(" Equalization --- End ") 
    print (" Now Checking ")    
    #print(detected_signal)    

    #signal constellation Plotting
    plt.figure(1)
    plt.title('Softmax Constellation')
    plt.ylabel('quadrature phase')
    plt.xlabel('inphase')
    plt.ylim(0,1)
    plt.xlim(0,1)

    num_point = 1000
    start =20000
    end = start + num_point
    y=np.zeros(num_point)
    plt.scatter(detected_signal.real[start:end], \
                                      detected_signal.imag[start:end],20)
    plt.show(1)

    # learning Curve Plotting
    plt.figure(2)
    plt.title('Learning Curve')
    plt.ylabel('MSE')
    plt.xlabel('# of sample')

    num_sample = 3000
    x = np.linspace(1, num_sample, num_sample)
    #plt.plot(x,10*np.log10(est_error_signal[0:num_sample]+0.0001),marker='.')
    plt.show(2)
    
    return np.float(error) / (np.size(train_set_y) - num_training)
       
def snr_simulation(start_snr) : # ber curve
    snr = np.array([4,6,8,10,12])
    
    result__ber = np.zeros((2,np.size(snr)))

    for i in range(np.size(snr)) :
       
        n_ff            = 10
        n_fb            = 0
        m_ary           = 4
        num_of_data     = 40000
        learning_rate   = 0.01
        num_training_mode    = 20000   
        
        print(" \n \n Loop ", i+1, " snr = ", start_snr + snr[i])
        snr_in = start_snr + snr[i]
        # data generation
        datasets = sim_data_gen(snr_in, n_ff, n_fb, m_ary, num_of_data)
        # equalization    
        error_rate = softmax_equ(datasets, m_ary  , n_ff , n_fb,learning_rate, 
                                                     num_training_mode )
        print("Bit Error Rate", error_rate)
        result__ber[0,i] = snr_in
        result__ber[1,i] = error_rate

    plt.figure(10)
    plt.title('BER performance of Logistic Regression')
    plt.ylim(1e-5,1e0)
    plt.ylabel('BER')
    plt.xlabel('SNR(dB)')
    plt.yscale('log')
    plt.grid()
    plt.plot( result__ber[0,:], result__ber[1,:], marker='o')
    
    return result__ber

 
if __name__ == '__main__':
    
    ber = snr_simulation(start_snr = 6)
