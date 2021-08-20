
#!/usr/bin/python3
'''Copyright (c) 2017-2018 Mozilla
                 2020-2021 Seonghun Noh
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import torch
import sys
import rnn_train
from torch.nn import Sequential, GRU, Conv1d, Linear
import numpy as np

def printVector(f, vector, name, dtype='float'):
    #torch.transpose(vector, 0, 1)
    v = np.reshape(vector.detach().numpy(), (-1))
    #print('static const float ', name, '[', len(v), '] = \n', file=f)
    f.write('static const {} {}[{}] = {{\n   '.format(dtype, name, len(v)))
    for i in range(0, len(v)):
        f.write('{}'.format(v[i]))
        if (i!=len(v)-1):
            f.write(',')
        else:
            break
        if (i%8==7):
            f.write("\n   ")
        else:
            f.write(" ")
    #print(v, file=f)
    f.write('\n};\n\n')
    return

def dump_sequential_module(self, f, name):
    activation = self[1].__class__.__name__.upper()
    self[0].dump_data(f,name,activation)
Sequential.dump_data = dump_sequential_module

def dump_linear_module(self, f, name, activation):
    print("printing layer " + name)
    weight = self.weight
    bias = self.bias
    #print("weight:", weight)
    #activation = self[1].__class__.__name__.upper()
    printVector(f, torch.transpose(weight, 0, 1), name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write('const DenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, weight.shape[1], weight.shape[0], activation))
Linear.dump_data = dump_linear_module

def convert_gru_input_kernel(kernel):
    kernel_r, kernel_z, kernel_h = np.vsplit(kernel, 3)
    kernels = [kernel_z, kernel_r, kernel_h]
    return torch.tensor(np.hstack([k.T for k in kernels]))

def convert_gru_recurrent_kernel(kernel):
    kernel_r, kernel_z, kernel_h = np.vsplit(kernel, 3)
    kernels = [kernel_z, kernel_r, kernel_h]
    return torch.tensor(np.hstack([k.T for k in kernels]))

def convert_bias(bias):
    bias = bias.reshape(2, 3, -1) 
    return torch.tensor(bias[:, [1, 0, 2], :].reshape(-1))

def dump_gru_module(self, f, name):
    print("printing layer " + name )
    weights = convert_gru_input_kernel(self.weight_ih_l0.detach().numpy())
    recurrent_weights = convert_gru_recurrent_kernel(self.weight_hh_l0.detach().numpy())
    bias = torch.cat((self.bias_ih_l0, self.bias_hh_l0))
    bias = convert_bias(bias.detach().numpy())
    printVector(f, weights, name + '_weights')
    printVector(f, recurrent_weights, name + '_recurrent_weights')
    printVector(f, bias, name + '_bias')
    if hasattr(self, 'activation'):
        activation = self.activation.__name__.upper()
    else:
        activation = 'TANH'
    if hasattr(self, 'reset_after') and not self.reset_after:
        reset_after = 0
    else:
        reset_after = 1
    neurons = weights.shape[0]//3
    #max_rnn_neurons = max(max_rnn_neurons, neurons)
    print('const GRULayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}_recurrent_weights,\n   {}, {}, ACTIVATION_{}, {}\n}};\n\n'
            .format(name, name, name, name, weights.shape[0], weights.shape[1]//3, activation, reset_after))
    f.write('const GRULayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}_recurrent_weights,\n   {}, {}, ACTIVATION_{}, {}\n}};\n\n'
            .format(name, name, name, name, weights.shape[0], weights.shape[1]//3, activation, reset_after))
    #hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights[0].shape[1]//3))
    #hf.write('#define {}_STATE_SIZE {}\n'.format(name.upper(), weights[0].shape[1]//3))
    #hf.write('extern const GRULayer {};\n\n'.format(name))
GRU.dump_data = dump_gru_module

def dump_conv1d_module(self, f, name, activation):
    print("printing layer " + name )
    weights = self.weight
    printVector(f, weights.permute(2,1,0), name + '_weights')
    printVector(f, self.bias, name + '_bias')
    #activation = self.activation.__name__.upper()
    #max_conv_inputs = max(max_conv_inputs, weights[0].shape[1]*weights[0].shape[0])
    #warn! activation hard codedW
    print('const Conv1DLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, weights.shape[1], weights.shape[2], weights.shape[0], activation))
    f.write('const Conv1DLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, weights.shape[1], weights.shape[2], weights.shape[0], activation))
    #hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights[0].shape[2]))
    #hf.write('#define {}_STATE_SIZE ({}*{})\n'.format(name.upper(), weights[0].shape[1], (weights[0].shape[0]-1)))
    #hf.write('#define {}_DELAY {}\n'.format(name.upper(), (weights[0].shape[0]-1)//2))
    #hf.write('extern const Conv1DLayer {};\n\n'.format(name));
Conv1d.dump_data = dump_conv1d_module

if __name__ == '__main__':
    model = rnn_train.PercepNet()
    #model = (
    model.load_state_dict(torch.load(sys.argv[1]))

    if len(sys.argv) > 2:
        cfile = sys.argv[2]
        #hfile = sys.argv[3];
    else:
        cfile = 'src/nnet_data.cpp'
        #hfile = 'nnet_data.h'

    f = open(cfile, 'w')
    #hf = open(hfile, 'w')

    f.write('/*This file is automatically generated from a Pytorch model*/\n\n')
    f.write('#ifdef HAVE_CONFIG_H\n#include "config.h"\n#endif\n\n#include "nnet.h"\n#include "nnet_data.h"\n\n')

    for name, module in model.named_children():
        module.dump_data(f, name)

    f.write('extern const RNNModel percepnet_model_orig = {\n')
    for name, module in model.named_children():
            f.write('    &{},\n'.format(name))
    f.write('};\n')

    f.close()
    print("done")