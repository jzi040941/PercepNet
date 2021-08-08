/* Copyright (c) 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
/*
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
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include "opus_types.h"
#include "common.h"
#include "arch.h"
#include "tansig_table.h"
#include "nnet.h"
#include "nnet_data.h"
#include <stdio.h>


void compute_rnn(RNNState *rnn, float *gains, float *strengths, const float *input) {
  int i;
  float dense_out[MAX_NEURONS];
  float first_conv1d_out[CONV_DIM];
  float second_conv1d_out[CONV_DIM];
  float gb_dense_input[CONV_DIM*5];
  float rb_gru_input[CONV_DIM*2];
  compute_dense(rnn->model->fc, dense_out, input);
  compute_conv1d(rnn->model->conv1, first_conv1d_out/*512*/, rnn->first_conv1d_state, dense_out);
  compute_conv1d(rnn->model->conv2, second_conv1d_out/*512*/, rnn->second_conv1d_state, first_conv1d_out);
  
  //align 3 conv data
  //RNN_MOVE(rnn->convout_buf, &rnn->convout_buf[CONV_DIM], CONVOUT_BUF_SIZE-CONV_DIM);
  //RNN_COPY(&rnn->convout_buf[CONVOUT_BUF_SIZE-CONV_DIM], rnn->second_conv1d_state, CONV_DIM);
  //T-3 convout input for gru1
  compute_gru(rnn->model->gru1, rnn->gru1_state, second_conv1d_out);
  compute_gru(rnn->model->gru2, rnn->gru2_state, rnn->gru1_state);
  compute_gru(rnn->model->gru3, rnn->gru3_state, rnn->gru2_state);
  
  //for temporary input for gb_gru and rb_gru is gru3_state
  //but it might be need concat convout_buf through gru1,2,3_state
  compute_gru(rnn->model->gru_gb, rnn->gb_gru_state, rnn->gru3_state);

  //concat for rb gru
  for (i=0;i<CONV_DIM;i++) rb_gru_input[i] = second_conv1d_out[i];
  for (i=0;i<CONV_DIM;i++) rb_gru_input[i+CONV_DIM] = rnn->gru3_state[i];
  compute_gru(rnn->model->gru_rb, rnn->rb_gru_state, rb_gru_input);
  
  //concat for gb denseW
  for (i=0;i<CONV_DIM;i++) gb_dense_input[i] = second_conv1d_out[i];
  for (i=0;i<CONV_DIM;i++) gb_dense_input[i+CONV_DIM] = rnn->gru1_state[i];
  for (i=0;i<CONV_DIM;i++) gb_dense_input[i+2*CONV_DIM] = rnn->gru2_state[i];
  for (i=0;i<CONV_DIM;i++) gb_dense_input[i+3*CONV_DIM] = rnn->gru3_state[i];
  for (i=0;i<CONV_DIM;i++) gb_dense_input[i+4*CONV_DIM] = rnn->gb_gru_state[i];
  compute_dense(rnn->model->fc_gb, gains, gb_dense_input);

  compute_dense(rnn->model->fc_rb, strengths, rnn->rb_gru_state);

}
