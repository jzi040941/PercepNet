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
  float noise_input[MAX_NEURONS*3];
  float denoise_input[MAX_NEURONS*3];
  compute_dense(rnn->model->input_dense, dense_out, input);
  compute_conv1d(rnn->model->first_conv1d, first_conv1d_out/*512*/, rnn->first_conv1d_state, dense_out);
  compute_conv1d(rnn->model->second_conv1d, second_conv1d_out/*512*/, rnn->second_conv1d_state, first_conv1d_out);
  
  //align 3 conv data
  RNN_MOVE(rnn->convout_buf, &rnn->convout_buf[CONV_DIM], CONVOUT_BUF_SIZE-CONV_DIM);
  RNN_COPY(&rnn->convout_buf[CONVOUT_BUF_SIZE-CONV_DIM], rnn->second_conv1d_state, CONV_DIM);
  //T-3 convout input for gru1
  compute_gru(rnn->model->gru1, rnn->gru1_state, rnn->convout_buf);
  compute_gru(rnn->model->gru2, rnn->gru2_state, rnn->gru1_state);
  compute_gru(rnn->model->gru3, rnn->gru3_state, rnn->gru2_state);
  
  //for temporary input for gb_gru and rb_gru is gru3_state
  //but it might be need concat convout_buf through gru1,2,3_state
  compute_gru(rnn->model->gb_gru, rnn->gb_gru_state, rnn->gru3_state);
  compute_gru(rnn->model->rb_gru, rnn->rb_gru_state, rnn->gru3_state);
  
  compute_dense(rnn->model->gb_output, gains, rnn->gb_gru_state);
  compute_dense(rnn->model->rb_output, strengths, rnn->rb_gru_state);
  /*
  compute_gru(rnn->model->vad_gru, rnn->vad_gru_state, dense_out);
  compute_dense(rnn->model->vad_output, vad, rnn->vad_gru_state);
  for (i=0;i<rnn->model->input_dense_size;i++) noise_input[i] = dense_out[i];
  for (i=0;i<rnn->model->vad_gru_size;i++) noise_input[i+rnn->model->input_dense_size] = rnn->vad_gru_state[i];
  for (i=0;i<INPUT_SIZE;i++) noise_input[i+rnn->model->input_dense_size+rnn->model->vad_gru_size] = input[i];
  compute_gru(rnn->model->noise_gru, rnn->noise_gru_state, noise_input);

  for (i=0;i<rnn->model->vad_gru_size;i++) denoise_input[i] = rnn->vad_gru_state[i];
  for (i=0;i<rnn->model->noise_gru_size;i++) denoise_input[i+rnn->model->vad_gru_size] = rnn->noise_gru_state[i];
  for (i=0;i<INPUT_SIZE;i++) denoise_input[i+rnn->model->vad_gru_size+rnn->model->noise_gru_size] = input[i];
  compute_gru(rnn->model->denoise_gru, rnn->denoise_gru_state, denoise_input);
  compute_dense(rnn->model->denoise_output, gains, rnn->denoise_gru_state);
  */
}
