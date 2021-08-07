#ifndef NNET_DATA_H
#define NNET_DATA_H

#include "nnet.h"

typedef struct RNNModel {
  const DenseLayer *fc;

  const Conv1DLayer *conv1;
  
  const Conv1DLayer *conv2;
  
  const GRULayer *gru1;

  const GRULayer *gru2;

  const GRULayer *gru3;

  const GRULayer *gb_gru;

  const GRULayer *rb_gru;

  const DenseLayer *gb_output;

  const DenseLayer *rb_output;
} RNNModel;

typedef struct RNNState {
  const RNNModel *model;
  float *first_conv1d_state;
  float *second_conv1d_state;
  float *gru1_state;
  float *gru2_state;
  float *gru3_state;
  float *gb_gru_state;
  float *rb_gru_state;
  float convout_buf[CONV_DIM*3];
} RNNState;


#endif