#include "nnet.h"
#include "nnet_data_test.h"
#include <vector>
#include <stdio.h>
#include <gtest/gtest.h>

// Demonstrate some basic assertions.
/*
TEST(TestNnet, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}
*/



TEST(TestNnet, fcCheck) {
    float eps = 1e-5;
    std::vector<float> fc_input(fc.nb_inputs, 0.5);
    std::vector<float> fc_output_c(fc.nb_neurons, 0);
    compute_dense(&fc, &fc_output_c[0], &fc_input[0]);
    
    for(int i=0; i<fc_output_c.size(); i++){
        //EXPECT_EQ(fc_output[i], fc_output_c[i]);
        EXPECT_LT(std::abs(fc_output[i] - fc_output_c[i]), eps);
    }
}

TEST(TestNnet, conv1dCheck) {
    std::vector<float> first_conv1d_state(conv1.kernel_size*conv1.nb_inputs,0);
    float eps = 1e-5;
    std::vector<float> conv1_input(conv1.nb_inputs*3, 0.5);
    std::vector<float> conv1_output_c(conv1.nb_neurons, 0);
    compute_conv1d(&conv1, &conv1_output_c[0], &first_conv1d_state[0], &conv1_input[0]);
    compute_conv1d(&conv1, &conv1_output_c[0], &first_conv1d_state[0], &conv1_input[0]);
    //EXPECT_EQ(conv1_output_c.size(), sizeof(conv1_output)/sizeof(float));
    for(int i=0; i<conv1_output_c.size(); i++){
        //EXPECT_EQ(conv1_output[i], conv1_output_c[i]);
        EXPECT_LT(std::abs(conv1_output[i] - conv1_output_c[i]), eps);
    }
    compute_conv1d(&conv1, &conv1_output_c[0], &first_conv1d_state[0], &conv1_input[0]);
    for(int i=0; i<conv1_output_c.size(); i++){
        //EXPECT_EQ(conv1_output[i+3], conv1_output_c[i]);
        EXPECT_LT(std::abs(conv1_output[i+3] - conv1_output_c[i]), eps);
    }
}

TEST(TestNnet, gruCheck) {
    float eps = 1e-5;
    std::vector<float> gru1_state(gru1.nb_neurons,0);
    std::vector<float> gru1_input(gru1.nb_inputs, 0.5);
    //std::vector<float> gru1_output_c(gr.nb_neurons, 0);
    compute_gru(&gru1,&gru1_state[0], &gru1_input[0]);

    for(int i=0; i<gru1_state.size(); i++){
        //EXPECT_FLOAT_EQ(gru1_output[i], gru1_state[i]);
        EXPECT_LT(gru1_output[i] - gru1_state[i], eps);
    }
    compute_gru(&gru1,&gru1_state[0], &gru1_input[0]);
    for(int i=0; i<gru1_state.size(); i++){
        //EXPECT_FLOAT_EQ(gru1_output[i+3], gru1_state[i]);
        EXPECT_LT(gru1_output[i+3] - gru1_state[i], eps);
    }
}


/*
int main(){
    std::vector<float> fc_input(fc.nb_inputs, 0.5);
    std::vector<float> fc_output_c(fc.nb_neurons, 0);
    compute_dense(&fc, &fc_output_c[0], &fc_input[0]);
    
    for(int i=0; i<fc_output_c.size(); i++){
        if(fc_output[i] != fc_output_c[i]){
            printf("fail");
            return 0;
        }
    }
    //compute_conv1d()
    //compute_gru(rnn->model->gb_gru, rnn->gb_gru_state, rnn->gru3_state);
    

    return 0;
}
*/