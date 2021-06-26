#include "nnet.h"
#include "nnet_data.h"
#include <vector>
#include <stdio.h>
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