#!/bin/bash
# Copyright (c) 2021 Seonghun Noh 

   # Redistribution and use in source and binary forms, with or without
   # modification, are permitted provided that the following conditions
   # are met:

   # - Redistributions of source code must retain the above copyright
   # notice, this list of conditions and the following disclaimer.

   # - Redistributions in binary form must reproduce the above copyright
   # notice, this list of conditions and the following disclaimer in the
   # documentation and/or other materials provided with the distribution.

   # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   # ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   # LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   # A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


. ./path.sh || exit 1;
. ./parse_options.sh || exit 1;

dataset_dir="training_set_sept12"
noisy_wav_dir="noisy"
clean_wav_dir="clean"
noisy_pcm_dir="noisy_pcm"
clean_pcm_dir="clean_pcm"
feature_dir="feature"
h5_train_dir="h5_train"
h5_dev_dir="h5_dev"
model_filename="model.pt"
train_size_per_batch=2000

stage=3     #stage to start
stop_stage=3 #stop stage

NR_CPUS=3 #TODO: automatically detect how many cpu have


###################################################
# mkdir related dir if not exist                  #
###################################################
mkdir -p ${PRJ_ROOT}/${dataset_dir}/{${noisy_pcm_dir},${clean_pcm_dir},${h5_train_dir},${h5_dev_dir},${feature_dir}}

###################################################
# stage1 :resample to 48khz and convert wav to pcm#
###################################################
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
   job_count=0
   for wavfilepath in ${PRJ_ROOT}/${dataset_dir}/${noisy_wav_dir}/*.wav; do
      pcmfilename="`basename "${wavfilepath##*fileid_}" .wav`.pcm"
      pcmfilepath=${PRJ_ROOT}/${dataset_dir}/${noisy_pcm_dir}/${pcmfilename}
      sox ${wavfilepath} -b 16 -e signed-integer -c 1 -r 48k -t raw ${pcmfilepath}
   done

   for wavfilepath in ${PRJ_ROOT}/${dataset_dir}/${clean_wav_dir}/*.wav; do
      pcmfilename="`basename "${wavfilepath##*fileid_}" .wav`.pcm"
      pcmfilepath=${PRJ_ROOT}/${dataset_dir}/${clean_pcm_dir}/${pcmfilename}
      sox ${wavfilepath} -b 16 -e signed-integer -c 1 -r 48k -t raw ${pcmfilepath}
   done
   
fi

###################################################
#Generate c++ feature data for each noisy and clean data      #
###################################################
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
   for noisy_pcm_filepath in ${PRJ_ROOT}/${dataset_dir}/${noisy_pcm_dir}/*.pcm; do
      fileid=`basename "${noisy_pcm_filepath%%.pcm*}" .pcm`
      clean_pcm_filepath="${PRJ_ROOT}/${dataset_dir}/${clean_pcm_dir}/${fileid}.pcm"
      ${PRJ_ROOT}/bin/src/percepNet ${clean_pcm_filepath} \
      ${noisy_pcm_filepath} \
      ${train_size_per_batch} \
      ${PRJ_ROOT}/${dataset_dir}/${feature_dir}/${fileid}.out
      echo "genereated ${fileid}.out"
   done
fi

###################################################
#Convert features to h5 files & split dataset     #
###################################################
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
   python3 split_feature_dataset.py ${PRJ_ROOT}/${dataset_dir}/${feature_dir}
   for featurefile in `cat ${PRJ_ROOT}/${dataset_dir}/${feature_dir}/train.txt`; do
      fileid=`basename ${featurefile} .out`
      python3 bin2h5.py ${featurefile} ${PRJ_ROOT}/${dataset_dir}/${h5_train_dir}/${fileid}.h5
   done
   for featurefile in `cat ${PRJ_ROOT}/${dataset_dir}/${feature_dir}/dev.txt`; do
      fileid=`basename ${featurefile} .out`
      python3 bin2h5.py ${featurefile} ${PRJ_ROOT}/${dataset_dir}/${h5_dev_dir}/${fileid}.h5
   done
fi

###################################################
#Train pytorch model                              #
###################################################
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
   python3 rnn_train.py --train_length_size ${train_size_per_batch} --h5_dir ${PRJ_ROOT}/${dataset_dir}/${h5_dir} \
                        --model_filename ${model_filename}
fi

###################################################
#Convert pytorch model to c++ header              #
###################################################
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
   python3 dump_percepnet.py ${model_filename}
fi