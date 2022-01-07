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

dataset_dir="training_set_sept12_500h"
noisy_wav_dir="noisy"
clean_wav_dir="clean"
noisy_pcm_dir="noisy_pcm"
clean_pcm_dir="clean_pcm"
feature_dir="features"
h5_train_dir="h5_train"
h5_dev_dir="h5_dev"
out_dir="exp_erbfix_x30_snr45_rmax99"
#out_dir="exp_test"
model_filename="model.pt"
train_size_per_batch=2000
config="DNS_Challenge.yaml"
#pretrain="/home/seonghun/develop/PercepNet/training_set_sept12_500h/exp_erbfix_band30times_nopretrain/checkpoint-10000steps.pkl"

stage=4     #stage to start
stop_stage=4 #stop stage

NR_CPUS=8 #TODO: automatically detect how many cpu have


###################################################
# mkdir related dir if not exist                  #
###################################################
mkdir -p ${PRJ_ROOT}/${dataset_dir}/{${noisy_pcm_dir},${clean_pcm_dir},${h5_train_dir},${h5_dev_dir},${feature_dir},${out_dir}}

###################################################
# stage1 :resample to 48khz and convert wav to pcm#
###################################################
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
   i=0
   mkdir -p ${PRJ_ROOT}/${dataset_dir}/${noisy_wav_dir}/fileid
   mkdir -p ${PRJ_ROOT}/${dataset_dir}/${clean_wav_dir}/fileid
   for wavfilepath in ${PRJ_ROOT}/${dataset_dir}/${noisy_wav_dir}/*.wav; do
      ((i=i%NR_CPUS)); ((i++==0)) && wait
      (
      # pcmfilename="`basename "${wavfilepath##*fileid_}" .wav`.pcm"
      # pcmfilepath=${PRJ_ROOT}/${dataset_dir}/${noisy_pcm_dir}/${pcmfilename}
      # sox ${wavfilepath} -b 16 -e signed-integer -c 1 -r 48k -t raw ${pcmfilepath}
      
      # reduce disk usage
      newwavfilename="`basename "${wavfilepath##*fileid_}" .wav`.wav"
      mv ${wavfilepath} ${PRJ_ROOT}/${dataset_dir}/${noisy_pcm_dir}/${newwavfilename}
      ) &
   done
   wait

   i=0
   for wavfilepath in ${PRJ_ROOT}/${dataset_dir}/${clean_wav_dir}/*.wav; do
      ((i=i%NR_CPUS)); ((i++==0)) && wait
      (
      # pcmfilename="`basename "${wavfilepath##*fileid_}" .wav`.pcm"
      # pcmfilepath=${PRJ_ROOT}/${dataset_dir}/${clean_pcm_dir}/${pcmfilename}
      # sox ${wavfilepath} -b 16 -e signed-integer -c 1 -r 48k -t raw ${pcmfilepath}
      newwavfilename="`basename "${wavfilepath##*fileid_}" .wav`.wav"
      mv ${wavfilepath} ${PRJ_ROOT}/${dataset_dir}/${clean_pcm_dir}/${newwavfilename}
      ) &
   done
   wait
fi

###################################################
#Generate c++ feature data for each noisy and clean data      #
###################################################
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
   i=0
   
   for noisy_wav_filepath in ${PRJ_ROOT}/${dataset_dir}/${noisy_pcm_dir}/*.wav; do
      ((i=i%NR_CPUS)); ((i++==0)) && wait
      (
      fileid=`basename "${noisy_wav_filepath%%.wav*}" .wav`
      noisy_pcm_filepath="${PRJ_ROOT}/${dataset_dir}/${noisy_pcm_dir}/${fileid}.pcm"
      sox ${noisy_wav_filepath} -b 16 -e signed-integer -c 1 -r 48k -t raw ${noisy_pcm_filepath}
      clean_wav_filepath="${PRJ_ROOT}/${dataset_dir}/${clean_pcm_dir}/${fileid}.wav"
      clean_pcm_filepath="${PRJ_ROOT}/${dataset_dir}/${clean_pcm_dir}/${fileid}.pcm"
      sox ${clean_wav_filepath} -b 16 -e signed-integer -c 1 -r 48k -t raw ${clean_pcm_filepath}
      ${PRJ_ROOT}/bin/src/percepNet ${clean_pcm_filepath} \
      ${noisy_pcm_filepath} \
      ${train_size_per_batch} \
      ${PRJ_ROOT}/${dataset_dir}/${feature_dir}/${fileid}.out
      rm ${noisy_pcm_filepath}
      rm ${clean_pcm_filepath}
      echo "genereated ${fileid}.out"
      ) &
   done
   wait
fi

###################################################
#Convert features to h5 files & split dataset     #
###################################################
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
   python3 split_feature_dataset.py ${PRJ_ROOT}/${dataset_dir}/${feature_dir}
   # for featurefile in `cat ${PRJ_ROOT}/${dataset_dir}/${feature_dir}/train.txt`; do
   #    fileid=`basename ${featurefile} .out`
   #    python3 bin2h5.py ${featurefile} ${PRJ_ROOT}/${dataset_dir}/${h5_train_dir}/${fileid}.h5
   # done
   # for featurefile in `cat ${PRJ_ROOT}/${dataset_dir}/${feature_dir}/dev.txt`; do
   #    fileid=`basename ${featurefile} .out`
   #    python3 bin2h5.py ${featurefile} ${PRJ_ROOT}/${dataset_dir}/${h5_dev_dir}/${fileid}.h5
   # done
fi

###################################################
#Train pytorch model                              #
###################################################

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
   echo "'--train_length_size', '${train_size_per_batch}', '--train_filelist_path', '${PRJ_ROOT}/${dataset_dir}/${feature_dir}/train.txt', \
                        '--dev_filelist_path', '${PRJ_ROOT}/${dataset_dir}/${feature_dir}/dev.txt', \
                        '--out_dir', '${PRJ_ROOT}/${dataset_dir}/${out_dir}', '--config', '${config}'"
   python3 ${PRJ_ROOT}/rnn_train.py --train_length_size ${train_size_per_batch} --train_filelist_path ${PRJ_ROOT}/${dataset_dir}/${feature_dir}/train.txt \
                        --dev_filelist_path ${PRJ_ROOT}/${dataset_dir}/${feature_dir}/dev.txt \
                        --out_dir ${PRJ_ROOT}/${dataset_dir}/${out_dir} --config ${config} 
fi

###################################################
#Convert pytorch model to c++ header              #
###################################################
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
   python3 dump_percepnet.py ${model_filename}
fi