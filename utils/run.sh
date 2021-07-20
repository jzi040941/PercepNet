/* Copyright (c) 2021 Seonghun Noh */
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

. ./path.sh || exit 1;

dataset_dir="sampledata"
noise_dir="noise"
clean_dir="clean"
h5_dir="h5"
model_filename="training_h5"

###################################################
#resample to 48khz and convert wav to pcm         #
###################################################
##TODO: with sox

###################################################
#Generate data for each noise and clean data      #
###################################################
for noise_pcm in "${PRJ_ROOT}/${dataset_dir}/${clean_dir}/*.pcm"
do
   for clean_pcm in "${PRJ_ROOT}/${dataset_dir}/${noise_dir}/*.pcm"
   do
      echo "$noise_pcm $clean_pcm"
   done
done

###################################################
#Convert features to h5 files                     #
###################################################

###################################################
#Train pytorch model                              #
###################################################

###################################################
#Convert pytorch model to c++ header              #
###################################################
#dump_percepnet.py ...