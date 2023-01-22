# PercepNet
Unofficial implementation of PercepNet: A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech described in https://arxiv.org/abs/2008.04259

https://www.researchgate.net/publication/343568932_A_Perceptually-Motivated_Approach_for_Low-Complexity_Real-Time_Enhancement_of_Fullband_Speech

## Todo

- [X] pitch estimation
- [X] Comb filter
- [X] ERBBand c++ implementation
- [X] Feature(r,g,pitch,corr) Generator(c++) for pytorch
- [X] DNNModel pytorch
- [X] DNNModel c++ implementation
- [ ] Pretrained model
- [X] Postfiltering (done by [@TeaPoly](https://github.com/TeaPoly ) )


## Requirements
 - CMake
 - Sox
 - Python>=3.6
 - Pytorch
 
## Build & Training
This repository is tested on Ubuntu 20.04(WSL2)

1. setup CMake build environments
```
sudo apt-get install cmake
```
2. make binary directory & build
```
mkdir bin && cd bin
cmake ..
make -j
cd ..
```

3. feature generation for training with sampleData
```
bin/src/percepNet sampledata/speech/speech.pcm sampledata/noise/noise20db.raw 4000 test.output
```

4. Convert output binary to h5
```
python3 utils/bin2h5.py test.output training.h5
```

5. Training
```
python3 rnn_train.py
```

6. Dump weight from pytorch to c++ header
```
python3 dump_percepnet.py model.pt
```

7. Inference
```
cd bin
cmake ..
make -j1
cd ..
bin/src/percepNet_run test_input.pcm percepnet_output.pcm
```

## SampleData
download and sythesize data DNS-Challenge 2020 Dataset before excute utils/run.sh for training. 
```shell
git clone -b interspeech2020/master  https://github.com/microsoft/DNS-Challenge.git
```
Follow the Usage instruction in DNS Challenge repo(https://github.com/microsoft/DNS-Challenge) at interspeech2020/master branch.

modify save dir at DNS-Challenge/noisyspeech_synthesizer.cfg sampledata/speech and sampledata/noise each.

## Acknowledgements
[@jasdasdf]( https://github.com/jasdasdf ), [@sTarAnna]( https://github.com/sTarAnna ), [@cookcodes]( https://github.com/cookcodes ), [@xyx361100238]( https://github.com/xyx361100238 ), [@zhangyutf]( https://github.com/zhangyutf ), [@TeaPoly](https://github.com/TeaPoly ), [@rameshkunasi]( https://github.com/rameshkunasi ),  [@OscarLiau]( https://github.com/OscarLiau ), [@YangangCao]( https://github.com/YangangCao ), [Jaeyoung Yang]( https://www.linkedin.com/in/jaeyoung-yang-354b21146 )

[IIP Lab. Sogang Univ]( http://iip.sogang.ac.kr/) 



## Reference
https://github.com/wil-j-wil/py_bank

https://github.com/dgaspari/pyrapt

https://github.com/xiph/rnnoise

https://github.com/mozilla/LPCNet
