# PercepNet(Work In Progress)
Unofficial implementation of PercepNet: A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech described in https://arxiv.org/abs/2008.04259

https://www.researchgate.net/publication/343568932_A_Perceptually-Motivated_Approach_for_Low-Complexity_Real-Time_Enhancement_of_Fullband_Speech

## Todo

- [X] pitch estimation
- [X] Comb filter
- [X] ERBBand c++ implementation
- [X] Feature(r,g,pitch,corr) Generator(c++) for pytorch
- [X] DNNModel pytorch
- [ ] DNNModel c++ implementation
- [ ] Postfiltering


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
src/PercetNet sampledata/speech/speech.pcm sampledata/noise/noise.pcm 4000 test.output
```

4. Convert output binary to h5
```
python3 bin2h5.py test.output training.h5
```

5. Training
```
python3 rnn_train.py
```

6. Dump weight from pytorch to c++ header
```
python3 dump_percepnet.py model.pt
```

7. Inference(WIP)
...

## SampleData

clean speech - VCTK 48k wav https://datashare.is.ed.ac.uk/handle/10283/2791 (clean_train_set)

noise data - DEMAND 48k wav https://zenodo.org/record/1227121#__sid=js0 (*.48k.zip)

## Acknowledgements
[@jasdasdf]( https://github.com/jasdasdf ), [@sTarAnna]( https://github.com/sTarAnna ), [@cookcodes]( https://github.com/cookcodes ), [@xyx361100238]( https://github.com/xyx361100238 ), [@zhangyutf]( https://github.com/zhangyutf ), [@TeaPoly](https://github.com/TeaPoly ), [@rameshkunasi]( https://github.com/rameshkunasi ),  [@OscarLiau]( https://github.com/OscarLiau ), [@YangangCao]( https://github.com/YangangCao ), Jaeyoung Yang( https://www.linkedin.com/in/jaeyoung-yang-354b21146 )

[IIP Lab. Sogang Univ]( http://iip.sogang.ac.kr/) 



## Reference
https://github.com/wil-j-wil/py_bank

https://github.com/dgaspari/pyrapt

https://github.com/xiph/rnnoise

https://github.com/mozilla/LPCNet
