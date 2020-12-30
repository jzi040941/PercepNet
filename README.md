# PercepNet(Under construct)
Unofficial implementation of PercepNet: A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech

https://www.researchgate.net/publication/343568932_A_Perceptually-Motivated_Approach_for_Low-Complexity_Real-Time_Enhancement_of_Fullband_Speech

## Todo

- [X] pitch estimation
- [X] Comb filter
- [X] ERBBand c++ implementation(need fix)
- [ ] Data Creater(c++) for pytorch
- [ ] DNNModel pytorch
- [ ] DNNModel c++ implementation
- [ ] Postfiltering

## SampleData

clean speech - VCTK 48k wav https://datashare.is.ed.ac.uk/handle/10283/2791 (clean_train_set)
noise data - DEMAND 48k wav https://zenodo.org/record/1227121#__sid=js0 (*.48k.zip)

# Reference
https://github.com/wil-j-wil/py_bank

https://github.com/dgaspari/pyrapt

https://github.com/xiph/rnnoise
