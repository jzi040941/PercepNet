# PercepNet Copilot Instructions

# Agent policy (machine-readable)
require_approval_for: [shell, patch]
summarize_from_comments_only: true
search_mode: literal
max_search_hits: 50

## Overview
PercepNet is a real-time speech enhancement system implementing the paper "A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech". The codebase has dual-mode architecture: **training mode** (Python/PyTorch) and **inference mode** (C++ real-time processing).

## Architecture & Key Components

### Dual-Mode System
- **Training**: Python scripts with PyTorch neural networks (`rnn_train.py`, `dump_percepnet.py`)
- **Inference**: C++ real-time processing (`src/denoise.cpp`, `src/main.cpp`)
- Mode controlled by `TRAINING` preprocessor flag: `TRAINING=1` for training, `TRAINING=0` for inference

### Core Signal Processing Pipeline
1. **Frame Analysis** (`frame_analysis`): 480-sample frames → FFT → ERB band energy
2. **Feature Extraction** (`compute_frame_features`): Extract 70-dim features (band energies, pitch, correlation)
3. **Neural Network**: PyTorch GRU model → gain estimates for 34 ERB bands
4. **Post-filtering** (`post_filtering`): Envelope post-filtering with perceptual weighting
5. **Frame Synthesis** (`frame_synthesis`): Apply gains → IFFT → overlap-add

### ERB Filterbank (`src/erbband.h`)
- 34 perceptually-motivated frequency bands
- C++ template-based implementation with linear interpolation
- Critical for both training feature extraction and inference gain application

## Development Workflows

### Training Pipeline
```bash
# 1. Generate training features from audio data
bin/src/percepNet speech.pcm noise.pcm 4000 training.output

# 2. Convert binary features to HDF5
python3 utils/bin2h5.py training.output training.h5

# 3. Train neural network
cd utils && ./run.sh

# 4. Export trained model to C++ header
python3 dump_percepnet.py model.pt
```

### Build System (CMake)
```bash
mkdir bin && cd bin
cmake ..
make -j
```
- Main executable: `bin/src/percepNet` (training mode)
- Inference executable: `bin/src/percepNet_run` (with `-DTRAINING=0`)
- Tests: GoogleTest framework in `tests/`

### Key Build Targets
- Use VS Code task `"C/C++: g++ build active file(PercepNet_run)"` for inference builds
- Training builds include feature extraction capabilities
- Vectorization support: AVX, ARM NEON, or fallback scalar (`src/vec*.h`)

## Critical Patterns & Conventions

### Data Flow & Dimensions
- **Audio**: 48kHz, 480 samples/frame (10ms), float32 normalized to [-1,1]
- **Features**: 70-dim input (68 band energies + 2 pitch features)
- **Neural output**: 68-dim gain values for ERB bands
- **ERB bands**: 34 bands, gains interpolated across frequency bins

### Model Architecture (`rnn_train.py`)
```python
# Standard PercepNet: Conv1D → GRU layers → Dense output
# Key hyperparameters: sequence_length=500, feature_dim=70
```

### C++/Python Integration
- `dump_percepnet.py`: Converts PyTorch weights → C arrays in `nnet_data.h`
- Python training uses same feature extraction as C++ (`utils/bin2h5.py`)
- Binary file format: float32 arrays, 138 values per frame (70 input + 68 target)

### Memory Management
- Fixed-size circular buffers for real-time processing
- `DenoiseState`: Maintains frame overlap, pitch history, RNN states
- `CommonState`: Shared FFT twiddles, window functions, DCT tables

## Testing & Debugging

### Sample Data Structure
```
sampledata/
├── speech/speech.pcm  # Clean speech samples
└── noise/noise.pcm    # Noise samples for training
```

### Key Test Commands
```bash
# Feature extraction test
bin/src/percepNet sampledata/speech/speech.pcm sampledata/noise/noise.pcm 4000 test.output

# Inference test  
bin/src/percepNet_run input.pcm output.pcm

# Unit tests
cd bin && make test
```

## External Dependencies

### Training Dependencies (`requirements.txt`)
- PyTorch, librosa, h5py for neural network training
- tensorboardX for training visualization

### DNS Challenge Integration
- Clone `microsoft/DNS-Challenge` (interspeech2020/master branch) 
- Synthesize training data using their noise+speech mixer
- Configure paths in `DNS-Challenge/noisyspeech_synthesizer.cfg`

## Common Pitfalls
- **Build mode confusion**: Ensure correct `TRAINING` flag for intended use
- **Sample rate**: Code assumes 48kHz audio; other rates need resampling
- **Feature scaling**: Training features scaled by 30x for band energies (`rnn_train.py:49`)
- **Model export**: Always run `dump_percepnet.py` after training before C++ inference