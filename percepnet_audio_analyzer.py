import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import io
import tempfile
import os
import subprocess
import time

def mix_speech_noise(speech_audio, noise_audio, snr_db=0):
    """
    Mix speech and noise signals at specified SNR.
    Truncates to shortest length.
    """
    # Truncate to shortest length
    min_len = min(len(speech_audio), len(noise_audio))
    speech_trimmed = speech_audio[:min_len]
    noise_trimmed = noise_audio[:min_len]
    
    # Calculate power levels
    speech_power = np.mean(speech_trimmed ** 2)
    noise_power = np.mean(noise_trimmed ** 2)
    
    # Calculate noise scaling factor for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    if noise_power > 0:
        noise_scale = np.sqrt(speech_power / (snr_linear * noise_power))
    else:
        noise_scale = 0
    
    # Mix signals
    mixed_audio = speech_trimmed + noise_scale * noise_trimmed
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0.95:
        mixed_audio = mixed_audio * 0.95 / max_val
    
    return mixed_audio

def run_percepnet(input_pcm_path, output_pcm_path):
    """
    Run PercepNet inference on the input PCM file.
    """
    try:
        # Check if PercepNet executable exists
        percepnet_exe = "bin/src/percepNet_run"
        if not os.path.exists(percepnet_exe):
            return False, f"PercepNet executable not found at {percepnet_exe}"
        
        # Run PercepNet
        cmd = [percepnet_exe, input_pcm_path, output_pcm_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=580)
        
        if result.returncode == 0:
            return True, "PercepNet processing completed successfully"
        else:
            return False, f"PercepNet error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "PercepNet processing timed out (>60s)"
    except Exception as e:
        return False, f"Error running PercepNet: {str(e)}"

def pcm_to_wav(pcm_file_path, sample_rate=48000):
    """
    Convert PCM file to WAV format.
    PercepNet uses 16-bit PCM at 48kHz.
    """
    try:
        # Read PCM file as 16-bit signed integers
        with open(pcm_file_path, 'rb') as f:
            pcm_data = np.frombuffer(f.read(), dtype=np.int16)
        
        # Normalize to [-1, 1] range
        wav_data = pcm_data.astype(np.float32) / 32768.0
        
        return wav_data, sample_rate
    except Exception as e:
        st.error(f"Error reading PCM file: {e}")
        return None, None

def create_spectrogram(audio_data, sample_rate, title="Spectrogram"):
    """
    Create a spectrogram plot of the audio data.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    
    # Plot spectrogram
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', 
                                   sr=sample_rate, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Magnitude (dB)')
    
    plt.tight_layout()
    return fig

def save_wav_for_playback(audio_data, sample_rate):
    """
    Save audio data to temporary WAV file for playback.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, audio_data, sample_rate)
    return temp_file.name

def main():
    st.set_page_config(
        page_title="PercepNet Audio Analyzer",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 PercepNet Audio Analyzer")
    st.markdown("---")
    st.markdown("**Analyze PercepNet input/output PCM files with audio playback and spectrogram visualization**")
    
    # File upload section
    st.subheader("📁 File Upload Options")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["🎵 Upload Files", "🔀 Create Noisy Mix"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📥 Input PCM File (Noisy)**")
            input_file = st.file_uploader(
                "Upload input PCM file", 
                type=['pcm'], 
                key="input",
                help="Upload the noisy input PCM file (16-bit, 48kHz)"
            )
        
        with col2:
            st.markdown("**📤 Output PCM File (Denoised)**")
            output_file = st.file_uploader(
                "Upload output PCM file", 
                type=['pcm'], 
                key="output",
                help="Upload the denoised output PCM file from PercepNet"
            )
    
    with tab2:
        st.markdown("**🎯 Create synthetic noisy audio by mixing clean speech with noise**")
        
        col_speech, col_noise, col_mix = st.columns([1, 1, 1])
        
        with col_speech:
            st.markdown("**🗣️ Clean Speech**")
            speech_file = st.file_uploader(
                "Upload clean speech PCM", 
                type=['pcm'], 
                key="speech",
                help="Upload clean speech PCM file (16-bit, 48kHz)"
            )
        
        with col_noise:
            st.markdown("**🔊 Noise**")
            noise_file = st.file_uploader(
                "Upload noise PCM", 
                type=['pcm'], 
                key="noise",
                help="Upload noise PCM file (16-bit, 48kHz)"
            )
        
        with col_mix:
            st.markdown("**⚙️ Mix Settings**")
            snr_db = st.slider(
                "SNR (dB)", 
                min_value=-10, 
                max_value=20, 
                value=5, 
                help="Signal-to-Noise Ratio in decibels"
            )
            
            if speech_file is not None and noise_file is not None:
                if st.button("🔀 Mix Audio", type="primary", help="Combine speech and noise"):
                    with st.spinner("Mixing audio files..."):
                        # Save files temporarily and process
                        with tempfile.NamedTemporaryFile(delete=False, suffix='_speech.pcm') as tmp_speech:
                            tmp_speech.write(speech_file.getvalue())
                            speech_path = tmp_speech.name
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='_noise.pcm') as tmp_noise:
                            tmp_noise.write(noise_file.getvalue())
                            noise_path = tmp_noise.name
                        
                        # Convert to audio arrays
                        speech_audio, speech_sr = pcm_to_wav(speech_path)
                        noise_audio, noise_sr = pcm_to_wav(noise_path)
                        
                        if speech_audio is not None and noise_audio is not None:
                            # Mix the audio
                            mixed_audio = mix_speech_noise(speech_audio, noise_audio, snr_db)
                            
                            # Convert back to PCM and store in session state
                            mixed_pcm = (mixed_audio * 32768).astype(np.int16)
                            mixed_bytes = mixed_pcm.tobytes()
                            
                            st.session_state['mixed_input'] = mixed_bytes
                            st.session_state['mix_info'] = {
                                'speech_len': len(speech_audio),
                                'noise_len': len(noise_audio),
                                'mixed_len': len(mixed_audio),
                                'snr_db': snr_db
                            }
                            
                            st.success(f"✅ Audio mixed successfully! SNR: {snr_db} dB, Duration: {len(mixed_audio)/speech_sr:.2f}s")
                            
                            # Clean up
                            os.unlink(speech_path)
                            os.unlink(noise_path)
                            
                            st.rerun()  # Refresh to use mixed audio as input
                        else:
                            st.error("Error processing speech or noise files")
                            # Clean up on error
                            if os.path.exists(speech_path):
                                os.unlink(speech_path)
                            if os.path.exists(noise_path):
                                os.unlink(noise_path)
    
    # Handle mixed audio from session state
    if 'mixed_input' in st.session_state:
        input_file = st.session_state['mixed_input']
        if 'mix_info' in st.session_state:
            mix_info = st.session_state['mix_info']
            st.info(f"🎵 Using mixed audio: SNR={mix_info['snr_db']} dB, "
                   f"Speech: {mix_info['speech_len']} samples, "
                   f"Noise: {mix_info['noise_len']} samples, "
                   f"Mixed: {mix_info['mixed_len']} samples")
    
    # PercepNet Processing Section
    if input_file is not None:
        st.markdown("---")
        st.subheader("🚀 PercepNet Processing")
        
        col_run1, col_run2 = st.columns([1, 2])
        
        with col_run1:
            if st.button("🎯 Run PercepNet", type="primary", help="Process the input file with PercepNet"):
                with st.spinner("Running PercepNet inference..."):
                    # Handle both uploaded files and processed session state data
                    if isinstance(input_file, bytes):
                        # This is from session state (mixed audio)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='_input.pcm') as tmp_input:
                            tmp_input.write(input_file)
                            input_pcm_path = tmp_input.name
                    else:
                        # This is an uploaded file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='_input.pcm') as tmp_input:
                            tmp_input.write(input_file.getvalue())
                            input_pcm_path = tmp_input.name                    # Create temporary output file path
                    output_pcm_path = input_pcm_path.replace('_input.pcm', '_output.pcm')
                    
                    # Run PercepNet
                    success, message = run_percepnet(input_pcm_path, output_pcm_path)
                    
                    if success:
                        st.success("✅ " + message)
                        
                        # Store the processed output in session state
                        if os.path.exists(output_pcm_path):
                            with open(output_pcm_path, 'rb') as f:
                                st.session_state['processed_output'] = f.read()
                            st.session_state['output_available'] = True
                            
                            # Clean up temporary files
                            os.unlink(input_pcm_path)
                            os.unlink(output_pcm_path)
                            
                            st.rerun()  # Refresh to show the processed output
                        else:
                            st.error("Output file was not created")
                    else:
                        st.error("❌ " + message)
                        # Clean up input file
                        if os.path.exists(input_pcm_path):
                            os.unlink(input_pcm_path)
        
        with col_run2:
            st.info("""
            **ℹ️ PercepNet Processing Info:**
            - Processes uploaded input file through PercepNet
            - Generates denoised output automatically
            - Output will appear below for analysis
            - Requires PercepNet executable at `bin/src/percepNet_run`
            """)
    
    # Handle processed output from session state
    if 'output_available' in st.session_state and st.session_state['output_available']:
        # Create a temporary file-like object for the processed output
        if 'processed_output' in st.session_state:
            output_file = st.session_state['processed_output']
            st.success("🎉 Using PercepNet-processed output for analysis")
    
    # Audio processing section
    if input_file is not None or output_file is not None:
        st.markdown("---")
        
        # Process input file
        if input_file is not None:
            with st.spinner("Processing input file..."):
                # Handle both uploaded files and processed session state data
                if isinstance(input_file, bytes):
                    # This is from session state (mixed audio)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pcm') as tmp_input:
                        tmp_input.write(input_file)
                        tmp_input_path = tmp_input.name
                else:
                    # This is an uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pcm') as tmp_input:
                        tmp_input.write(input_file.getvalue())
                        tmp_input_path = tmp_input.name
                
                # Convert PCM to WAV
                input_audio, input_sr = pcm_to_wav(tmp_input_path)
                
                if input_audio is not None:
                    st.success(f"✅ Input file loaded: {len(input_audio)/input_sr:.2f} seconds")
                    
                    # Audio playback section
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.subheader("🔊 Input Audio Playback")
                        input_wav_path = save_wav_for_playback(input_audio, input_sr)
                        
                        with open(input_wav_path, 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/wav', start_time=0)
                        
                        # Audio statistics
                        st.markdown("**Audio Stats:**")
                        st.write(f"Duration: {len(input_audio)/input_sr:.2f}s")
                        st.write(f"Sample Rate: {input_sr} Hz")
                        st.write(f"Samples: {len(input_audio):,}")
                        st.write(f"RMS Level: {np.sqrt(np.mean(input_audio**2)):.4f}")
                        st.write(f"Peak Level: {np.max(np.abs(input_audio)):.4f}")
                    
                    with col2:
                        st.subheader("📊 Input Spectrogram")
                        input_fig = create_spectrogram(input_audio, input_sr, "Input (Noisy) Spectrogram")
                        st.pyplot(input_fig)
                
                # Cleanup
                os.unlink(tmp_input_path)
                if 'input_wav_path' in locals():
                    os.unlink(input_wav_path)
        
        # Process output file
        if output_file is not None:
            with st.spinner("Processing output file..."):
                # Handle both uploaded files and processed session state data
                if isinstance(output_file, bytes):
                    # This is from session state (processed output)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pcm') as tmp_output:
                        tmp_output.write(output_file)
                        tmp_output_path = tmp_output.name
                else:
                    # This is an uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pcm') as tmp_output:
                        tmp_output.write(output_file.getvalue())
                        tmp_output_path = tmp_output.name
                
                # Convert PCM to WAV
                output_audio, output_sr = pcm_to_wav(tmp_output_path)
                
                if output_audio is not None:
                    st.success(f"✅ Output file loaded: {len(output_audio)/output_sr:.2f} seconds")
                    
                    # Audio playback section
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.subheader("🔊 Output Audio Playback")
                        output_wav_path = save_wav_for_playback(output_audio, output_sr)
                        
                        with open(output_wav_path, 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/wav', start_time=0)
                        
                        # Audio statistics
                        st.markdown("**Audio Stats:**")
                        st.write(f"Duration: {len(output_audio)/output_sr:.2f}s")
                        st.write(f"Sample Rate: {output_sr} Hz")
                        st.write(f"Samples: {len(output_audio):,}")
                        st.write(f"RMS Level: {np.sqrt(np.mean(output_audio**2)):.4f}")
                        st.write(f"Peak Level: {np.max(np.abs(output_audio)):.4f}")
                    
                    with col2:
                        st.subheader("📊 Output Spectrogram")
                        output_fig = create_spectrogram(output_audio, output_sr, "Output (Denoised) Spectrogram")
                        st.pyplot(output_fig)
                
                # Cleanup
                os.unlink(tmp_output_path)
                if 'output_wav_path' in locals():
                    os.unlink(output_wav_path)
        
        # Comparison section
        if input_file is not None and output_file is not None and 'input_audio' in locals() and 'output_audio' in locals():
            st.markdown("---")
            st.subheader("🔍 Comparison Analysis")
            
            # Ensure same length for comparison
            min_len = min(len(input_audio), len(output_audio))
            input_trimmed = input_audio[:min_len]
            output_trimmed = output_audio[:min_len]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Signal Metrics Comparison:**")
                
                input_rms = np.sqrt(np.mean(input_trimmed**2))
                output_rms = np.sqrt(np.mean(output_trimmed**2))
                input_peak = np.max(np.abs(input_trimmed))
                output_peak = np.max(np.abs(output_trimmed))
                
                metrics_df = {
                    "Metric": ["RMS Level", "Peak Level", "SNR Estimate (dB)"],
                    "Input": [f"{input_rms:.4f}", f"{input_peak:.4f}", "N/A"],
                    "Output": [f"{output_rms:.4f}", f"{output_peak:.4f}", "N/A"]
                }
                
                # Estimate SNR improvement (rough approximation)
                noise_est = input_trimmed - output_trimmed
                signal_power = np.mean(output_trimmed**2)
                noise_power = np.mean(noise_est**2)
                if noise_power > 0:
                    snr_improvement = 10 * np.log10(signal_power / noise_power)
                    metrics_df["Output"][2] = f"{snr_improvement:.1f}"
                
                st.table(metrics_df)
            
            with col2:
                st.markdown("**🎯 Difference Analysis:**")
                difference = input_trimmed - output_trimmed
                
                st.write(f"**Difference RMS:** {np.sqrt(np.mean(difference**2)):.4f}")
                st.write(f"**Difference Peak:** {np.max(np.abs(difference)):.4f}")
                st.write(f"**Correlation:** {np.corrcoef(input_trimmed, output_trimmed)[0,1]:.4f}")
                
                # Plot difference signal
                fig_diff, ax_diff = plt.subplots(figsize=(10, 4))
                time_axis = np.arange(len(difference)) / input_sr
                ax_diff.plot(time_axis, difference, alpha=0.7, color='red')
                ax_diff.set_title('Difference Signal (Input - Output)', fontweight='bold')
                ax_diff.set_xlabel('Time (s)')
                ax_diff.set_ylabel('Amplitude')
                ax_diff.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_diff)
            
            # Side-by-side spectrogram comparison
            st.markdown("### 📊 Side-by-Side Spectrogram Comparison")
            fig_comparison, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Input spectrogram
            D_input = librosa.amplitude_to_db(np.abs(librosa.stft(input_trimmed)), ref=np.max)
            librosa.display.specshow(D_input, y_axis='hz', x_axis='time', sr=input_sr, ax=ax1)
            ax1.set_title('Input (Noisy)', fontweight='bold')
            ax1.set_ylabel('Frequency (Hz)')
            
            # Output spectrogram
            D_output = librosa.amplitude_to_db(np.abs(librosa.stft(output_trimmed)), ref=np.max)
            img = librosa.display.specshow(D_output, y_axis='hz', x_axis='time', sr=output_sr, ax=ax2)
            ax2.set_title('Output (Denoised)', fontweight='bold')
            ax2.set_ylabel('Frequency (Hz)')
            
            # Shared colorbar
            plt.colorbar(img, ax=[ax1, ax2], format='%+2.0f dB', label='Magnitude (dB)')
            plt.tight_layout()
            st.pyplot(fig_comparison)
    
    # Usage instructions
    if input_file is None and output_file is None:
        st.markdown("---")
        st.info("""
        **📋 How to use:**
        
        **Method 1: Upload existing files**
        1. Upload PCM files using the "Upload Files" tab
        2. Listen to audio and analyze spectrograms
        3. Compare input vs output performance
        
        **Method 2: Create synthetic noisy audio**
        1. Go to "Create Noisy Mix" tab
        2. Upload clean speech PCM file
        3. Upload noise PCM file  
        4. Adjust SNR slider and click "Mix Audio"
        5. Use "Run PercepNet" to process the mixed audio
        
        **Method 3: Full pipeline test**
        1. Create noisy mix (Method 2)
        2. Click "Run PercepNet" to denoise
        3. Compare original clean speech vs noisy vs denoised
        
        **💡 Expected file format:**
        - 16-bit signed PCM files
        - 48 kHz sample rate (PercepNet default)
        - Mono audio
        """)
        
        st.markdown("**🎯 Example workflows:**")
        st.code("""
        # Workflow 1: Test with synthetic data
        # 1. Upload clean speech + noise files
        # 2. Mix at desired SNR (e.g., 5 dB)  
        # 3. Click "Run PercepNet" 
        # 4. Analyze denoising performance
        
        # Workflow 2: Manual command line
        bin/src/percepNet_run input_noisy.pcm output_clean.pcm
        
        # Then upload both files for analysis
        """, language="bash")

if __name__ == "__main__":
    main()