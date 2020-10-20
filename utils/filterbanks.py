'''
Created 06/03/2018
@author: Will Wilkinson
'''

import numpy as np


class FilterBank(object):
    """
    Based on Josh McDermott's Matlab filterbank code:
    http://mcdermottlab.mit.edu/Sound_Texture_Synthesis_Toolbox_v1.7.zip

    leny = filter bank length in samples
    fs = sample rate
    N = number of frequency channels / subbands (excluding high-&low-pass which are added for perfect reconstruction)
    low_lim = centre frequency of first (lowest) channel
    high_lim = centre frequency of last (highest) channel
    """
    def __init__(self, leny, fs, N, low_lim, high_lim):
        self.leny = leny
        self.fs = fs
        self.N = N
        self.low_lim = low_lim
        self.high_lim, self.freqs, self.nfreqs = self.check_limits(leny, fs, high_lim)

    def check_limits(self, leny, fs, high_lim):
        if np.remainder(leny, 2) == 0:
            nfreqs = leny / 2
            max_freq = fs / 2
        else:
            nfreqs = (leny - 1) / 2
            max_freq = fs * (leny - 1) / 2 / leny
        freqs = np.linspace(0, max_freq, nfreqs + 1)
        if high_lim > fs / 2:
            high_lim = max_freq
        return high_lim, freqs, int(nfreqs)

    def generate_subbands(self, signal):
        if signal.shape[0] == 1:  # turn into column vector
            signal = np.transpose(signal)
        N = self.filters.shape[1] - 2
        signal_length = signal.shape[0]
        filt_length = self.filters.shape[0]
        # watch out: numpy fft acts on rows, whereas Matlab fft acts on columns
        fft_sample = np.transpose(np.asmatrix(np.fft.fft(signal)))
        # generate negative frequencies in right place; filters are column vectors
        if np.remainder(signal_length, 2) == 0:  # even length
            fft_filts = np.concatenate([self.filters, np.flipud(self.filters[1:filt_length - 1, :])])
        else:  # odd length
            fft_filts = np.concatenate([self.filters, np.flipud(self.filters[1:filt_length, :])])
        # multiply by array of column replicas of fft_sample
        tile = np.dot(fft_sample, np.ones([1, N + 2]))
        fft_subbands = np.multiply(fft_filts, tile)
        # ifft works on rows; imag part is small, probably discretization error?
        self.subbands = np.transpose(np.real(np.fft.ifft(np.transpose(fft_subbands))))


class EqualRectangularBandwidth(FilterBank):
    def __init__(self, leny, fs, N, low_lim, high_lim):
        super(EqualRectangularBandwidth, self).__init__(leny, fs, N, low_lim, high_lim)
        # make cutoffs evenly spaced on an erb scale
        erb_low = self.freq2erb(self.low_lim)
        erb_high = self.freq2erb(self.high_lim)
        erb_lims = np.linspace(erb_low, erb_high, self.N + 2)
        self.cutoffs = self.erb2freq(erb_lims)
        self.filters = self.make_filters(self.N, self.nfreqs, self.freqs, self.cutoffs)

    def freq2erb(self, freq_Hz):
        n_erb = 9.265 * np.log(1 + np.divide(freq_Hz, 24.7 * 9.265))
        return n_erb

    def erb2freq(self, n_erb):
        freq_Hz = 24.7 * 9.265 * (np.exp(np.divide(n_erb, 9.265)) - 1)
        return freq_Hz

    def make_filters(self, N, nfreqs, freqs, cutoffs):
        cos_filts = np.zeros([nfreqs + 1, N])
        for k in range(N):
            l_k = cutoffs[k]
            h_k = cutoffs[k + 2]  # adjacent filters overlap by 50%
            l_ind = np.min(np.where(freqs > l_k))
            h_ind = np.max(np.where(freqs < h_k))
            avg = (self.freq2erb(l_k) + self.freq2erb(h_k)) / 2
            rnge = self.freq2erb(h_k) - self.freq2erb(l_k)
            # map cutoffs to -pi/2, pi/2 interval
            cos_filts[l_ind:h_ind + 1, k] = np.cos((self.freq2erb(freqs[l_ind:h_ind + 1]) - avg) / rnge * np.pi)
        # add lowpass and highpass to get perfect reconstruction
        filters = np.zeros([nfreqs + 1, N + 2])
        filters[:, 1:N + 1] = cos_filts
        # lowpass filter goes up to peak of first cos filter
        h_ind = np.max(np.where(freqs < cutoffs[1]))
        filters[:h_ind + 1, 0] = np.sqrt(1 - np.power(filters[:h_ind + 1, 1], 2))
        # highpass filter goes down to peak of last cos filter
        l_ind = np.min(np.where(freqs > cutoffs[N]))
        filters[l_ind:nfreqs + 1, N + 1] = np.sqrt(1 - np.power(filters[l_ind:nfreqs + 1, N], 2))
        return filters


class Linear(FilterBank):
    def __init__(self, leny, fs, N, low_lim, high_lim):
        super(Linear, self).__init__(leny, fs, N, low_lim, high_lim)
        self.cutoffs = np.linspace(self.low_lim, self.high_lim, self.N + 2)
        self.filters = self.make_filters(self.N, self.nfreqs, self.freqs, self.cutoffs)

    def make_filters(self, N, nfreqs, freqs, cutoffs):
        cos_filts = np.zeros([nfreqs + 1, N])
        for k in range(N):
            l_k = cutoffs[k]
            h_k = cutoffs[k + 2]  # adjacent filters overlap by 50%
            l_ind = np.min(np.where(freqs > l_k))
            h_ind = np.max(np.where(freqs < h_k))
            avg = (l_k + h_k) / 2
            rnge = h_k - l_k
            # map cutoffs to -pi/2, pi/2 interval
            cos_filts[l_ind:h_ind + 1, k] = np.cos((freqs[l_ind:h_ind + 1] - avg) / rnge * np.pi)
        # add lowpass and highpass to get perfect reconstruction
        filters = np.zeros([nfreqs + 1, N + 2])
        filters[:, 1:N + 1] = cos_filts
        # lowpass filter goes up to peak of first cos filter
        h_ind = np.max(np.where(freqs < cutoffs[1]))
        filters[:h_ind + 1, 0] = np.sqrt(1 - np.power(filters[:h_ind + 1, 1], 2))
        # highpass filter goes down to peak of last cos filter
        l_ind = np.min(np.where(freqs > cutoffs[N]))
        filters[l_ind:nfreqs + 1, N + 1] = np.sqrt(1 - np.power(filters[l_ind:nfreqs + 1, N], 2))
        return filters
