"""
This module encapsulates the rapt function, which runs a pitch tracker
based on David Talkin's Robust Algorithm for Pitch Tracking (RAPT).
"""

import math
import numpy
from scipy import signal
from scipy.io import wavfile

#import raptparams
#import nccfparams


def rapt(wavfile_path, **kwargs):
    """
    F0 estimator inspired by RAPT algorithm to determine vocal
    pitch of an audio sample.
    """
    # Process optional keyword args and build out rapt params
    param = _setup_rapt_params(kwargs)

    # TODO: Flesh out docstring, describe args, expected vals in kwargs
    original_audio = _get_audio_data(wavfile_path)

    if param.is_two_pass_nccf:
        # downsample audio and run nccf on that first
        downsampled_audio = _get_downsampled_audio(original_audio,
                                                   param.maximum_allowed_freq,
                                                   param.is_run_filter)
        # calculate parameters for RAPT with input audio
        _calculate_params(param, original_audio, downsampled_audio)
        # get f0 candidates using nccf
        nccf_results = _run_nccf(original_audio, param, downsampled_audio)
    else:
        _calculate_params(param, original_audio)
        nccf_results = _run_nccf(original_audio, param)

    # Dynamic programming - determine voicing state at each period candidate
    freq_estimate = _get_freq_estimate(nccf_results[0], param,
                                       original_audio[0])

    # TODO: this is mainly for demo / niceness - don't keep this forever
    # filter out high freq points
    for i, item in enumerate(freq_estimate):
        if item > 500.0:
            freq_estimate[i] = 0.0

    # return output of nccf for now
    return freq_estimate


def rapt_with_nccf(wavfile_path, **kwargs):
    """
    F0 estimator inspired by RAPT algorithm to determine vocal
    pitch of an audio sample.
    """
    # Process optional keyword args and build out rapt params
    param = _setup_rapt_params(kwargs)

    # TODO: Flesh out docstring, describe args, expected vals in kwargs
    original_audio = _get_audio_data(wavfile_path)

    if param.is_two_pass_nccf:
        downsampled_audio = _get_downsampled_audio(original_audio,
                                                   param.maximum_allowed_freq,
                                                   param.is_run_filter)
        # calculate parameters for RAPT with input audio
        _calculate_params(param, original_audio, downsampled_audio)
        # get f0 candidates using nccf
        nccf_results = _run_nccf(original_audio, param, downsampled_audio)
    else:
        _calculate_params(param, original_audio)
        nccf_results = _run_nccf(original_audio, param)

    # Dynamic programming - determine voicing state at each period candidate
    freq_estimate = _get_freq_estimate(nccf_results[0], param,
                                       original_audio[0])
    # TODO: this is mainly for demo / niceness - don't keep this forever
    # filter out high freq points
    for i, item in enumerate(freq_estimate):
        if item > 500.0:
            freq_estimate[i] = 0.0

    # return output of nccf for now
    return (nccf_results, freq_estimate)


def _setup_rapt_params(kwargs):
    # Use optional args for RAPT parameters otherwise use defaults
    params = raptparams.Raptparams()
    if kwargs is not None and isinstance(kwargs, dict):
        for key, value in kwargs.items():
            setattr(params, key, value)
    return params


def _calculate_params(param, original_audio, downsampled_audio=None):
    param.original_audio = original_audio
    if downsampled_audio:
        param.sample_rate_ratio = (float(original_audio[0]) /
                                   float(downsampled_audio[0]))
    param.samples_per_frame = int(round(param.frame_step_size *
                                        original_audio[0]))
    param.hanning_window_length = int(round(0.03 * original_audio[0]))
    param.hanning_window_vals = numpy.hanning(param.hanning_window_length)
    # offset adjusts window centers to be 20ms apart regardless of frame
    # step size - so the goal here is to find diff btwn frame size & 20ms apart
    param.rms_offset = int(round((((float(original_audio[0]) / 1000.0) * 20.0) -
                           param.samples_per_frame)))


def _get_audio_data(wavfile_path):
    # Read wavfile and convert to mono
    sample_rate, audio_sample = wavfile.read(wavfile_path)

    # TODO: investigate whether this type of conversion to mono is suitable:
    if len(audio_sample.shape) > 1:
        audio_sample = audio_sample[:, 0]/2.0 + audio_sample[:, 1]/2.0
        audio_sample = audio_sample.astype(int)

    return (sample_rate, audio_sample)


def _get_downsampled_audio(original_audio, maximum_allowed_freq, is_filter):
    """
    Calc downsampling rate, downsample audio, return as tuple
    """
    downsample_rate = _calculate_downsampling_rate(original_audio[0],
                                                   maximum_allowed_freq)
    # low pass filter:
    if is_filter:
        freq_cutoff = 0.05 / (0.5 * float(downsample_rate))
        taps = 100
        filter = signal.firwin(taps, cutoff=freq_cutoff, width=0.005,
                               window='hanning')
        filtered_audio = signal.lfilter(filter, 1, original_audio[1])
        filtered_audio = (original_audio[0], filtered_audio)
        downsampled_audio = _downsample_audio(filtered_audio, downsample_rate)
    else:
        downsampled_audio = _downsample_audio(original_audio, downsample_rate)

    return (downsample_rate, downsampled_audio)


def _downsample_audio(original_audio, downsampling_rate):
    """
    Given the original audio sample/rate and a desired downsampling
    rate, returns a downsampled version of the audio input.
    """
    # TODO: look into applying low pass filter prior to downsampling, as
    # suggested in rapt paper.
    try:
        sample_rate_ratio = float(downsampling_rate) / float(original_audio[0])
    except ZeroDivisionError:
        raise ValueError('Input audio sampling rate is zero. Cannot determine '
                         'downsampling ratio.')
    # resample audio so it only uses a fraction of the original # of samples:
    number_of_samples = len(original_audio[1]) * sample_rate_ratio
    downsampled_audio = signal.resample(original_audio[1], number_of_samples)

    return downsampled_audio


def _calculate_downsampling_rate(initial_sampling_rate, maximum_f0):
    """
    Determines downsampling rate to apply to the audio input passed for
    RAPT processing
    """

    """
    NOTE: Using Python 2.7 so division is integer division by default
    Different default behavior in in Python 3+. That said, keeping the
    round() around the denominator of this formula as it is specified in
    the formula in David Talkin's paper:
    """
    try:
        aReturn = (initial_sampling_rate /
                   round(initial_sampling_rate / (4 * maximum_f0)))
    except ZeroDivisionError:
        raise ValueError('Ratio of sampling rate and max F0 leads to '
                         'division by zero. Cannot perform 1st pass of nccf '
                         'on downsampled audio.')
    return int(aReturn)


# NCCF Functionality:
# TODO: Consider moving nccf functions into a separate module / file?


def _run_nccf(original_audio, raptparam, downsampled_audio=None):
    if raptparam.is_two_pass_nccf:
        first_pass = _first_pass_nccf(downsampled_audio, raptparam)
        # run second pass
        nccf_results = _second_pass_nccf(original_audio, first_pass, raptparam)
        return (nccf_results, first_pass)
    else:
        nccf_results = _one_pass_nccf(original_audio, raptparam)
        return (nccf_results, None)


def _one_pass_nccf(audio, raptparam):
    """
    Runs NCCF on full audio sample and returns top correlations per frame
    """
    nccfparam = _get_nccf_params(audio, raptparam, True)
    params = (raptparam, nccfparam)
    # TODO: do i really use a -1 here?
    # Difference between "K-1" and starting value of "k"
    lag_range = ((params[1].longest_lag_per_frame - 1) -
                 params[1].shortest_lag_per_frame)
    candidates = [None] * params[1].max_frame_count

    for i in xrange(0, params[1].max_frame_count):
        all_lag_results = _get_correlations_for_all_lags(audio, i, lag_range,
                                                         params)
        candidates[i] = _get_marked_results(all_lag_results, params, False)

    return candidates


def _first_pass_nccf(audio, raptparam):
    # Runs normalized cross correlation function (NCCF) on downsampled audio,
    # outputting a set of potential F0 candidates that could be used to
    # determine the pitch at each given frame of the audio sample.

    nccfparam = _get_nccf_params(audio, raptparam, True)
    params = (raptparam, nccfparam)

    # Difference between "K-1" and starting value of "k"
    lag_range = ((params[1].longest_lag_per_frame - 1) -
                 params[1].shortest_lag_per_frame)

    # TODO: Re-read discussion of using double-precision arithmetic in rapt 3.3

    # NOTE: Because we are using max_frame_count exclusively for array size,
    # we do not run into issues with using xrange to iterate thru each frame, i

    candidates = [None] * params[1].max_frame_count

    for i in xrange(0, params[1].max_frame_count):
        candidates[i] = _get_firstpass_frame_results(
            audio, i, lag_range, params)

    return candidates


def _second_pass_nccf(original_audio, first_pass, raptparam):
    # Runs NCCF on original audio, but only for lags highlighted from first
    # pass results. Will output the finalized F0 candidates for each frame
    nccfparam = _get_nccf_params(original_audio, raptparam, False)
    params = (raptparam, nccfparam)

    # Difference between "K-1" and the starting value of "k"
    lag_range = ((params[1].longest_lag_per_frame - 1) -
                 params[1].shortest_lag_per_frame)

    candidates = [None] * params[1].max_frame_count

    for i in xrange(0, params[1].max_frame_count):
        candidates[i] = _get_secondpass_frame_results(
            original_audio, i, lag_range, params, first_pass)

    return candidates


def _get_nccf_params(audio_input, raptparams, is_firstpass):
    """
    Creates and returns nccfparams object w/ nccf-specific values
    """
    nccfparam = nccfparams.Nccfparams()
    # Value of "n" in NCCF equation:
    nccfparam.samples_correlated_per_lag = int(round(
        raptparams.correlation_window_size * audio_input[0]))
    # Starting value of "k" in NCCF equation:
    if(is_firstpass):
        nccfparam.shortest_lag_per_frame = int(round(audio_input[0] /
                                               raptparams.maximum_allowed_freq))
    else:
        nccfparam.shortest_lag_per_frame = 0
    # Value of "K" in NCCF equation
    nccfparam.longest_lag_per_frame = int(round(audio_input[0] /
                                          raptparams.minimum_allowed_freq))
    # Value of "z" in NCCF equation
    nccfparam.samples_per_frame = int(round(raptparams.frame_step_size *
                                      audio_input[0]))
    # TODO: do i really need to use the -1 here?
    # Value of "M-1" in NCCF equation:
    nccfparam.max_frame_count = int(round(float(len(audio_input[1])) /
                                    float(nccfparam.samples_per_frame)) - 1)
    return nccfparam


def _get_firstpass_frame_results(audio, current_frame, lag_range, params):
    # calculate correlation (theta) for all lags, and get the highest
    # correlation val (theta_max) from the calculated lags:
    all_lag_results = _get_correlations_for_all_lags(audio, current_frame,
                                                     lag_range, params)

    marked_values = _get_marked_results(all_lag_results, params, True)
    return marked_values


def _get_secondpass_frame_results(audio, current_frame, lag_range, params,
                                  first_pass):

    lag_results = _get_correlations_for_input_lags(audio, current_frame,
                                                   first_pass,  lag_range,
                                                   params)

    marked_values = _get_marked_results(lag_results, params, False)
    return marked_values


def _get_correlations_for_all_lags(audio, current_frame, lag_range, params):
    # Value of theta_max in NCCF equation, max for the current frame
    candidates = [0.0] * lag_range
    max_correlation_val = 0.0
    for k in xrange(0, lag_range):
        current_lag = k + params[1].shortest_lag_per_frame

        # determine if the current lag value causes us to go past the
        # end of the audio sample - if so - skip and set val to 0
        if ((current_lag + (params[1].samples_correlated_per_lag - 1)
             + (current_frame * params[1].samples_per_frame)) >= len(audio[1])):
            # candidates[k] = 0.0
            # TODO: Verify this behavior in unit test - no need to set val
            # since 0.0 is default
            continue

        candidates[k] = _get_correlation(audio, current_frame,
                                         current_lag, params)

        if candidates[k] > max_correlation_val:
            max_correlation_val = candidates[k]

    return (candidates, max_correlation_val)


def _get_correlations_for_input_lags(audio, current_frame, first_pass,
                                     lag_range, params):
    candidates = [0.0] * lag_range
    max_correlation_val = 0.0
    sorted_firstpass_results = first_pass[current_frame]
    sorted_firstpass_results.sort(key=lambda tup: tup[0])
    for lag_val in sorted_firstpass_results:
        # 1st pass lag value has been interpolated for original audio sample:
        lag_peak = lag_val[0]

        # for each peak check the closest 7 lags (if proposed peak is ok):
        if lag_peak > 10 and lag_peak < lag_range - 11:
            for k in xrange(lag_peak - 10, lag_peak + 11):
                # determine if the current lag value causes us to go past the
                # end of the audio sample - if so - skip and set val to 0
                sample_range = (k + (params[1].samples_correlated_per_lag - 1) +
                                (current_frame * params[1].samples_per_frame))
                if sample_range >= len(audio[1]):
                    # TODO: Verify this behavior in unit test -
                    # no need to set val
                    # since 0.0 is default
                    continue
                candidates[k] = _get_correlation(audio, current_frame, k,
                                                 params, False)
                if candidates[k] > max_correlation_val:
                    max_correlation_val = candidates[k]

    return (candidates, max_correlation_val)


# TODO: this can be used for 2nd pass - use parameter to decide 1stpass run?
def _get_marked_results(lag_results, params, is_firstpass=True):
    # values that meet certain threshold shall be marked for consideration
    min_valid_correlation = (lag_results[1] * params[0].min_acceptable_peak_val)
    max_allowed_candidates = params[0].max_hypotheses_per_frame - 1

    candidates = []

    if is_firstpass:
        candidates = _extrapolate_lag_val(lag_results, min_valid_correlation,
                                          max_allowed_candidates, params)
    else:
        for k, k_val in enumerate(lag_results[0]):
            if k_val > min_valid_correlation:
                current_lag = k + params[1].shortest_lag_per_frame
                candidates.append((current_lag, k_val))

    # now check to see if selected candidates exceed max allowed:
    if len(candidates) > max_allowed_candidates:
        candidates.sort(key=lambda tup: tup[1], reverse=True)
        returned_candidates = candidates[0:max_allowed_candidates]
        # re-sort before returning so that it is in order of low to highest k
        returned_candidates.sort(key=lambda tup: tup[0])
    else:
        returned_candidates = candidates

    return returned_candidates


def _get_correlation(audio, frame, lag, params, is_firstpass=True):
    numpysum = numpy.sum
    samples = 0
    audio_sample = audio[1]
    samples_correlated_per_lag = params[1].samples_correlated_per_lag
    frame_start = frame * params[1].samples_per_frame
    final_correlated_sample = frame_start + samples_correlated_per_lag

    frame_sum = numpysum(audio_sample[frame_start:final_correlated_sample])
    mean_for_window = ((1.0 / float(samples_correlated_per_lag)) * frame_sum)

    audio_slice = audio_sample[frame_start:final_correlated_sample]
    lag_audio_slice = audio_sample[frame_start + lag:
                                   final_correlated_sample + lag]

    samples = numpysum((audio_slice - mean_for_window) *
                       (lag_audio_slice - mean_for_window))

    denominator_base = numpysum((audio_slice - float(mean_for_window))**2)
    denominator_lag = numpysum((lag_audio_slice - float(mean_for_window))**2)

    if is_firstpass and params[0].is_two_pass_nccf:
        denominator = math.sqrt(denominator_base * denominator_lag)
    else:
        denominator = ((denominator_base * denominator_lag) +
                       params[0].additive_constant)
        denominator = math.sqrt(denominator)

    return float(samples) / float(denominator)


def _extrapolate_lag_val(lag_results, min_valid_correlation,
                         max_allowed_candidates, params):
    extrapolated_cands = []

    if len(lag_results[0]) == 0:
        return extrapolated_cands
    elif len(lag_results[0]) == 1:
        current_lag = 0 + params[1].shortest_lag_per_frame
        new_lag = int(round(current_lag * params[0].sample_rate_ratio))
        extrapolated_cands.append((new_lag, lag_results[0][0]))
        return extrapolated_cands

    least_lag = params[0].sample_rate_ratio * params[1].shortest_lag_per_frame
    most_lag = params[0].sample_rate_ratio * params[1].longest_lag_per_frame
    for k, k_val in enumerate(lag_results[0]):
        if k_val > min_valid_correlation:
            current_lag = k + params[1].shortest_lag_per_frame
            new_lag = int(round(current_lag * params[0].sample_rate_ratio))
            if k == 0:
                # if at 1st lag value, interpolate using 0,0 input on left
                prev_lag = k - 1 + params[1].shortest_lag_per_frame
                new_prev = int(round(prev_lag * params[0].sample_rate_ratio))
                next_lag = (k + 1 + params[1].shortest_lag_per_frame)
                new_next = int(round(next_lag * params[0].sample_rate_ratio))
                lags = numpy.array([new_prev, new_lag, new_next])
                vals = numpy.array([0.0, k_val, lag_results[0][k+1]])
                para = numpy.polyfit(lags, vals, 2)
                final_lag = int(round(-para[1] / (2 * para[0])))
                final_corr = float(para[0] * final_lag**2 + para[1] *
                                   final_lag + para[2])
                if (final_lag < least_lag or final_lag > most_lag or
                        final_corr < -1.0 or final_corr > 1.0):
                    current_lag = k + params[1].shortest_lag_per_frame
                    new_lag = int(round(current_lag *
                                        params[0].sample_rate_ratio))
                    extrapolated_cands.append((new_lag, k_val))
                else:
                    extrapolated_cands.append((final_lag, final_corr))
            elif k == len(lag_results[0]) - 1:
                # if at last lag value, interpolate using 0,0 input on right
                next_lag = k + 1 + params[1].shortest_lag_per_frame
                new_next = int(round(next_lag * params[0].sample_rate_ratio))
                prev_lag = (k - 1 + params[1].shortest_lag_per_frame)
                new_prev = int(round(prev_lag * params[0].sample_rate_ratio))
                lags = numpy.array([new_prev, new_lag, new_next])
                vals = numpy.array([lag_results[0][k-1], k_val, 0.0])
                para = numpy.polyfit(lags, vals, 2)
                final_lag = int(round(-para[1] / (2 * para[0])))
                final_corr = float(para[0] * final_lag**2 + para[1] *
                                   final_lag + para[2])
                if (final_lag < least_lag or final_lag > most_lag or
                        final_corr < -1.0 or final_corr > 1.0):
                    current_lag = k + params[1].shortest_lag_per_frame
                    new_lag = int(round(current_lag *
                                        params[0].sample_rate_ratio))
                    extrapolated_cands.append((new_lag, k_val))
                else:
                    extrapolated_cands.append((final_lag, final_corr))
            else:
                # we are in middle of lag results - use left and right
                next_lag = (k + 1 + params[1].shortest_lag_per_frame)
                new_next = int(round(next_lag * params[0].sample_rate_ratio))
                prev_lag = (k - 1 + params[1].shortest_lag_per_frame)
                new_prev = int(round(prev_lag * params[0].sample_rate_ratio))
                lags = numpy.array([new_prev, new_lag, new_next])
                vals = numpy.array([lag_results[0][k-1], k_val,
                                   lag_results[0][k+1]])
                para = numpy.polyfit(lags, vals, 2)
                final_lag = int(round(-para[1] / (2 * para[0])))
                final_corr = float(para[0] * final_lag**2 + para[1] *
                                   final_lag + para[2])
                if (final_lag < least_lag or final_lag > most_lag or
                        final_corr < -1.0 or final_corr > 1.0):
                    current_lag = k + params[1].shortest_lag_per_frame
                    new_lag = int(round(current_lag *
                                        params[0].sample_rate_ratio))
                    extrapolated_cands.append((new_lag, k_val))
                else:
                    extrapolated_cands.append((final_lag, final_corr))

    return extrapolated_cands


# TODO: Try and get peaks instead of just taking the basic lag value, but
# don't introduce wildly different lag values.
def _get_peak_lag_val(lag_results, lag_index, params):
    current_lag = lag_index + params[1].shortest_lag_per_frame
    extrapolated_lag = int(round(current_lag * params[0].sample_rate_ratio))
    return (extrapolated_lag, lag_results[lag_index])

    # lag peak is the maxima of a given peak obtained by results
    # lag_peak = lag_index + params[1].shortest_lag_per_frame
    # x_vals = []
    # y_vals = []

    # if lag_index == 0:
    #    y_vals = lag_results[lag_index:lag_index + 3]
    #    x_vals = range(lag_peak, lag_peak+3)
    # elif lag_index == (len(lag_results)-1):
    #    y_vals = lag_results[lag_index-2:lag_index+1]
    #    x_vals = range(lag_peak-2, lag_peak+1)
    # else:
    #    y_vals = lag_results[lag_index-1:lag_index+2]
    #    x_vals = range(lag_peak-1, lag_peak+2)

    # parabolic_func = numpy.polyfit(x_vals, y_vals, 2)
    # return maxima of the parabola, shifted to appropriate lag value
    # lag_peak = -parabolic_func[1] / (2 * parabolic_func[0])
    # lag_peak = round(lag_peak * params[0].sample_rate_ratio)
    # lag_peak = int(lag_peak)
    # return (lag_peak, lag_results[lag_index])


# Dynamic Programming / Post-Processing:

# this method will obtain best candidate per frame and calc freq est per frame
def _get_freq_estimate(nccf_results, raptparam, sample_rate):
    results = []
    candidates = _determine_state_per_frame(nccf_results, raptparam,
                                            sample_rate)
    for candidate in candidates:
        if candidate > 0:
            results.append(sample_rate/candidate)
        else:
            results.append(0.0)
    return results


# this method will prepare to call a recursive function that will determine
# the optimal voicing state / candidate per frame
def _determine_state_per_frame(nccf_results, raptparam, sample_rate):
    candidates = []
    # Add unvoiced candidate entry per frame (tuple w/ 0 lag, 0 correlation)
    for result in nccf_results:
        result.append((0, 0.0))

    # now call recursive function that will calculate cost per candidate:
    final_candidates = _select_candidates(nccf_results, raptparam, sample_rate)
    # sort results - take the result with the lowest cost for its last item
    final_candidates.sort(key=lambda y: y[-1][0])

    # with the results, take the lag of the lowest cost candidate per frame
    for result in final_candidates[0]:
        # sort results - the first value, the cost, is used by default
        candidates.append(result[1][0])
    return candidates


def _select_candidates(nccf_results, params, sample_rate):
    # start by calculating frame 0:
    max_for_frame = _select_max_correlation_for_frame(nccf_results[0])
    frame_candidates = []
    for candidate in nccf_results[0]:
        best_cost = None
        local_cost = _calculate_local_cost(candidate, max_for_frame, params,
                                           sample_rate)
        for initial_candidate in [(0.0, (1, 0.1)), (0.0, (0, 0.0))]:
            delta_cost = _get_delta_cost(candidate, initial_candidate,
                                         0, params)
            total_cost = local_cost + delta_cost
            if best_cost is None or total_cost <= best_cost:
                best_cost = total_cost
        frame_candidates.append([(best_cost, candidate)])
    # now we have initial costs for frame 0. lets loop thru later frames
    final_candidates = _get_next_cands(1, frame_candidates, nccf_results,
                                       params, sample_rate)
    return final_candidates


def _get_next_cands(frame_idx, prev_candidates, nccf_results, params,
                    sample_rate):
    frame_max = _select_max_correlation_for_frame(nccf_results[frame_idx])
    final_candidates = []
    for candidate in nccf_results[frame_idx]:
        best_cost = None
        returned_path = None
        local_cost = _calculate_local_cost(candidate, frame_max, params,
                                           sample_rate)
        for prev_candidate in prev_candidates:
            delta_cost = _get_delta_cost(candidate, prev_candidate[-1],
                                         frame_idx, params)
            total_cost = local_cost + delta_cost
            if best_cost is None or total_cost <= best_cost:
                best_cost = total_cost
                returned_path = list(prev_candidate)
        returned_path.append((best_cost, candidate))
        final_candidates.append(returned_path)
    next_idx = frame_idx + 1
    if next_idx < len(nccf_results):
        return _get_next_cands(next_idx, final_candidates, nccf_results,
                               params, sample_rate)
    return final_candidates


def _select_max_correlation_for_frame(nccf_results_frame):
    maxval = 0.0
    for hypothesis in nccf_results_frame:
        if hypothesis[1] > maxval:
            maxval = hypothesis[1]
    return maxval


def _calculate_local_cost(candidate, max_corr_for_frame, params, sample_rate):
    # calculate local cost of hypothesis (d_i,j in RAPT)
    lag_val = candidate[0]
    correlation_val = candidate[1]
    if lag_val == 0 and correlation_val == 0.0:
        # unvoiced hypothesis: add VO_BIAS to largest correlation val in frame
        cost = params.voicing_bias + max_corr_for_frame
    else:
        # voiced hypothesis
        lag_weight = (float(params.lag_weight) / float(sample_rate /
                      float(params.minimum_allowed_freq)))
        cost = (1.0 - correlation_val * (1.0 - float(lag_weight)
                * float(lag_val)))
    return cost


# determine what type of transition for candidate and previous, and return delta
def _get_delta_cost(candidate, prev_candidate, frame_idx, params):
    # determine what type of transition:
    if _is_unvoiced(candidate) and _is_unvoiced(prev_candidate[1]):
        return _get_unvoiced_to_unvoiced_cost(prev_candidate)
    elif _is_unvoiced(candidate):
        return _get_voiced_to_unvoiced_cost(candidate, prev_candidate,
                                            frame_idx, params)
    elif _is_unvoiced(prev_candidate[1]):
        return _get_unvoiced_to_voiced_cost(candidate, prev_candidate,
                                            frame_idx, params)
    else:
        return _get_voiced_to_voiced_cost(candidate, prev_candidate, params)


# for a candidate tuple w/ lag and correlation value, determine if it is a
# placeholder for unvoiced hypothesis
def _is_unvoiced(candidate):
    return candidate[0] == 0 and candidate[1] == 0.0


# determines cost of voiced to voice delta w/ prev entry's global cost:
def _get_voiced_to_voiced_cost(candidate, prev_entry, params):
    numpylog = numpy.log
    prev_cost = prev_entry[0]
    prev_candidate = prev_entry[1]
    # value of epsilon in voiced-to-voiced delta formula:
    freq_jump_cost = numpylog(float(candidate[0]) / float(prev_candidate[0]))
    transition_cost = (params.freq_weight * (params.doubling_cost +
                       abs(freq_jump_cost - numpylog(2.0))))
    final_cost = prev_cost + transition_cost
    return final_cost


# delta cost of unvoiced to unvoiced is 0, so just return previous entry's
# global cost:
def _get_unvoiced_to_unvoiced_cost(prev_entry):
    return prev_entry[0] + 0.0


def _get_voiced_to_unvoiced_cost(candidate, prev_entry, frame_idx, params):
    prev_cost = prev_entry[0]
    # prev_candidate = prev_entry[1]
    # NOTE: Not using spec_mod / itakura distortion for delta cost
    # delta = (params.transition_cost + (params.spec_mod_transition_cost *
    #         _get_spec_stationarity()) + (params.amp_mod_transition_cost *
    #         _get_rms_ratio(sample_rate)))
    delta = (params.transition_cost + (params.amp_mod_transition_cost *
             _get_rms_ratio(frame_idx, params)))
    return prev_cost + delta


def _get_unvoiced_to_voiced_cost(candidate, prev_entry, frame_idx, params):
    prev_cost = prev_entry[0]
    # prev_candidate = prev_entry[1]
    # NOTE: Not using spec_mod / itakura distortion for delta cost
    # delta = (params.transition_cost + (params.spec_mod_transition_cost *
    #         _get_spec_stationarity()) + (params.amp_mod_transition_cost /
    #         _get_rms_ratio(sample_rate)))
    current_rms_ratio = _get_rms_ratio(frame_idx, params)

    # TODO: figure out how to better handle rms ratio on final frame
    if current_rms_ratio <= 0:
        return prev_cost + params.transition_cost

    delta = (params.transition_cost + (params.amp_mod_transition_cost /
             current_rms_ratio))
    return prev_cost + delta


# NOTE: this method is not being utilized for transition costs
# spectral stationarity function, denoted as S_i in the delta formulas:
def _get_spec_stationarity():
    # TODO: Figure out how to calculate this:
    itakura_distortion = 1
    return_val = 0.2 / (itakura_distortion - 0.8)
    return return_val


# RMS ratio, denoted as rr_i in the delta formulas:
def _get_rms_ratio(frame_idx, params):
    samples_per_frame = params.samples_per_frame
    rms_offset = params.rms_offset
    hanning_win_len = params.hanning_window_length
    hanning_win_vals = params.hanning_window_vals
    audio_sample = params.original_audio[1]
    curr_frame_start = frame_idx * samples_per_frame
    prev_frame_start = (frame_idx - 1) * samples_per_frame

    if prev_frame_start < 0:
        prev_frame_start = 0
    # TODO: determine if this adjustment is appropriate:
    # because of the offset we might go beyond the array of samples, so set
    # limit here:
    max_window_diff = (len(audio_sample) - (curr_frame_start +
                       rms_offset + hanning_win_len))
    if max_window_diff < 0:
        hanning_win_len += max_window_diff

    if hanning_win_len < 0:
        hanning_win_len = 0

    # use range(0,window_length) for sigma/summation (effectivey 0 to J-1)
    curr_sum = 0
    prev_sum = 0
    curr_frame_index = curr_frame_start + rms_offset
    prev_frame_index = prev_frame_start - rms_offset
    if prev_frame_index < 0:
        prev_frame_index = 0
    audio_slice = audio_sample[curr_frame_index:curr_frame_index +
                               hanning_win_len]
    prev_audio_slice = audio_sample[prev_frame_index:prev_frame_index +
                                    hanning_win_len]
    # since window len may be reduced (since we are end of sample), make sure
    # the hanning window vals match up with our slice of the audio sample
    hanning_win_val = hanning_win_vals[:hanning_win_len]

    curr_sum = numpy.sum((audio_slice * hanning_win_val)**2)
    prev_sum = numpy.sum((prev_audio_slice * hanning_win_val)**2)

    # TODO: Do a better job of handling the case where we are at the end
    # of the audio sample and the last frame has no samples to analyze for ratio
    if curr_sum == 0.0 and prev_sum == 0.0 and hanning_win_len == 0:
        return 0.0

    rms_curr = math.sqrt(float(curr_sum) / float(hanning_win_len))
    rms_prev = math.sqrt(float(prev_sum) / float(hanning_win_len))
    return (rms_curr / rms_prev)
