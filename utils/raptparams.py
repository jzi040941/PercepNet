"""
Simple object that stores parameters to the RAPT processing logic.
"""


class Raptparams:
    """
    Simple object that stores const parameters for use in RAPT equations
    """

    def __init__(self):
        # Boolean flag to determine if we should do 2-pass NCCF:
        self.is_two_pass_nccf = True

        # Boolean flag to run a low pass filter on the 1st pass of NCCF:
        self.is_run_filter = True

        # Value of "F0_max" in NCCF equation:
        self.maximum_allowed_freq = 500

        # Value of "F0_min" in NCCF equation:
        self.minimum_allowed_freq = 50

        # Value of "t" in NCCF equation:
        self.frame_step_size = 0.01

        # Value of "w" in NCCF equation:
        self.correlation_window_size = 0.0075

        # Value of "N_CANDS" in NCCF equation:
        self.max_hypotheses_per_frame = 20

        # Value of "CAND_TR" in NCCF equation:
        self.min_acceptable_peak_val = 0.3

        # Value of A_FACT" in NCCF equation:
        self.additive_constant = 10000

        # Value of VO_BIAS used when calculating voicing state:
        self.voicing_bias = 0.0

        # Value of DOUBL_C used when calculating voicing state:
        self.doubling_cost = 0.35

        # Value of LAG_WT used when calculating voicing state:
        self.lag_weight = 0.3

        # Value of FREQ_WT used when calculating voicing state:
        self.freq_weight = 0.02

        # Value of VTRAN_C used when calculating voicing state:
        self.transition_cost = 0.005

        # Value of VTR_A_C used when calculating voicing state:
        self.amp_mod_transition_cost = 0.5

        # Value of VTR_S_C used when calculating voicing state:
        self.spec_mod_transition_cost = 0.5

        # Below are fields populated by calculate_params method based on
        # audio inputs. There is no default value set for these:

        # sample rate ratio used to extrapolate 1st pass lags onto 2nd pass:
        self.sample_rate_ratio = None

        # audio being analyzed (tuple, the first item is sample rate)
        self.original_audio = None

        # frame step size (z in RAPT equations)
        self.samples_per_frame = None

        # hanning window length (used in RMS ratio calc for post processing)
        self.hanning_window_length = None

        # hanning window (actual values, used in RMS ratio calc)
        self.hanning_window_vals = None

        # value of h in rms ratio calc, used for + or - 20 ms offset
        self.rms_offset = None