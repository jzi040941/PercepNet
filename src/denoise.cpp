#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define _USE_MATH_DEFINES
#include <cmath>

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include "kiss_fft.h"
#include "rnnoise.h"
#include "common.h"
#include "pitch.h"
#include "erbband.h"
#include "nnet_data.h"

#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)

#define COMB_M 3

#define PITCH_MIN_PERIOD 60
#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960

#define FRAME_LOOKAHEAD 5 //round(PITCH_MAX_PERIOD*COMB_M/WINDOW_SIZE + 0.5)
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)
#define FRAME_LOOKAHEAD_SIZE (FRAME_LOOKAHEAD*FRAME_SIZE)
#define COMB_BUF_SIZE (FRAME_LOOKAHEAD*2*FRAME_SIZE+PITCH_FRAME_SIZE)
#define SQUARE(x) ((x)*(x))

#define NB_BANDS 34

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (NB_BANDS*2+2)
#define NORM_RATIO 32768

#ifndef TEST
#define TEST 1
#endif

#if !TRAINING
extern const RNNModel percepnet_model_orig;
#endif

int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;

static const opus_int16 eband5ms[] = {
/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};

typedef struct {
  int init;
  kiss_fft_state *kfft;
  float half_window[FRAME_SIZE];
  float dct_table[NB_BANDS*NB_BANDS];
  float comb_hann_window[COMB_M*2+1];
  float power_noise_attenuation;
  float n0;/*noise-masking-tone threshold*/
} CommonState;

struct DenoiseState {
  float analysis_mem[FRAME_SIZE];
  //float cepstral_mem[CEPS_MEM][NB_BANDS];
  int memid;
  float synthesis_mem[FRAME_SIZE];
  float pitch_buf[PITCH_BUF_SIZE];
  float comb_buf[COMB_BUF_SIZE];
  float pitch_enh_buf[PITCH_BUF_SIZE];
  float last_gain;
  int last_period;
  float pitch_corr;
  float mem_hp_x[2];
  //float lastg[NB_BANDS];
  RNNState rnn;
};

ERBBand *erb_band = new ERBBand(WINDOW_SIZE, NB_BANDS-2, 0/*low_freq*/, 20000/*high_freq*/);

void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (erb_band->nfftborder[i+1]-erb_band->nfftborder[i]);
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = SQUARE(X[(erb_band->nfftborder[i]) + j].r);
      tmp += SQUARE(X[(erb_band->nfftborder[i]) + j].i);
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
    /*
    //ERBBand cosfilter is not working in interp_gain
    int low_nfft_idx = erb_band->filters[i].first.first;
    int high_nfft_idx = erb_band->filters[i].first.second;
    for(j=low_nfft_idx; j<high_nfft_idx; j++){
      float tmp;
      tmp = SQUARE(X[j].r);
      tmp += SQUARE(X[j].i);
      sum[i] += tmp*erb_band->filters[i].second[j-low_nfft_idx];
    }
    */
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sqrt(sum[i]);
  }
}

void compute_band_corr(float *bandE, const kiss_fft_cpx *X, const kiss_fft_cpx *P) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (erb_band->nfftborder[i+1]-erb_band->nfftborder[i]);
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = X[(erb_band->nfftborder[i]) + j].r * P[(erb_band->nfftborder[i]) + j].r;
      tmp += X[(erb_band->nfftborder[i]) + j].i * P[(erb_band->nfftborder[i]) + j].i;
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
    /*
    int low_nfft_idx = erb_band->filters[i].first.first;
    int high_nfft_idx = erb_band->filters[i].first.second;
    for(j=low_nfft_idx; j<high_nfft_idx; j++){
      float tmp;
      tmp = X[j].r * P[j].r;
      tmp += X[j].i * P[j].i;
      sum[i] += tmp*erb_band->filters[i].second[j-low_nfft_idx];

    }
    */
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
 
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (erb_band->nfftborder[i+1]-erb_band->nfftborder[i]);
    for (j=0;j<band_size;j++) {
      float frac = (float)j/band_size;
      g[(erb_band->nfftborder[i]) + j] = (1-frac)*bandE[i] + frac*bandE[i+1];
    }
    /*
    int low_nfft_idx = erb_band->filters[i].first.first;
    int high_nfft_idx = erb_band->filters[i].first.second;
    for(j=low_nfft_idx; j<high_nfft_idx; j++){
      g[j] += bandE[i]/erb_band->filters[i].second[j-low_nfft_idx];
    }
    */
  }
}

CommonState common;

static void check_init() {
  int i;
  float temp_sum=0;
  if (common.init) return;
  common.kfft = opus_fft_alloc_twiddles(2*FRAME_SIZE, NULL, NULL, NULL, 0);
  for (i=0;i<FRAME_SIZE;i++)
    common.half_window[i] = sin(.5*M_PI*sin(.5*M_PI*(i+.5)/FRAME_SIZE) * sin(.5*M_PI*(i+.5)/FRAME_SIZE));
  for (i=0;i<NB_BANDS;i++) {
    int j;
    for (j=0;j<NB_BANDS;j++) {
      common.dct_table[i*NB_BANDS + j] = cos((i+.5)*j*M_PI/NB_BANDS);
      if (j==0) common.dct_table[i*NB_BANDS + j] *= sqrt(.5);
    }
  }
  for (i=1;i<COMB_M*2+2; i++){
    common.comb_hann_window[i-1] = 0.5 - 0.5*cos(2.0*M_PI*i/(COMB_M*2+2));
    temp_sum += common.comb_hann_window[i-1];
  }
  for (i=1;i<COMB_M*2+2; i++){
    common.comb_hann_window[i-1] /= temp_sum;
  }
  common.power_noise_attenuation = 0;
  for (i=1;i<COMB_M*2+2; i++){
    common.power_noise_attenuation += common.comb_hann_window[i-1]*common.comb_hann_window[i-1];
  }
  common.n0 = 0.03;

  common.init = 1;
}

DenoiseState *rnnoise_create(RNNModel *model) {
  DenoiseState *st;
  st = (DenoiseState*)malloc(rnnoise_get_size());
  rnnoise_init(st, model);
  return st;
}

int rnnoise_init(DenoiseState *st, RNNModel *model) {
  memset(st, 0, sizeof(*st));
  
  if (model)
    st->rnn.model = model;
  else
  {
    #if !TRAINING
    st->rnn.model = &percepnet_model_orig;
    st->rnn.first_conv1d_state = (float*)calloc(sizeof(float), st->rnn.model->conv1->kernel_size*st->rnn.model->conv1->nb_inputs);
    st->rnn.second_conv1d_state = (float*)calloc(sizeof(float), st->rnn.model->conv2->kernel_size*st->rnn.model->conv2->nb_inputs);
    st->rnn.gru1_state = (float*)calloc(sizeof(float), st->rnn.model->gru1->nb_neurons);
    st->rnn.gru2_state = (float*)calloc(sizeof(float), st->rnn.model->gru2->nb_neurons);
    st->rnn.gru3_state = (float*)calloc(sizeof(float), st->rnn.model->gru3->nb_neurons);
    st->rnn.gb_gru_state = (float*)calloc(sizeof(float), st->rnn.model->gru_gb->nb_neurons);
    st->rnn.rb_gru_state = (float*)calloc(sizeof(float), st->rnn.model->gru_rb->nb_neurons);
    #endif

  }
  
  return 0;
}

static void apply_window(float *x) {
  int i;
  check_init();
  for (i=0;i<FRAME_SIZE;i++) {
    x[i] *= common.half_window[i];
    x[WINDOW_SIZE - 1 - i] *= common.half_window[i];
  }
}

static void forward_transform(kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();
  for (i=0;i<WINDOW_SIZE;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }
  opus_fft(common.kfft, x, y, 0);
  for (i=0;i<FREQ_SIZE;i++) {
    out[i] = y[i];
  }
}

static void inverse_transform(float *out, const kiss_fft_cpx *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();
  for (i=0;i<FREQ_SIZE;i++) {
    x[i] = in[i];
  }
  for (;i<WINDOW_SIZE;i++) {
    x[i].r = x[WINDOW_SIZE - i].r;
    x[i].i = -x[WINDOW_SIZE - i].i;
  }
  opus_fft(common.kfft, x, y, 0);
  /* output in reverse order for IFFT. */
  out[0] = WINDOW_SIZE*y[0].r;
  for (i=1;i<WINDOW_SIZE;i++) {
    out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
  }
}

void rnnoise_destroy(DenoiseState *st) {
  //free(st->rnn.vad_gru_state);
  //free(st->rnn.noise_gru_state);
  //free(st->rnn.denoise_gru_state);
  free(st);
}

static void frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in) {
  int i;
  float x[WINDOW_SIZE];
  RNN_COPY(x, st->analysis_mem, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) x[FRAME_SIZE + i] = in[i];
  RNN_COPY(st->analysis_mem, in, FRAME_SIZE);
  apply_window(x);
  forward_transform(X, x);
#if TRAINING
  for (i=lowpass;i<FREQ_SIZE;i++)
    X[i].r = X[i].i = 0;
#endif
  compute_band_energy(Ex, X);
}

int rnnoise_get_size() {
  return sizeof(DenoiseState);
}

static void frame_synthesis(DenoiseState *st, float *out, const kiss_fft_cpx *y) {
  float x[WINDOW_SIZE];
  int i;
  inverse_transform(x, y);
  apply_window(x);
  for (i=0;i<FRAME_SIZE;i++) out[i] = x[i] + st->synthesis_mem[i];
  RNN_COPY(st->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
}

static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}
static int compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
                                  float *Ex, float *Ep, float *Exp, float *features, const float *in) {
  int i,k;
  float E = 0;
  float *ceps_0, *ceps_1, *ceps_2;
  float spec_variability = 0;
  float Ly[NB_BANDS];
  float p[WINDOW_SIZE];
  float pitch_buf[PITCH_BUF_SIZE>>1];
  int pitch_index;
  float pitch_corr;
  float gain;
  float *(pre[1]);
  float tmp[NB_BANDS];
  float follow, logMax;
  
  RNN_MOVE(st->comb_buf, &st->comb_buf[FRAME_SIZE], COMB_BUF_SIZE-FRAME_SIZE);
  RNN_COPY(&st->comb_buf[COMB_BUF_SIZE-FRAME_SIZE], in, FRAME_SIZE);
  
  
  for(int i=0; i<FRAME_SIZE; i++){
    celt_assert(st->comb_buf[COMB_BUF_SIZE-FRAME_SIZE+i] == in[i]);
  }
  
  RNN_MOVE(st->pitch_buf, &st->pitch_buf[FRAME_SIZE], PITCH_BUF_SIZE-FRAME_SIZE);
  RNN_COPY(&st->pitch_buf[PITCH_BUF_SIZE-FRAME_SIZE], &st->comb_buf[COMB_BUF_SIZE-FRAME_SIZE*(FRAME_LOOKAHEAD+1)], FRAME_SIZE);

  //float incombn[FRAME_SIZE];
  //RNN_COPY(incombn,&st->pitch_buf[PITCH_BUF_SIZE-FRAME_SIZE*4], FRAME_SIZE);

  frame_analysis(st, X, Ex, &st->comb_buf[COMB_BUF_SIZE-FRAME_SIZE*(FRAME_LOOKAHEAD+1)]); 
  
  pre[0] = &st->pitch_buf[0];
  pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);
  pitch_search(pitch_buf+(PITCH_MAX_PERIOD>>1), pitch_buf, PITCH_FRAME_SIZE,
               PITCH_MAX_PERIOD-3*PITCH_MIN_PERIOD, &pitch_index, &pitch_corr);
  pitch_index = PITCH_MAX_PERIOD-pitch_index;

  gain = remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
          PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
  st->last_period = pitch_index;
  st->last_gain = gain;
  st->pitch_corr = pitch_corr;

  for (i=0;i<WINDOW_SIZE;i++)
      p[i]=0;

  for (k=-COMB_M;k<COMB_M+1; k++){
    for (i=0;i<WINDOW_SIZE;i++)
      p[i] += st->comb_buf[COMB_BUF_SIZE-FRAME_SIZE*(FRAME_LOOKAHEAD)-WINDOW_SIZE-pitch_index*k+i]*common.comb_hann_window[k+COMB_M];
  } 
  apply_window(p);
  forward_transform(P, p);
  compute_band_energy(Ep, P);
  compute_band_corr(Exp, X, P);
  for (i=0;i<NB_BANDS;i++) Exp[i] = fmin(1,fmax(0,Exp[i]/sqrt(1e-15+Ex[i]*Ep[i])));

  for (i=0;i<NB_BANDS;i++) {
    E += Ex[i];
  }

  return E<0.1;
}

void pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g, const float *r) {
  int i;
  //float r[NB_BANDS];
  float rf[FREQ_SIZE] = {0};
  float inv_r[NB_BANDS] = {0};
  //FLOAT inv_rf[FREQ_SIZE] = {0};
  for (int i=0; i<NB_BANDS; i++){
    inv_r[i]=1-r[i];
  }
  /*
  for (i=0;i<NB_BANDS;i++) {
    
#if 0
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = Exp[i]*(1-g[i])/(.001 + g[i]*(1-Exp[i]));
    r[i] = MIN16(1, MAX16(0, r[i]));
#else
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = SQUARE(Exp[i])*(1-SQUARE(g[i]))/(.001 + SQUARE(g[i])*(1-SQUARE(Exp[i])));
    r[i] = sqrt(MIN16(1, MAX16(0, r[i])));
#endif
    r[i] *= sqrt(Ex[i]/(1e-8+Ep[i]));
  }
  */
  interp_band_gain(rf, inv_r);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r = rf[i]*X[i].r;
    X[i].i = rf[i]*X[i].i;
  }
  interp_band_gain(rf, r);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r += rf[i]*P[i].r;
    X[i].i += rf[i]*P[i].i;
  }
  /*
  float newE[NB_BANDS];
  compute_band_energy(newE, X);
  float norm[NB_BANDS];
  float normf[FREQ_SIZE]={0};
  for (i=0;i<NB_BANDS;i++) {
    norm[i] = sqrt(Ex[i]/(1e-8+newE[i]));
  }
  interp_band_gain(normf, norm);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r *= normf[i];
    X[i].i *= normf[i];
  }
  */
}

static void create_features(float* Ey_lookahead, float* pitch_coh, float T, float pitchcorr, float* features){
  RNN_COPY(&features[0], Ey_lookahead, NB_BANDS);
  RNN_COPY(&features[NB_BANDS], pitch_coh, NB_BANDS);
  //normalize
  for(int i=0; i<68; i++){
    features[i] = features[i]*30;
  }
  features[68] = T;
  features[69] = pitchcorr;
}

static void compute_lookahead_band_energy(DenoiseState *st, float *Ey_ahead){
  float y[WINDOW_SIZE];
  kiss_fft_cpx Y[WINDOW_SIZE];
  RNN_COPY(y, &st->comb_buf[COMB_BUF_SIZE-WINDOW_SIZE], WINDOW_SIZE);
  
  apply_window(y);
  forward_transform(Y, y);
  compute_band_energy(Ey_ahead, Y);
}

float rnnoise_process_frame(DenoiseState *st, float *out, const float *in, FILE* f_feature) {
  int i;
  kiss_fft_cpx X[FREQ_SIZE];
  kiss_fft_cpx P[WINDOW_SIZE];
  float x[FRAME_SIZE];
  float Ex[NB_BANDS], Ep[NB_BANDS], Ex_lookahead[NB_BANDS];
  float Exp[NB_BANDS];
  float features[NB_FEATURES];
  float g[NB_BANDS];
  float gf[FREQ_SIZE]={1};  
  float vad_prob = 0;
  float r[NB_BANDS];
  int silence;
  
  //static const float a_hp[2] = {-1.99599, 0.99600};
  //static const float b_hp[2] = {-2, 1};
  //biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
  silence = compute_frame_features(st, X, P, Ex, Ep, Exp, features, in);

  compute_lookahead_band_energy(st,Ex_lookahead);
  float T = (float)st->last_period/(PITCH_MAX_PERIOD-3*PITCH_MIN_PERIOD);
  float pitchcorr = st->pitch_corr;
  create_features(Ex_lookahead,Exp,T,pitchcorr,features);
  
  compute_rnn(&st->rnn,g,r,features);
  fwrite(g, sizeof(float), 34, f_feature);
  fwrite(r, sizeof(float), 34, f_feature);
  //r will be estimated by dnn
  if(!silence){
  pitch_filter(X, P, Ex, Ep, Exp, g, r);
  }
  interp_band_gain(gf, g);
    
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r *= gf[i];
    X[i].i *= gf[i];
  }
  frame_synthesis(st, out, X);
  return 0;
};

void estimate_phat_corr(CommonState st, float *Eyp, float *Ephatp){
  for(int i=0; i<NB_BANDS; i++){
    Ephatp[i] = Eyp[i]/sqrt((1-st.power_noise_attenuation)*pow(Eyp[i],2) + st.power_noise_attenuation);
  }
}

void filter_strength_calc(float *Exp, float *Eyp, float *Ephatp, float* r){
  float alpha;
  float a;
  float b;
  float c;
  for(int i=0; i<NB_BANDS; ++i){
    a = Ephatp[i]*Ephatp[i] - Exp[i]*Exp[i];
    if (a<0) a=0;
    b = Ephatp[i]*Eyp[i]*(1-Exp[i]*Exp[i]);
    c = Exp[i]*Exp[i]-Eyp[i]*Eyp[i];
    if (c<0) c=0;
    alpha = (sqrt(b*b + a *(c))-b)/(a+1e-8);
    r[i] = alpha/(1+alpha);
  }
}

void calc_ideal_gain(float *X, float *Y, float* g){
  for(int i=0; i<NB_BANDS; ++i){
    g[i] = X[i]/(.0001+Y[i]);
    if (g[i]>1) g[i] = 1;
    if (g[i]<0) g[i] = 0;
  }
}

void adjust_gain_strength_by_condition(CommonState st, float *Ephatp, float *Exp, float* g, float* r){
  float g_att;
  for(int i=0; i<NB_BANDS; ++i){
    if(Ephatp[i]<Exp[i])
    {
      g_att = sqrt((1+st.n0-Exp[i]*Exp[i])/(1+st.n0-Ephatp[i]*Ephatp[i]));
      r[i] = 0.99;
      g[i] *= g_att;
    }
  }
}


static float uni_rand() {
  return rand()/(double)RAND_MAX-.5;
}

static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

int train(int argc, char **argv) {
  int i;
  int count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_noise[2] = {0};
  float b_noise[2] = {0};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float mem_hp_x[2]={0};
  float mem_hp_n[2]={0};
  float mem_resp_x[2]={0};
  float mem_resp_n[2]={0};
  float x[FRAME_SIZE];
  float n[FRAME_SIZE];
  float xn[FRAME_SIZE];
  int vad_cnt=0;
  int gain_change_count=0;
  float speech_gain = 1, noise_gain = 1;
  FILE *f1, *f2, *f3;
  #ifdef TEST
  FILE *f4;
  FILE *f5;
  float out[FRAME_SIZE];
  short out_short[FRAME_SIZE];
  float gf[FREQ_SIZE]={1};
  #endif
  int maxCount;
  DenoiseState *st;
  DenoiseState *noise_state;
  DenoiseState *noisy;
  st = rnnoise_create(NULL);
  noise_state = rnnoise_create(NULL);
  noisy = rnnoise_create(NULL);
  if (argc!=5) {
    fprintf(stderr, "usage: %s <speech> <noisy> <count> <output>\n", argv[0]);
    return 1;
  }
  f1 = fopen(argv[1], "rb");
  f2 = fopen(argv[2], "rb");
  f3 = fopen(argv[4], "wb");
  #ifdef TEST 
  f4 = fopen("test_output.pcm", "wb");
  f5 = fopen("test_input.pcm","wb");
  #endif
  maxCount = atoi(argv[3]);
  //for(i=0;i<150;i++) {
  //  short tmp[FRAME_SIZE];
  //  fread(tmp, sizeof(short), FRAME_SIZE, f2);
  //}
  while (1) {
    kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], N[FREQ_SIZE], P[FREQ_SIZE];
    kiss_fft_cpx Phat[FREQ_SIZE];/*only for build*/
    float Ex[NB_BANDS], Ey[NB_BANDS], En[NB_BANDS], Ep[NB_BANDS], Ey_lookahead[NB_BANDS];
    float Ephat[NB_BANDS], Ephaty[NB_BANDS]; /*only for build*/
    float Exp[NB_BANDS], Eyp[NB_BANDS], Ephatp[NB_BANDS];
    float Ln[NB_BANDS];
    float features[NB_FEATURES];
    float g[NB_BANDS];
    float r[NB_BANDS];
    short tmp[FRAME_SIZE];
    float norm_tmp[FRAME_SIZE];
    float vad=0;
    float E=0;
    if (count==maxCount) break;
    //if ((count%1000)==0) fprintf(stderr, "%d\r", count);

    //DNS-Challenge Dataset can generate clean&noise data accroding to SNR,RIR setting
    //Disable gain change & Ignore lowpass filtering for convenience
    /*
    if (++gain_change_count > 2821) {
      speech_gain = pow(10., (-40+(rand()%60))/20.);
      noise_gain = pow(10., (-30+(rand()%50))/20.);
      if (rand()%10==0) noise_gain = 0;
      noise_gain *= speech_gain;
      if (rand()%10==0) speech_gain = 0;
      gain_change_count = 0;
      rand_resp(a_noise, b_noise);
      rand_resp(a_sig, b_sig);
      lowpass = FREQ_SIZE * 3000./24000. * pow(50., rand()/(double)RAND_MAX);
      for (i=0;i<NB_BANDS;i++) {
        if (eband5ms[i]<<FRAME_SIZE_SHIFT > lowpass) {
          band_lp = i;
          break;
        }
      }
    }
    */
    if (speech_gain != 0) {
      fread(tmp, sizeof(short), FRAME_SIZE, f1);
      if (feof(f1)) {
        rewind(f1);
        fread(tmp, sizeof(short), FRAME_SIZE, f1);
      }
      for (i=0;i<FRAME_SIZE;i++) norm_tmp[i] = ((float)tmp[i])/NORM_RATIO;
      for (i=0;i<FRAME_SIZE;i++) x[i] = speech_gain*norm_tmp[i];
      for (i=0;i<FRAME_SIZE;i++) E += norm_tmp[i]*(float)norm_tmp[i];
    } else {
      for (i=0;i<FRAME_SIZE;i++) x[i] = 0;
      E = 0;
    }
    if (noise_gain!=0) {
      fread(tmp, sizeof(short), FRAME_SIZE, f2);
      if (feof(f2)) {
        rewind(f2);
        fread(tmp, sizeof(short), FRAME_SIZE, f2);
      }
      for (i=0;i<FRAME_SIZE;i++) norm_tmp[i] = ((float)tmp[i])/NORM_RATIO;
      for (i=0;i<FRAME_SIZE;i++) n[i] = noise_gain*norm_tmp[i];
    } else {
      for (i=0;i<FRAME_SIZE;i++) n[i] = 0;
    }
    // biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
    // biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
    // biquad(n, mem_hp_n, n, b_hp, a_hp, FRAME_SIZE);
    // biquad(n, mem_resp_n, n, b_noise, a_noise, FRAME_SIZE);
    //for (i=0;i<FRAME_SIZE;i++) xn[i] = x[i] + n[i];
    // DNS challenge data is already mixed
    for (i=0;i<FRAME_SIZE;i++) xn[i] = n[i];
    #ifdef TEST
    for(int i=0; i<FRAME_SIZE; i++){
      out_short[i] = (short)fmax(-32768,fmin(32767, xn[i]*NORM_RATIO));
      //xn[i] = (float)out_short[i]/NORM_RATIO;
    }
    
    fwrite(out_short, sizeof(short), FRAME_SIZE, f5);
    #endif
    //frame_analysis(st, , Ey, x);
    //frame_analysis(noise_state, N, En, n);
    //for (i=0;i<NB_BANDS;i++) Ln[i] = log10(1e-2+En[i]);
    int silence = compute_frame_features(noisy, Y, Phat/*only use for Test*/, Ey, Ephat/*only use for Test*/, Ephaty, features, xn);
    compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);
    calc_ideal_gain(Ex, Ey, g);
    compute_band_corr(Eyp, Y, P);
    for (i=0;i<NB_BANDS;i++) Eyp[i] = fmin(1,fmax(0,Eyp[i]/sqrt(.001+Ey[i]*Ep[i])));
    estimate_phat_corr(common, Eyp, Ephatp);
    filter_strength_calc(Exp, Eyp, Ephatp, r);
    adjust_gain_strength_by_condition(common, Ephatp, Exp, g, r);
    
    #ifdef TEST
      if(!silence){
      pitch_filter(Y, Phat, Ey, Ephat, Ephaty, g, r);
      }
      interp_band_gain(gf, g);
      
      for (i=0;i<FREQ_SIZE;i++) {
        Y[i].r *= gf[i];
        Y[i].i *= gf[i];
      }
      
      frame_synthesis(st, out, Y);
      for(int i=0; i<FRAME_SIZE; i++){
        out_short[i] = (short)fmax(-32768,fmin(32767, out[i]*NORM_RATIO));
      }
      fwrite(out_short, sizeof(short), FRAME_SIZE, f4);
    #endif

    compute_lookahead_band_energy(noisy,Ey_lookahead);
    //fwrite(features, sizeof(float), NB_FEATURES, stdout);
    //fwrite(Ey, sizeof(float), NB_BANDS, f3);//Y(l+M)
    fwrite(Ey_lookahead, sizeof(float), NB_BANDS, f3);//Y(l+M)
    fwrite(Ephaty, sizeof(float), NB_BANDS, f3);//pitch coherence
    
    float T = (float)noisy->last_period/(PITCH_MAX_PERIOD-3*PITCH_MIN_PERIOD);
    float pitchcorr = noisy->pitch_corr;
    fwrite(&T, sizeof(float), 1, f3);//pitch
    fwrite(&pitchcorr, sizeof(float), 1, f3);//pitch correlation

    fwrite(g, sizeof(float), NB_BANDS, f3);//gain    
    fwrite(r, sizeof(float), NB_BANDS, f3);//filtering strength

    //fwrite(&vad, sizeof(float), 1, stdout);

    count++;
  }
  fclose(f1);
  fclose(f2);
  fclose(f3);
  #ifdef TEST
  fclose(f4);
  fclose(f5);
  #endif
  return 0;
}//
