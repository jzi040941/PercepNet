#include "stdio.h"
#include <math.h>
#include <iostream>
#include <vector>

template<typename T>
std::vector<float> linspace(T start_in, T end_in, int num_in)
{

  std::vector<float> linspaced;

  float start = static_cast<float>(start_in);
  float end = static_cast<float>(end_in);
  float num = static_cast<float>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  float delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}

class ERBBand{
    float erb_low,erb_high;
    int bandN;
    std::vector<float> cutoffs,erb_lims;
    
    
    public:
    std::vector<int> nfftborder,centerfreqs;
      std::vector<std::pair<std::pair<int,int>,std::vector<float>>> filters;
      ERBBand(int window_size, int N, float low_lim, float high_lim){
          cutoffs.assign(N+2,0);
          int i;
          erb_low = freq2erb(low_lim);
          erb_high = freq2erb(high_lim);
          erb_lims = linspace(erb_low, erb_high, N+2);
          for(i=0; i<N+2; i++){
              cutoffs[i] = erb2freq(erb_lims[i]);
          }
          bandN = N;
          filters = make_filters(N);
      }

      float freq2erb(float freq_hz){
          return 9.265 * log(1+freq_hz/(24.7*9.265));
      }  
      float erb2freq(float n_erb){
          return 24.7 * 9.265 * (exp(n_erb/9.265) -1);
      }

      std::vector<std::pair<std::pair<int,int>,std::vector<float>>> make_filters(int N){
          std::vector<std::pair<std::pair<int,int>,std::vector<float>>> cos_filter;
          float freqRangePerBin = 50;//for 48000 smaplerate and 960 window_size fft
          float l_k, h_k, avg, rnge;
          int l_nfftind, h_nfftind;
          for(int k=0; k<N+2; k++){
            nfftborder.push_back((cutoffs[k]+25)/freqRangePerBin);//divide by 50 and round up
          }
          //impose mininum 100hz(2 nfft)
          for(int k=0; k<N; k++){
            if(nfftborder[k+1]-nfftborder[k]<2)
              nfftborder[k+1]+=(2-(nfftborder[k+1]-nfftborder[k]));
          }
          for(int k=0; k<N; k++){
              l_k = cutoffs[k];
              h_k = cutoffs[k+2];
              //impose minimum 100hz
              if(h_k-l_k < 100)
                  h_k=l_k + 100;
              
              l_nfftind = (int)(l_k/freqRangePerBin) +1;
              h_nfftind = (int)(h_k/freqRangePerBin);
              avg = (freq2erb(l_k) + freq2erb(h_k))/2;
              rnge = freq2erb(h_k) - freq2erb(l_k);
              std::pair<std::pair<int,int>,std::vector<float>> kthfilter = {{l_nfftind,h_nfftind+1},{}};
              //cos_filter.push_back()
              for(int i=l_nfftind; i<h_nfftind+1; i++)
                  kthfilter.second.push_back(cos( (freq2erb(freqRangePerBin*i)-avg)/rnge * M_PI )  );
              cos_filter.push_back(kthfilter);
          }
          for (auto kthfilter : cos_filter){
              
          }
          return cos_filter;
      }

};
