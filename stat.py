from scipy.stats import ttest_ind

# the average MAPE results of LSTM, FFNN, and GCRNN models for the London dataset
lstm1 = [13.411, 26.715, 8.484, 7.529, 12.3, 9.331, 13.107, 14.942]
gcrnn1 = [13.131, 21.009, 7.958, 8.400, 11.147, 9.429, 12.524, 13.827]
ffnn1 = [18.284, 25.402, 15.911, 12.048, 15.401, 14.537, 21.464, 23.284]

# the average MAPE results of LSTM, FFNN, and GCRNN models for the CBT dataset
lstm2 = [17.076, 18.189, 18.990, 19.356, 16.304, 17.068]
gcrnn2 = [15.678, 16.752, 16.333, 17.530, 15.151, 15.344]
ffnn2 = [16.347, 17.304, 17.189, 18.859, 15.860, 16.480]

print('[*] Statistical significance of the hypothesis that GCRNN has lower MAPE than LSTM in London dataset:', ttest_ind(gcrnn1, lstm1, alternative='less'))
print('[*] Statistical significance of the hypothesis that GCRNN has lower MAPE than FFNN in London dataset:', ttest_ind(gcrnn1, ffnn1, alternative='less'))
print('[*] Statistical significance of the hypothesis that GCRNN has lower MAPE than LSTM in CBT dataset:', ttest_ind(gcrnn2, lstm2, alternative='less'))
print('[*] Statistical significance of the hypothesis that GCRNN has lower MAPE than FFNN in CBT dataset:', ttest_ind(gcrnn2, ffnn2, alternative='less'))

