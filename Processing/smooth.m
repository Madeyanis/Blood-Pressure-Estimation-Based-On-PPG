function signal_smoothed = smooth(time, raw_signal, filter_size)
fps = 1 ./ mean(diff(time));
n = 1;
afilt = 1;
bfilt = ones(1,filter_size)/(filter_size);
dum1 = raw_signal(1);
sig2=raw_signal-dum1;
sig2 = filter(bfilt,afilt,sig2);
dum2=sig2(end);
sig2 = sig2-dum2;
sig2(end:-1:1) = filter(bfilt,afilt,sig2(end:-1:1));
signal_smoothed = sig2+dum1+dum2;
end