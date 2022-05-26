function p = smoothAll(p, filter_size)
N = size(p);
fs = 1 / 0.008;
time = 0:(1/fs):(256/fs)-(1/fs);
for i = 1 : N(2)
    p(:, i) = smooth(time, p(:, i), filter_size);
end
end