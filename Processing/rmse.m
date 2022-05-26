function r = rmse(p, pr)

r =  sqrt(mean((p - pr).^2));

end