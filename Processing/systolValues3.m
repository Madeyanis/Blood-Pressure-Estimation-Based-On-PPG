function [maxs] = systolValues3(p)
maxs = findpeaks(p);
maxs = sort(maxs, 'descend');
maxs = [maxs(1) maxs(2) maxs(3)];
end
