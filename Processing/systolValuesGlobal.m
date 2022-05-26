function [maxs] = systolValuesGlobal(p)
maxs = [];
sizeP = size(p);
taille = sizeP(1);

for i = 1 : taille
    maxsp = systolValues3(p(i, :));
    maxs = [maxs; maxsp];
end
maxs = mean(maxs, 2);

end