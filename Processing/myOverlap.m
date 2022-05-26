function p3 = myOverlap(p)
sizeP = size(p);
p3 = zeros(sizeP(1) - 2, 768);

for i = 1 : sizeP(1) - 2
   p3(i, :) = [p(i, :) p(i+1, :) p(i+2, :)];
end


end

