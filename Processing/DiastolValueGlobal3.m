function [DBP] = DiastolValueGlobal3(p)

DBP = [];
sizeP = size(p);
N = sizeP(1);

    for i = 1 : N
        pp = p(i, :);
        mins = MinLocs3ondes(pp);
        DBP(i) = mean(mins);

    end 


DBP = DBP';
end
