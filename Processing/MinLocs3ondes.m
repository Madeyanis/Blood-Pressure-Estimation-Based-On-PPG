function [mins] = MinLocs3ondes(p)
    
    [maxs locs_maxs] = MaxsLocs3ondes(p);
    mins = [];
    locs_maxs = sort(locs_maxs, 'ascend');

    mins(1) = p(1);
    mins(4) = p(end);

    a= findpeaks(-p);
    a = sort(a, 'descend');
    mins(2) = abs(a(2));
    mins(3) = abs(a(1));

end