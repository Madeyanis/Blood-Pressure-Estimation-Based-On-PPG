function [maxs_final locs_max_final] = MaxsLocs3ondes(p3)

    locs_max_final = [];
    maxs_final = [];
    [maxs locs_maxs] = findpeaks(p3);
    
    maxSorted = sort(maxs, 'descend');
    if length(maxs) > 3
        
        for i = 4 : length(maxSorted)
            maxSorted(i) = 0;
        end
    end
    
    for i = 1 : length(maxSorted)
    for j = 1 : length(maxs)
       if maxs(j) == maxSorted(i)
           locs_max_final = [locs_max_final locs_maxs(j)];
           maxs_final = [maxs_final maxs(j)];
       end
    end
    end

    
end
