function f = hard_thresholding(f_coe, sigma,max_level)
index = [1 2*[1:max_level-1]+1 2*max_level];
    for s = 1:length(index)
        
        temp = f_coe{index(s)};
     
        for j = 1:size(temp)
            if abs(temp(j)) - sigma > 0
                ;
            else
                temp(j) = 0;
            end
        end
        
        f{s} = temp;
    end

end