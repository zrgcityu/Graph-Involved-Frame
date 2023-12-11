function f = hard_thresholding(f_w, sigma, d_level,Colorednodes)

for level = 1:d_level
   
    for i = 1:4
        temp = f_w{level}(Colorednodes{level,i},i);
        for j = 1:size(temp)
            if abs(temp(j))-sigma > 0
                ;
            else 
                temp(j)=0;
            end
        end
        f_w{level}(Colorednodes{level,i},i) = temp;
    end
    
end
f = f_w;
end