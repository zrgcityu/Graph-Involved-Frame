%% Function for non-linear approximation
function f = top_N(f_w, N, d_level,Colorednodes)

lin_vec = [];
for level = 1:d_level
    lin_vec = [lin_vec; f_w{level}(Colorednodes{level,1},1); f_w{level}(Colorednodes{level,2},2); f_w{level}(Colorednodes{level,3},3); f_w{level}(Colorednodes{level,4},4)];
end
[lin_vec,idx] = sort(abs(lin_vec(:)),'descend');

flag = zeros(1,size(idx,1));
for i = 1:size(idx,1)
    if i <= N
        flag(1,idx(i,1)) = 1;
    end
end
cnt = 0;
tot = 0;
for level = 1:d_level
   
    for i = 1:4
        temp = f_w{level}(Colorednodes{level,i},i);
        for j = 1:size(temp)
            cnt = cnt + 1;
            if flag(1,cnt) ~= 1
                temp(j) = 0;
                tot = tot + 1;
            end
        end
        f_w{level}(Colorednodes{level,i},i) = temp;
    end
    
end

f = f_w;
end