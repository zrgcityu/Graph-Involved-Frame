%% Function for non-linear approximation
function f = top_N(f_coe, N, max_level)
index = [1 2*[1:max_level-1]+1 2*max_level];
lin_vec = [];
for s = 1:length(index)
    lin_vec = [lin_vec;f_coe{index(s)}];
    
end
%nCoeffs = floor(length(lin_vec)*nnz);
[lin_vec,idx] = sort(abs(lin_vec(:)),'descend');

flag = zeros(length(idx));
for i = 1:length(idx)
    if i <= N
        flag(idx(i)) = 1;
    end
end
cnt = 0;
for s = 1:length(index)
    
    temp = f_coe{index(s)};
 
    for j = 1:size(temp)
        cnt = cnt + 1;
        if flag(cnt) ~= 1
            temp(j) = 0;
        end
    end
    
    %     temp(isolated{level},:) = f_w{level}(isolated{level},2:4);
    f{s} = temp;
end

end