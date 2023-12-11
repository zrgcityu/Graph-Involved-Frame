function res = get_percentage(f)
    tot = sum(power(f,2));
    disp(["!!!",num2str(tot)]);
    temp_f = sort(abs(f),'descend');
    res = zeros(size(f,1),size(f,2)+1);
    cnt = 0;
    res(1,1) = 100;
    for i = 1:size(f,2)
        cnt = cnt + temp_f(1,i)*temp_f(1,i);
        res(1,i+1) = (1 - (cnt/tot))*100;
    end
end