folder = {"1_20","4_20","Laplacian","n_Laplacian","GraphBior","GraphSS"};
line_spec = {'-r','--b',':m','-.k',':^',':v'};

figure;
hold on;
for i = 1:6
    temp_res = zeros(1,101);
    temp_res(1,1) = 500;
    x = linspace(0,1,101);
    
    for j = 0:4
        name = strcat(folder{1,i},strcat('/res_',strcat(num2str(j))));
        load(name);
        for k = 2:101
            temp_res(1,k) = temp_res(1,k) + res(1,k);
        end
    end
    temp_res = temp_res.*0.2;
    temp_spec = line_spec{1,i};
    if i <= 4
        plot(x,temp_res,temp_spec,'LineWidth', 2);
    elseif i == 5
        plot(x,temp_res,temp_spec,'LineWidth', 2,'Color',[0.9290 0.6940 0.1250],'LineWidth', 2,'MarkerSize',5,'MarkerIndices', 1:5:101);
    else
        plot(x,temp_res,temp_spec,'LineWidth', 2,'Color',[0.4660 0.6740 0.1880], 'LineWidth', 2,'MarkerSize',5,'MarkerIndices', 1:5:101);
    end
end

set(gca, 'XTick', 0:0.1:1,  'YTick', 0:5:50,...            
         'Xlim' ,[0 1],'Ylim' ,[0 50], ...               
         'Xticklabel',{0:0.1:1},...                        
         'Yticklabel',{0:5:50})  
legend('GIB-II(1,20)','GIB-II(4,20)','UL','NL','GraphBior','GraphSS','Location','northeast','Fontname','Times New Roman')
xlabel('Fraction of Top-N coefficients', 'Fontname','Helvetica')
ylabel('Average Relative Error (%)','Fontname','Helvetica')