load new_traffic.mat
lap_type = 'comb';
filter_type = 'orth';
max_level = 3;

G = gsp_graph(A);
switch lap_type
    case 'comb'
        %         L = sgwt_laplacian(A,'opt','raw');
        %         [U,~] = eig(full(L));
    case 'norm'
        G = gsp_create_laplacian(G, 'normalized');
        %         L = sgwt_laplacian(A,'opt','normalized');g
        %         [U,~] = eig(full(L));
end
G = gsp_compute_fourier_basis(G);

h0 = @(x)(meyer_kernel(x))/sqrt(2);
h1 = @(x)(meyer_kernel(2-x)/sqrt(2));
g0 = @(x)(meyer_kernel(x))/sqrt(2);
g1 = @(x)(meyer_kernel(2-x)/sqrt(2));

h = {h0, h1};
g = {g0, g1};


tot_SNR = 0;
sigma = 1/2;
for i = 0:4
    name = strcat("signals/2/signal_",strcat(num2str(i),".mat"));
    load(name);
    f = transpose(f);
    f_hat=G.U'*f;
    f_coe=cell(max_level+1,1);
    
    for P=0:max_level-1
        x=linspace(0,2,N/(2^(P)));
        if P == 0
            f_coe{1}=[eye(N/2) -fliplr(eye(N/2))]*diag(h{2}(x))*f_hat;
            f_coe{2}=[eye(N/2) fliplr(eye(N/2))]*diag(h{1}(x))*f_hat;
        else
            f_coe{2*P+1}=[eye(N/(2^(P+1))) -fliplr(eye(N/(2^(P+1))))]*diag(h{2}(x))*f_coe{2*P};
            f_coe{2*P+2}=[eye(N/(2^(P+1))) fliplr(eye(N/(2^(P+1))))]*diag(h{1}(x))*f_coe{2*P};
        end
    end


    f_coe_tmp = hard_thresholding(f_coe,3*sigma,max_level);


    f_rec_tmp{max_level+1}=f_coe_tmp{max_level+1};
    for P=max_level:-1:1
        
        x=linspace(0,2,N/(2^(P-1)));
        f_rec_L=diag(g{1}(x))*[eye(N/(2^P)) fliplr(eye(N/(2^P)))]'*f_rec_tmp{P+1};
        f_rec_H=diag(g{2}(x))*[eye(N/(2^P)) -fliplr(eye(N/(2^P)))]'*f_coe_tmp{P};
        f_rec_tmp{P}=(f_rec_L+f_rec_H);
    end
    f_rec=G.U*f_rec_tmp{1};

    SNR = 10*log10(norm(f)/norm(f-f_rec))
    tot_SNR = tot_SNR + SNR;
end    
tot_SNR/5


