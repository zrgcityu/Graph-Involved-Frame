%% The following codes are adapted from the original codes of GraphBior
clear all
clc
close all

addpath Graph_Generators/
addpath Graph_Colorers/
addpath Graph_kernels/
addpath Bpt_Decomp_Algos/
addpath sgwt_toolbox/
addpath toolbox/
addpath Datasets/

%% Parameters
% max_level = 3; % number of decomposition levels
filter_type = 'exact'; % polynomial approximation of Meyer filters
norm_type='asym';

%% Section 1: Graph Formulation

% Minnesota traffic graph
[A xy] = Minnesota_traffic_graph(); % Minnesota Traffic Graph
% [A xy] = random_bipartite_graph(); % random bipartite graphs G(n,m,p)
A(348:349,:)=[];A(:,348:349)=[];
xy(348:349,:)=[];


%% Section 2: Bipartite subgraph decomposition

% Graph Coloring: F is the output coloring vector
%%%%%%%%%%%%%
% F = BSC(A); % Backtracking Sequential Coloring Algorithm: Exact but slow
% F = LF(A);   % Greedy Largest Degree First Coloring: Fast but inaccurate
% F = DSATUR(A); % Greedy Coloring based on improved heuristic: Moderate speed and accuracy
load min_coloring %  Preloaded Coloring for Minnesota graph
%%%%%%%%%%%%%%
F(348:349)=[];

% generate the downsampling functions
N = length(A);
[beta bptG beta_dist Colorednodes]= harary_decomp(A,F);
theta = size(beta,2);
Fmax = size(beta_dist,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute Normalized Laplacian Matrices for Each Bpt graph
disp('Computing normalized Laplacian matrices for each subgraph...');
switch norm_type
    case 'sym'
        Ln_bpt = zeros(N,N,theta);
        for i = 1:theta
            d1 = sum(bptG(:,:,i),2);
            d1(d1 == 0) = 1; % for isolated nodes
            d1_inv = d1.^(-0.5);
            D1_inv = diag(d1_inv);
            An = D1_inv*bptG(:,:,i)*D1_inv;
            An = 0.5*(An + An');
            Ln_bpt(:,:,i) = eye(N) - An;
            Ln_bpt(:,:,i) = (Ln_bpt(:,:,i) + (Ln_bpt(:,:,i))')./2;
        end
    case 'asym'
        Ln_bpt = zeros(N,N,theta);
        for i = 1:theta
            d1 = sum(bptG(:,:,i),2);
            d1(d1 == 0) = 1; % for isolated nodes
            d1_inv = d1.^(-1);
            D1_inv = diag(d1_inv);
            An = D1_inv*(0.5*(bptG(:,:,i) + bptG(:,:,i)'));
            Ln_bpt(:,:,i) = eye(size(An,1)) - An;
        end
        Ln_bptsym = zeros(N,N,theta);
        for i = 1:theta
            d1 = sum(bptG(:,:,i),2);
            d1(d1 == 0) = 1; % for isolated nodes
            d1_inv = d1.^(-0.5);
            D1_inv = diag(d1_inv);
            An = D1_inv*bptG(:,:,i)*D1_inv;
            An = 0.5*(An + An');
            Ln_bptsym(:,:,i) = eye(N) - An;
            Ln_bptsym(:,:,i) = (Ln_bptsym(:,:,i) + (Ln_bptsym(:,:,i))')./2;
        end
    otherwise
        disp('Unknown normalization option')
        return;
end

% design a low-pass kernel
% design biorthogonal kernels
Nd = 5;
Nr = 5;
% S = sprintf('%s%d%s ' , ' Computing a ',filterlen,' ^th order approximation of Meyer kernel');
% disp(S)
[lo_d,hi_d] = biorth_kernel_new(Nd,Nr); % Nd zeros for lowpass Nr zeros for highpass
% [hi_d,lo_d] = biorth_kernel(Nd,Nr);
filterlen_hi = length(roots(hi_d));
filterlen_lo = length(roots(lo_d));
g0 = @(x)(polyval(lo_d,x));
g1 = @(x)(polyval(hi_d,x));
h0 = @(x)(polyval(hi_d,2 - x));
h1 = @(x)(polyval(lo_d,2 - x));

arange = [0 2];
c_d{1}=sgwt_cheby_coeff(h0,filterlen_lo,filterlen_lo+1,arange);
c_d{2}=sgwt_cheby_coeff(h1,filterlen_hi,filterlen_hi+1,arange);
c_r{1}=sgwt_cheby_coeff(g0,filterlen_hi,filterlen_hi+1,arange);
c_r{2}=sgwt_cheby_coeff(g1,filterlen_lo,filterlen_lo+1,arange);


tot_SNR = 0;
sigma = 1/2;
for i = 0:4
    name = strcat("signals/2/signal_",strcat(num2str(i),".mat"));
    load(name);
    f = transpose(f);
    disp('Computing wavelet transform coefficients ...')
    switch filter_type
        case 'approximate'
            f_w = zeros(N,Fmax);
            Channel_Name = cell(Fmax,1);
            for i = 1:Fmax
                if ~isempty(Colorednodes{i})
                    tempf_w = f;
                    for j = 1: theta
                        if beta_dist(i,j) == 1
                            tempf_w = sgwt_cheby_op(tempf_w,Ln_bpt(:,:,j),c_d{1},arange);
                            Channel_Name{i} = strcat(Channel_Name{i},'L');
                        else
                            tempf_w = sgwt_cheby_op(tempf_w, Ln_bpt(:,:,j),c_d{2},arange);
                            Channel_Name{i} = strcat(Channel_Name{i},'H');
                        end
                    end
                    f_w(Colorednodes{i},i) = tempf_w(Colorednodes{i});
                end
            end
        case 'exact'
            switch norm_type
                case 'asym'
                    for i=1:size(Ln_bpt,3)
                        [U(:,:,i),D]=eig(full(Ln_bptsym(:,:,i)));
                        Lam(:,i)=diag(D);
                        U(:,:,i)=D1_inv*U(:,:,i);
                    end
                    f_w = zeros(N,Fmax);
                    Channel_Name = cell(Fmax,1);
                    for i = 1:Fmax
                        if ~isempty(Colorednodes{i})
                            tempf_w = f;
                            for j = 1: theta
                                if beta_dist(i,j) == 1
                                    tempf_w = U(:,:,j)*diag(h0(Lam(:,j)))*inv(U(:,:,j))*tempf_w;
                                    %tempf_w = sgwt_cheby_op(tempf_w,Ln_bpt(:,:,j),c_d{1},arange);
                                    Channel_Name{i} = strcat(Channel_Name{i},'L');
                                else
                                    tempf_w = U(:,:,j)*diag(h1(Lam(:,j)))*inv(U(:,:,j))*tempf_w;
    %                                 tempf_w = sgwt_cheby_op(tempf_w, Ln_bpt(:,:,j),c_d{2},arange);
                                    Channel_Name{i} = strcat(Channel_Name{i},'H');
                                end
                            end
                            f_w(Colorednodes{i},i) = tempf_w(Colorednodes{i});
                        end
                    end
                case 'sym'
                    for i=1:size(Ln_bpt,3)
                        [U(:,:,i),D]=eig(full(Ln_bpt(:,:,i)));
                        Lam(:,i)=diag(D);
                    end
                    f_w = zeros(N,Fmax);
                    Channel_Name = cell(Fmax,1);
                    for i = 1:Fmax
                        if ~isempty(Colorednodes{i})
                            tempf_w = f;
                            for j = 1: theta
                                if beta_dist(i,j) == 1
                                    tempf_w = U(:,:,j)*diag(h0(Lam(:,j)))*U(:,:,j)'*tempf_w;
    %                                 tempf_w = sgwt_cheby_op(tempf_w,Ln_bpt(:,:,j),c_d{1},arange);
                                    Channel_Name{i} = strcat(Channel_Name{i},'L');
                                else
                                    tempf_w = U(:,:,j)*diag(h1(Lam(:,j)))*U(:,:,j)'*tempf_w;
    %                                 tempf_w = sgwt_cheby_op(tempf_w, Ln_bpt(:,:,j),c_d{2},arange);
                                    Channel_Name{i} = strcat(Channel_Name{i},'H');
                                end
                            end
                            f_w(Colorednodes{i},i) = tempf_w(Colorednodes{i});
                        end
                    end
                otherwise
                    disp('Unknown normalization option')
                    return;
            end
    end



    
    f_w = hard_thresholding({f_w}, 3*sigma, 1, Colorednodes'); % threshold for nCoeffs best wavelet coefficients
    f_w=cell2mat(f_w);
    
    
    switch filter_type
        case 'exact'
            disp('Reconstructed signals using single channel coefficients ...')
            switch norm_type
                case 'asym'
                    f_hat = zeros(N,Fmax);
                    % Channel_Name = cell(Fmax,1);
                    for i = 1:Fmax
                        tempf_hat = f_w(:,i);
                        for j = theta:-1: 1
                            if beta_dist(i,j) == 1
                                tempf_hat = U(:,:,j)*diag(g0(Lam(:,j)))*inv(U(:,:,j))*tempf_hat;
                            else
                                tempf_hat = U(:,:,j)*diag(g1(Lam(:,j)))*inv(U(:,:,j))*tempf_hat;
                            end
                        end
                        f_hat(:,i) = tempf_hat;
                    end
                case 'sym'
                    f_hat = zeros(N,Fmax);
                    % Channel_Name = cell(Fmax,1);
                    for i = 1:Fmax
                        tempf_hat = f_w(:,i);
                        for j = theta:-1: 1
                            if beta_dist(i,j) == 1
                                tempf_hat = U(:,:,j)*diag(g0(Lam(:,j)))*U(:,:,j)'*tempf_hat;
                            else
                                tempf_hat = U(:,:,j)*diag(g1(Lam(:,j)))*U(:,:,j)'*tempf_hat;
                            end
                        end
                        f_hat(:,i) = tempf_hat;
                    end
                otherwise
                    disp('Unknown normalization option')
                    return;
            end
            
            
        case 'approximate'
            
            disp('Reconstructed signals using single channel coefficients ...')
            f_hat = zeros(N,Fmax);
            % Channel_Name = cell(Fmax,1);
            for i = 1:Fmax
                tempf_hat = f_w(:,i);
                for j = theta:-1: 1
                    if beta_dist(i,j) == 1
                        tempf_hat = sgwt_cheby_op(tempf_hat,Ln_bpt(:,:,j),c_r{1},arange);
                    else
                        tempf_hat = sgwt_cheby_op(tempf_hat, Ln_bpt(:,:,j),c_r{2},arange);
                    end
                end
                f_hat(:,i) = tempf_hat;
            end
            
        otherwise
    end
    f_hat = sum(f_hat,2);
    SNR = 10*log10(norm(f)/norm(f-f_hat))
    tot_SNR = tot_SNR + SNR;
end
tot_SNR/5


