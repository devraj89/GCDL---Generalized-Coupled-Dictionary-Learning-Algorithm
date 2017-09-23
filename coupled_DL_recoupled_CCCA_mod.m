% Main Function of Coupled Dictionary Learning
% Input:
% Alphap,Alphas: Initial sparse coefficient of two domains
% Xh    ,Xl    : Image Data Pairs of two domains
% Dh    ,Dl    : Initial Dictionaries
% Wh    ,Wl    : Initial Projection Matrix
% par          : Parameters 
%
%
% Output
% Alphap,Alphas: Output sparse coefficient of two domains
% Dh    ,Dl    : Output Coupled Dictionaries
% Uh    ,Ul    : Output Projection Matrix for Alpha
% 

function [Alphah, Alphal, XH_t, XL_t, Dh, Dl, Wh, Wl, Uh, Ul, f] = coupled_DL_recoupled_CCCA_mod(Alphah, Alphal, XH_t, XL_t, Dh, Dl, Wh, Wl, par, label_h, label_l,knn,eta,option)
% coupled_DL_recoupled(Alphah, Alphal, XH_t, XL_t, Dh, Dl, Wh, Wl, par);
%% parameter setting

[dimX, numX]        =       size(XH_t);
dimY                =       size(Alphah, 1);
numD                =       size(Dh, 2);
rho                 =       par.rho;
lambda1             =       par.lambda1;
lambda2             =       par.lambda2;
mu                  =       par.mu;
sqrtmu              =       sqrt(mu);
nu                  =       par.nu;
nIter               =       par.nIter;
t0                  =       par.t0;
epsilon             =       par.epsilon;
param.lambda        = 	    lambda1; % not more than 20 non-zeros coefficients
param.lambda2       =       lambda2;
%param.mode          = 	    1;       % penalized formulation
param.approx=0;
param.K = par.K;
param.L = par.L;
f = 0;
%keyboard;
%% Initialize Us, Up as I
% initially Wl and Wh are the identity matrices
Ul = Wl; 
Uh = Wh; 

% Iteratively solve D A U

for t = 1 : 10    
    
    %% Updating Ws and Wp => Updating Us and Up
    % Find the transformation matrices using CCA    
    set_kapa_cca;
    
    % modifications
    if option==1
        [Wl,Wh,~] = cluster_cca_mod(full(Alphal),full(Alphah),label_l,label_h,kapa_cca,knn,eta);
    elseif option==2
        % GCDL 1
        [Wl,Wh,~] = cluster_cca_mod2(full(Alphal),full(Alphah),label_l,label_h,kapa_cca,knn,eta,0);
    elseif option==3
        [Wl,Wh,~] = cluster_cca_mod2(full(Alphal),full(Alphah),label_l,label_h,kapa_cca,knn,eta,1);
    elseif option==4
        % GCDL 2
        [Wl,Wh,~] = cluster_cca_mod3(full(Alphal),full(Alphah),label_l,label_h,kapa_cca,knn,eta,0);
    elseif option==5
        [Wl,Wh,~] = cluster_cca_mod3(full(Alphal),full(Alphah),label_l,label_h,kapa_cca,knn,eta,1);
    end
    Wl = real(Wl);
    Wh = real(Wh);
    
    Ul = Wl.';
    Uh = Wh.';    
    
    sub_id = unique(label_h);
    nSub = length(sub_id);
    Alphal_full = full(Alphal);
    Alphah_full = full(Alphah);
    Alphal_inclass = zeros(size(Alphal_full,1),nSub);
    Alphah_inclass = Alphal_inclass;
    Xl_inclass = -0.5*ones(nSub,length(label_l));
    Xh_inclass = -0.5*ones(nSub,length(label_h));        
    
    % Here I am normalizing the data
    for i = 1:length(label_h)
        normVal = norm(Uh*Alphah_full(:,i));
        Alphah_full(:,i) = Alphah_full(:,i)/normVal;    
    end;
    % Here I am normalizing the data
    for i = 1:length(label_l)
        normVal = norm(Ul*Alphal_full(:,i));
        Alphal_full(:,i) = Alphal_full(:,i)/normVal;
    end;
     
    for subNo = 1:nSub
        currSubId = sub_id(subNo);
        indexvect = find(label_l == currSubId);
        Alphal_inclass(:,subNo) = median(Alphal_full(:,indexvect(1:length(indexvect))),2);
        Xl_inclass(subNo,indexvect) = 0.8;        
        
        indexvect = find(label_h == currSubId);
        Alphah_inclass(:,subNo) = median(Alphah_full(:,indexvect(1:length(indexvect))),2);
        Xh_inclass(subNo,indexvect) = 0.8;        
    end;
    
    Ph = (Uh'*Ul*Alphal_inclass)';
    Pl = (Ul'*Uh*Alphah_inclass)';    
    
    %% Updating Alphas and Alphap
    % What Happens If I vary the parameters ?
    mu = 0.04;
    sqrtmu = sqrt(mu);
    
    % Remember that Xl_inclass is basically Kx and Pl is basically Px
    % Remember that Xh_inclass is basically Ky and Ph is basically Py
    % The way that Kx and Px are formed are a little different 
    % instead of Kx being (N1XN2) we make it as Kx(unique labels (N1) X N2)
    % So accordingly also Px is formed : for that Alphal_inclasss is used.
    % Instead of using all the aplha's data we basically select the
    % mean/median of that particular class using the supervised
    % information.
    % The code will thus run much faster 
    % From the paper it is given as Px = Ay.'*Ty.'*Tx (Now Tx and Ty are
    % the Ul and Uh) and instead of using the whole Ay we utilize a subset
    % of that only for faster computation
    
    % Note using the whole matrix works fine but them again it is also time
    % consuming
    
    param.lambda = 0.01;
    Alphal = mexLasso([XL_t; sqrtmu * Xl_inclass], [Dl; sqrtmu * Pl],param);
    
    param.lambda = 0.01;
    Alphah = mexLasso([XH_t; sqrtmu * Xh_inclass], [Dh; sqrtmu * Ph],param);
       
    dictSize = par.K;
    
    %% Updating Ds and Dp 
    for i=1:dictSize
       ai        =    Alphal(i,:);
       Y         =    XL_t-Dl*Alphal+Dl(:,i)*ai;
       di        =    Y*ai';
       di        =    di./(norm(di,2) + eps);
       Dl(:,i)    =    di;
    end

    for i=1:dictSize
       ai        =    Alphah(i,:);
       Y         =    XH_t-Dh*Alphah+Dh(:,i)*ai;
       di        =    Y*ai';
       di        =    di./(norm(di,2) + eps);
       Dh(:,i)    =    di;
    end
    
end
return;