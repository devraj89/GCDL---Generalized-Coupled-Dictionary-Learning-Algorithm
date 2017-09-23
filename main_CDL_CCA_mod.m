function[NIR_coeffs, VIS_coeffs] = main_CDL_CCA_mod(XH_t,XL_t,train_h_labels,train_l_labels,XH_test,XL_test,knn,eta,option)

setparams;

%% Parameters setting
par.mu = par.mu*1;

par.K 	= x;
param.K = x;
par.L	= x;
param.L = x;
param.lambda        = par.lambda1; % not more than 20 non-zeros coefficients
param.lambda2       = par.lambda2;
param.mode          = 2;   %2    % penalized formulation
param.approx=0;


mean_XH_t = mean(XH_t,2);
mean_XL_t = mean(XL_t,2);
XH_t = XH_t - repmat(mean_XH_t, [1 size(XH_t,2)]);
XL_t = XL_t - repmat(mean_XL_t, [1 size(XL_t,2)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Start our proposed algorithm on training samples %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Intialize D,A, and W(U)
% *************************************************************************
% keyboard;
maxVal = max(nVIS_PerSub,nNIR_PerSub);
minVal = min(nVIS_PerSub,nNIR_PerSub);
sameFlag = 0;

if (size(XH_t,2) > size(XL_t,2))
    sameFlag = 1;
elseif (size(XH_t,2) < size(XL_t,2))
    sameFlag = 2;
end;

nTrainSub = length(trainSub);
if (sameFlag == 1)
    XH_t_train = [];
    for i=1:nTrainSub
        st_pt = maxVal*(i-1)+1;
        end_pt = maxVal*i;
        curr_train = XH_t(:,st_pt:end_pt);
        XH_t_train = [XH_t_train curr_train(:,1:minVal)];
    end;
    XL_t_train = XL_t;
elseif (sameFlag == 2)
    XL_t_train = [];
    for i=1:nTrainSub
        st_pt = maxVal*(i-1)+1;
        end_pt = maxVal*i;
        curr_train = XL_t(:,st_pt:end_pt);
        XL_t_train = [XL_t_train curr_train(:,1:minVal)];
    end;
    XH_t_train = XH_t;
else
    XH_t_train = XH_t;
    XL_t_train = XL_t;
end;

% *************************************************************************

D = mexTrainDL([XH_t_train;XL_t_train], param);
Dh = D(1:size(XH_t_train,1),:);
Dl = D(size(XH_t_train,1)+1:end,:);

% what are these ?? 
Wl = eye(size(Dh, 2));
Wh = eye(size(Dl, 2));


Alphah = mexLasso([XH_t_train;XL_t_train], D, param);
Alphal = Alphah;
clear D;

% Iteratively solve D,A, and W (U)
[Alphah, Alphal, XH_t, XL_t, Dh, Dl, Wh, Wl, Uh, Ul, f] = coupled_DL_recoupled_CCCA_mod(Alphah, Alphal, XH_t, XL_t, Dh, Dl, Wh, Wl, par, train_h_labels', train_l_labels',knn,eta,option);
clear XH_t XL_t;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Finish our proposed alorithm on training samples %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Prework on testing samples
%load mat_files/case2_soma;
XH_t = XH_test;
XL_t = XL_test;

XH_t = XH_t - repmat(mean_XH_t, [1 size(XH_t,2)]);
XL_t = XL_t - repmat(mean_XL_t, [1 size(XL_t,2)]);


%% Project the testing samples on the common feature space
% Dh and Dl is the learned dictionary from the training stage
% Xh and Xl are the new testing data
% Learn the sparse coefficients from this two
Alphah = full(mexLasso(XH_t, Dh, param));
Alphal = full(mexLasso(XL_t, Dl, param));

% use the sparse coefficients and project it into the subspace by using the
% learned projection matrices of the training stage
NIR_coeffs = Uh * Alphah;
VIS_coeffs = Ul * Alphal;
return;