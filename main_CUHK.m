% Main code
clc; clear all; close all;
setparams;

CDL_CCCA_flag = 1;


load('CUHK_data.mat');
XH_t = train_photo;
XL_t = train_sketch;
XH_test = test_photo;
XL_test = test_sketch;
train_h_labels = train_photo_labels.';
train_l_labels = train_sketch_labels.';
test_h_labels = test_photo_labels.';
test_l_labels = test_sketch_labels.';

XH_t = normc(XH_t);
XL_t = normc(XL_t);
XH_test = normc(XH_test);
XL_test = normc(XL_test);

% Can I do PCA on it ? 
% YES of course I can do it ...
ev_st = 1; ev_end = 87;
% *************************************************************************
% Do PCA
[eigenVector_nir, avgVect_nir] = gen_PCASpace(XH_t);
[eigenVector_vis, avgVect_vis] = gen_PCASpace(XL_t);

pca_nir = comp_pca_coeffs(avgVect_nir, eigenVector_nir, ev_st, ev_end, XH_t);
pca_vis = comp_pca_coeffs(avgVect_vis, eigenVector_vis, ev_st, ev_end, XL_t);
XH_t = pca_nir;
XL_t = pca_vis;

pca_nir = comp_pca_coeffs(avgVect_nir, eigenVector_nir, ev_st, ev_end, XH_test);
pca_vis = comp_pca_coeffs(avgVect_vis, eigenVector_vis, ev_st, ev_end, XL_test);
XH_test = pca_nir;
XL_test = pca_vis;
% *************************************************************************

[recog_direct] = test_fn(XH_test,XL_test,test_h_labels,test_l_labels);

m =10;

knn = 1;
eta = 0.1;
% Coupled dictionary learning with CCCA (Proposed)
if (CDL_CCCA_flag == 1)
    disp('GCDL');
    
    option = 2;
    parfor k=1:m
        % GCDL 1
        % please use option = 2        
        [NIR_coeffs, VIS_coeffs] = main_CDL_CCA_mod(XH_t,XL_t,train_h_labels,train_l_labels,XH_test,XL_test,knn,eta,option);
        [recog_cdl_ccca2(k)] = test_fn(NIR_coeffs,VIS_coeffs,test_h_labels,test_l_labels)        
    end
    
    option = 4;
    parfor k=1:m
        % GCDL 2
        % please use option = 4        
        [NIR_coeffs, VIS_coeffs] = main_CDL_CCA_mod(XH_t,XL_t,train_h_labels,train_l_labels,XH_test,XL_test,knn,eta,option);
        [recog_cdl_ccca3(k)] = test_fn(NIR_coeffs,VIS_coeffs,test_h_labels,test_l_labels)        
    end    
    
end;

% % *************************************************************************
% % *************************************************************************
disp(['Accuracy of GCDL1 = ',num2str(mean(recog_cdl_ccca2)),'%']);
disp(['Accuracy of GCDL2 = ',num2str(mean(recog_cdl_ccca3)),'%']);
% disp(' ************************************************************************* ');
% disp(' ************************************************************************* ');