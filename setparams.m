% for CUHK database
trainSub = [1:88];
testSub = 100;
nVIS_PerSub = 1;
nNIR_PerSub = 1;
nVIS_PerSub_test = 1;
nNIR_PerSub_test = 1;
% % *************************************************************************
ev_st = 1;
ev_end = 87; % 80  % 180 for others
x = 70; %50

% For CDL
addpath(genpath('SPAMS'));
addpath(genpath('Data'));
addpath(genpath('YIQRGB'));

load params
param.K= 512;
param.lambda= 0.1; %0.0100;
param.L= 25;
       
par.nIter=20;
param.iter=10;
% *************************************************************************