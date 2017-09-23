function[recog_nn] = test_fn(NIR_coeffs,VIS_coeffs,test_h_labels,test_l_labels)
% Compute scores and normalize
score_nn = pdist2(NIR_coeffs.',VIS_coeffs.','euclidean'); %euclidean
% Normalize score
for subNo =1:size(score_nn,2)
    score_nn(:,subNo) =  score_nn(:,subNo) / norm(score_nn(:,subNo),1);
end
% *************************************************************************
num_probe = size(NIR_coeffs,2);
rank1 = 0;
for k=1:num_probe
    finalScore = score_nn(k,:);
    [~,sortIndex] = sort(finalScore, 'ascend');
    gtLabel = test_h_labels(k);
    if (gtLabel == test_l_labels(sortIndex(1)))
        rank1 = rank1+1;
    end;
end
recog_nn = 100*rank1/num_probe;
return;