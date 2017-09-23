function [pca_FV] = comp_pca_coeffs(avgFace, eigenVector, ev_st, ev_end, images)

[~, nSubjects] = size(images);

eigenVector = eigenVector(:,ev_st:ev_end);
nEigs = size(eigenVector,2);
pca_FV = zeros(nEigs, nSubjects);

for i = 1 : nSubjects
    currImage = images(:, i);
    h = currImage(:) - avgFace;
    currF = eigenVector'*h;
    currF = currF / norm(currF);
    pca_FV(:, i) = currF;
end;

return;

