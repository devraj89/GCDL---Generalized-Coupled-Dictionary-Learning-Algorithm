function[eigenVector, avgFace] = gen_PCASpace(images)
[~, nSubjects] = size(images);
avgFace = sum(images, 2)/nSubjects;
imageTrain = images - repmat(avgFace, [1, nSubjects]); 
L = imageTrain'*imageTrain;
[eigenVector, eigenValue] = eigs(L, nSubjects-1);
nEigs = size(eigenVector,2);
for i = 1 : nEigs
    ev = diag(eigenValue);
    ev = ev(i);
 	eigenVector(:, i) = eigenVector(:, i)/ norm(eigenVector(:, i));
    eigenVector_new(:, i) = (1/sqrt(nSubjects*ev))*imageTrain*eigenVector(:, i);
end;
eigenVector = eigenVector_new;
return;

