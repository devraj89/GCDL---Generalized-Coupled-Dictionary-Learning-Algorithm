function [Wx,Wy,r] = cluster_cca_mod(train_a,train_b,a_labels,b_labels,kapa_cca,knn,eta)

%% Get the center of each clusters in both the sets (MEAN CCA)
% disp('>Getting the center of each clusters....');
unq_a_label = unique(a_labels); %1x100
unq_b_label = unique(b_labels); %1x100

%% Calculating the cardinality of all the classes in both the sets
% disp('>Calculating the cardinality of all the classes in both the sets...');
card_a = zeros(1,size(unq_a_label,2)); %1x100
card_b = zeros(1,size(unq_b_label,2)); %1x100
for i=1:size(card_a,2)
    c = 0; d = 0;
    for j=1:size(a_labels,2)
        if unq_a_label(i)==a_labels(j)
            c = c + 1;
        end
    end
    for j=1:size(b_labels,2)
        if unq_b_label(i)==b_labels(j)
            d = d + 1;
        end
    end    
    card_a(1,i) = c;
    card_b(1,i) = d;
end

%% Calculate the value of the constant M
% disp('>calculating the value of M...');
M = 0;
for i=1:size(unq_a_label,2)
    M = M + card_a(1,i)*card_b(1,i);
end

%% Calculating the covariance matrix Cxy
% Reordering the matrix
train_a = train_a.';
train_b = train_b.';
% initialization the matrix
train_a_mean = zeros(length(unq_a_label),size(train_a,2));
train_b_mean = zeros(length(unq_b_label),size(train_b,2));
% calculating the means of the clusters
for i=1:length(unq_a_label)
    sum = 0; count = 0;
    [~,idx]=find(a_labels==unq_a_label(i));
    for j=1:length(idx)        
        sum = sum + train_a(idx(j),:);
        count = count + 1;
    end
    train_a_mean(i,:) = sum/count;
end
for i=1:length(unq_b_label)
    sum = 0; count = 0;
    [~,idx]=find(b_labels==unq_b_label(i));
    for j=1:length(idx)        
        sum = sum + train_b(idx(j),:);
        count = count + 1;
    end
    train_b_mean(i,:) = sum/count;
end

C = size(unq_a_label,2);
Cxy = 0;
for c=1:C %for each class
    mu_x = train_a_mean(c,:).';
    mu_y = train_b_mean(c,:).';
    Cxy = Cxy + card_a(1,c)*card_b(1,c)*mu_x*mu_y.';
end
Cxy = Cxy./M;
Cyx = Cxy.';

% Reordering the matrix
train_a = train_a.';
train_b = train_b.';

%% Calcualting the k-nearest neighbours for each class
score_mean_a = pdist2(train_a_mean,train_a_mean,'cosine');
score_mean_b = pdist2(train_b_mean,train_b_mean,'cosine');
for k=1:size(score_mean_a,1)
    finalScore = score_mean_a(k,:);
    [~,sortIndex] = sort(finalScore);
    knn_index_a(k,:) = sortIndex';
end
for k=1:size(score_mean_b,1)
    finalScore = score_mean_b(k,:);
    [~,sortIndex] = sort(finalScore);
    knn_index_b(k,:) = sortIndex';
end

train_a_mean = train_a_mean.';
train_b_mean = train_b_mean.';

%% Calculating the covariance matrix Cxx
% n = number of classes
% card = cardinality of the classes in each set
% disp('...calculating the covariance matrix Cxx....')
C = size(unq_a_label,2);
Cxx = 0;
for c=1:C %for each class
    %find those vectors having that label
    [~,idx]=find(a_labels==unq_a_label(c));
    
    zz = train_a_mean(:,knn_index_a(c,:));
    zz = zz(:,2:knn+1);
    sum2 = zz*zz.';

    sum = 0;
    for j=1:length(idx)
        x = train_a(:,idx(j));
        sum = sum + x*x.';
    end
    sum = card_b(1,c)*(sum + sum2 * card_a(1,c));
    Cxx = Cxx + sum;
end
Cxx = Cxx./M;
Cxx = Cxx + kapa_cca*eye(size(train_a,1));
 
%% Calculating the covariance matrix Cyy
% disp('...calculating the covariance matrix Cyy....')
C = size(unq_a_label,2);
Cyy = 0;
for c=1:C %for each class
    %find those vectors having that label
    [~,idx]=find(b_labels==unq_b_label(c));
    sum = 0;

    zz = train_b_mean(:,knn_index_b(c,:));
    zz = zz(:,2:knn+1);
    sum2 = zz*zz.';

    for j=1:length(idx)
        y = train_b(:,idx(j));
        sum = sum + y*y.';
    end
    sum = card_a(1,c)*(sum + sum2 * card_b(1,c));
    Cyy = Cyy + sum;
end
Cyy = Cyy./M + kapa_cca*eye(size(train_b,1));


% disp('...calculating the projection matrices....')
%% Calculating the Wx cca matrix
Rx = chol(Cxx);
inv_Rx = inv(Rx);
Z = inv_Rx'*Cxy*(Cyy\Cyx)*inv_Rx;
Z = 0.5*(Z' + Z);  % making sure that Z is a symmetric matrix
[Wx,r] = eig(Z);   % basis in h (X)
r = sqrt(real(r)); % as the original r we get is lamda^2
Wx = inv_Rx * Wx;   % actual Wx values

%% Calculating Wy
Wy = (Cyy\Cyx) * Wx; 

% by dividing it by lamda
Wy = Wy./repmat(diag(r)',size(train_b,1),1);