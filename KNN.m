function predicted_labels = KNN(k,train_data,train_labels,test_data,distmethod)
%checks
if nargin < 4
    error('Too few input arguments.')
elseif nargin < 5
    distmethod = 'Euclidian';
% elseif nargin < 6 && distmethod == 'Minkowski'
%     p = 3;
end
if size(train_data,2) ~= size(test_data,2)
    error('data should have the same dimensionality');
end
%initialization
predicted_labels = zeros(size(test_data,1),1);
ed = zeros(size(test_data,1),size(train_data,1)); %ed: (MxN) euclidean distances 
sqed = zeros(size(test_data,1),size(train_data,1)); %sqed: (MxN) squared euclidian distances
man = zeros(size(test_data,1),size(train_data,1)); %man: (MxN) manhattan distances
mink = zeros(size(test_data,1),size(train_data,1)); %mink: (MxN) minkowski distances
mah = zeros(size(test_data,1),size(train_data,1)); %mah: (MxN) mahalanobis distances
ind = zeros(size(test_data,1),size(train_data,1)); %corresponding indices (MxN)
k_nn = zeros(size(test_data,1),k); %k-nearest neighbors for testing sample (Mxk)
switch distmethod
    case 'Euclidian'
        %calc euclidean distances between each testing data point and the training data samples
        for test_point = 1:size(test_data,1)
            for train_point = 1:size(train_data,1)
                %calc and store sorted euclidean distances with corresponding indices
                ed(test_point,train_point) = sqrt(sum((test_data(test_point,:) - train_data(train_point,:)).^2));
            end
            [ed(test_point,:),ind(test_point,:)] = sort(ed(test_point,:));
        end
    case 'SquareEuclidian'
        %calc square euclidean distances between each testing data point and the training data samples
        for test_point = 1:size(test_data,1)
            for train_point = 1:size(train_data,1)
                %calc and store sorted square euclidean distances with corresponding indices
                sqed(test_point,train_point) = sum((test_data(test_point,:) - train_data(train_point,:)).^2);
            end
            [sqed(test_point,:),ind(test_point,:)] = sort(sqed(test_point,:));
        end
    case 'Manhattan'
        %calc manhattan distances between each testing data point and the training data samples
        for test_point = 1:size(test_data,1)
            for train_point = 1:size(train_data,1)
                %calc and store sorted manhattan distances with corresponding indices
                man(test_point,train_point) = sum(abs(test_data(test_point,:) - train_data(train_point,:)));
            end
            [man(test_point,:),ind(test_point,:)] = sort(man(test_point,:));
        end
    case 'Minkowski'
        %calc minkowski distances between each testing data point and the training data samples
        p = 4;
        for test_point = 1:size(test_data,1)
            for train_point = 1:size(train_data,1)
                %calc and store sorted minkowski distances with corresponding indices
                mink(test_point,train_point) = nthroot(sum(abs((test_data(test_point,:) - train_data(train_point,:))).^p),p);
            end
        [mink(test_point,:),ind(test_point,:)] = sort(mink(test_point,:));
        end
    case 'Mahalanobis'
        %calc mahalanobis distances between each testing data point and the training data samples
        for test_point = 1:size(test_data,1)
            for train_point = 1:size(train_data,1)
                %calc and store sorted mahalanobis distances with corresponding indices
                mah(test_point,train_point) = sqrt(sum((test_data(test_point,:) - train_data(train_point,:))*(cov(test_data(test_point,:),train_data(train_point,:)))'*(test_data(test_point,:) - train_data(train_point,:))'));
            end
        [mah(test_point,:),ind(test_point,:)] = sort(mah(test_point,:));
        end
    otherwise
end

        
%find the nearest k for each data point of the testing data
k_nn = ind(:,1:k);
%get the majority vote 
for i = 1:size(k_nn,1)
    options = unique(train_labels(k_nn(i,:)'));
    max_count = 0;
    max_label = 0;
    for j = 1:length(options)
        L = length(find(train_labels(k_nn(i,:)') == options(j)));
        if L > max_count
            max_label = options(j);
            max_count = L;
        end
    end
    predicted_labels(i) = max_label;
end