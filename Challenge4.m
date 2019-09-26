% load training set and testing set
clear all;
train_set = loadImages('train-images.idx3-ubyte')';
train_label = loadLabels('train-labels.idx1-ubyte');
test_set = loadImages('t10k-images.idx3-ubyte')';
test_label = loadLabels('t10k-labels.idx1-ubyte');

% cliassify the testing set 
train_size = size(train_set);
test_size = size(test_set);
tic;
predicted_label = KNN(3,train_set,train_label,test_set,'Euclidian');
t = toc;

% calculate accuracy
num_correct = sum(test_label == predicted_label);
accuracy = num_correct / test_size(1);
save -mat time.mat t
save -mat accuracy.mat accuracy