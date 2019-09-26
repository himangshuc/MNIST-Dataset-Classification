function labels = loadLabels(filename)
%loadLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

f = fopen(filename, 'rb');
assert(f ~= -1, ['Could not open ', filename, '']);

magic = fread(f, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(f, 1, 'int32', 0, 'ieee-be');

labels = fread(f, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(f);

end