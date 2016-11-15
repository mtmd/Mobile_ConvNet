function check(layerName, intermedName, vectorize)
load(intermedName);
data = permute (data, [2 1 3]);
tmp = loader(strcat(layerName, '.bin'));
if (vectorize)
    [a, b, c] = size(data);
    counter = 1;
    data_1 = zeros(a, b, c);
    for mtmd = 1:4:c
        for jj = 1:a
            for kk = 1:b
                for ii = 0:3
                    data_1 (counter) = data(kk, jj, ii + mtmd);
                    counter = counter + 1;
                end
            end
        end
    end
    fprintf('Max error in %s %d\n', layerName, ...
    max(abs(tmp(:) - data_1(:))));
else
    fprintf('Max error in %s %d\n', layerName, ...
    max(abs(tmp(:) - data(:))));
end

end

