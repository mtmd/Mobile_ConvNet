function [ tmp ] = loader( fileName )
filename = strcat('C:\CodeBase\SqueezeNet_DSE\Android_Results\', fileName);
fileID = fopen(filename);
tmp = fread(fileID,'float', 'b');
fclose(fileID);
end

