clear;
% path where the MRI images are saved
path = input('Enter path of Directory ontaining Images\nFor example(D:\\Studies\\Sem 2\\Intelligent Systems\\Project\\Data): ','s');
%path = 'D:\Studies\Sem 2\Intelligent Systems\Project\D\';
sep = '//';
ext =  '.gif';
output_name = 'Alzheimer.csv';
output = strcat(path,sep,output_name);
fileID = fopen(output,'w');
index = 0;
fprintf(fileID,'%2s, %5s, %5s, %5s \n','#','Black','Gray', 'White');

% 415 images are present in our dataset
image_number = 415;
for p = 1:image_number

name = char(strcat(path,sep,string(p),ext));

A = imread(name);
A = im2double(A);

C = imadjust(A);

% BinaryImageis used to remove an noise if it is present
binaryImage = C > 0.0001;

%subplot(2,2,1);
%imshow(binaryImage, []);

resultImage = C;
resultImage(~binaryImage) = 0;
%subplot(2,2,2);
%imshow(A,[]);

% Use of K Means to cluster pixels with similar intensies
k = 3;
[idx, Mean] = kmeans(resultImage(:), k);
D = reshape(idx, size(C));

pixelintensity = zeros(k, 1);

for j = 1:k
    
%location of the pixel from the cluster
location = find(D == j, 1);
pixelintensity(j) = C(location);
end

ordering= zeros(k, 1);

% Ordering of Pixels
for j = 1:k
smallestIndex = find( pixelintensity == min(pixelintensity));
ordering(smallestIndex) = j;
pixelintensity(smallestIndex) = NaN;
end

D = ordering(D);

%subplot(2,2,4);
%imshow(D,[]);

% Number of Black, White and Gray Pixels
preblack = sum(resultImage(:)==0);
black = sum(D(:) == 1);

black = black - preblack;
white = sum(D(:) == 3);
gray = sum(D(:) == 2);

index = index + 1;

fprintf(fileID,'%3d, %5d, %5d, %5d\n',index, black, gray, white);

end

fclose(fileID);