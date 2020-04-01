% preprocess data for neural network (make into jpg)
% will just open up all zip files and extract images
filePathZips = '03282020/';

% call recursive function to open up files
preprocess_recursive(filePathZips);

function preprocess_recursive(filepath)
    
    if ~any(size(dir([filepath '*.zip' ]),1))
       % termination condition, convert tif files to jpg
       convertTifs(filepath);
    else
        % get files of directory
        zip_files = dir([filepath,'*.zip']);
        for ind = 1:length(zip_files)
            zip_file = zip_files(ind).name;
            fprintf('Zip file: %s\n',zip_file);
            new_filepath = strcat(filepath, zip_file(1:end-4),'/');
            
            % unzip file
            unzip(strcat(filepath, zip_file), new_filepath);
            
            % call recursion to reopen and run test
            preprocess_recursive(new_filepath)
        end
    end
    
end

function convertTifs(filepath)
    % obtain dir of tifs
    tif_dir = dir(strcat(filepath,'*.tif'));
    
    % iterate through tifs
    for ind = 1:length(tif_dir)
        % load in tiff
        tif_name = tif_dir(ind).name;
        tifnameNpath = strcat(filepath,tif_name);
        im = im2uint8(loadTiff(tifnameNpath));
        
        % write image data to jpg
        imwrite(im, strcat(tifnameNpath(1:end-3),'jpg'),'jpg');
        delete(tifnameNpath);
    end

end

%-----------------------------------------------------------------------------------------
function imData = loadTiff(path_to_file)
    im_info = imfinfo(path_to_file);
    TifLink = Tiff(path_to_file, 'r');
    num2read = length(im_info);
    imData = zeros(im_info(1).Height,im_info(1).Width,num2read,'like',TifLink.read());
    for i=1:num2read
       TifLink.setDirectory(i);
       imData(:,:,i)=TifLink.read();
    end
    TifLink.close()
end