% preprocess data for neural network (make into jpg)
% will just open up all zip files and extract images
filePathZips = '../experimental/364_TIF_FLIR/';

% call recursive function to open up files
% preprocess_recursive(filePathZips);

% separate files into different classes, crops, train and test
separateTrainTest(filePathZips)

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

%-----------------------------------------------------------------------------------------
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

%-----------------------------------------------------------------------------------------
function separateTrainTest(path_to_file)
    
    % get all folders
    exp_folders = getFolders(path_to_file);
    for s = 1:length(exp_folders)
        exp_folder = strcat(path_to_file,filesep,exp_folders{s});
        
        % mkdirs camera folders
        makeCameraFoldersAndAddFiles(exp_folder,'FLIR');
%         makeCameraFoldersAndAddFiles(exp_folder,'LWIR');
%         makeCameraFoldersAndAddFiles(exp_folder,'VIS');
    end

end

function folder_cell = getFolders(filepath)
    folders_st = dir(filepath);   % assume starting from current directory
    folders = {folders_st.name};
    folder_cell = folders([folders_st.isdir]);
    folder_cell((strcmp(folder_cell,'.') | ...
                 strcmp(folder_cell,'..') | ...
                 strcmp(folder_cell,'FLIR') | ...
                 strcmp(folder_cell,'VIS') | ...
                 strcmp(folder_cell,'LWIR') | ...
                 strcmp(folder_cell,'val') | ...
                 strcmp(folder_cell,'train'))) = [];
end

function makeCameraFoldersAndAddFiles(filepath,camera)
    cam_folder = strcat(filepath,filesep,camera);
    crop_folders = getFolders(filepath);
    for c = 1:length(crop_folders)
        crop_folder = strcat(filepath,filesep,crop_folders{c});
        files = dir(strcat(crop_folder,filesep,'*',camera,'*'));
        train_inds = 1:ceil(length(files)/2);
        train_folder = strcat(cam_folder,filesep,'train');
        if ~exist(train_folder,'dir')
            mkdir(train_folder);
        end
        if ~exist(strcat(train_folder,filesep,crop_folders{c}),'dir')
            mkdir(strcat(train_folder,filesep,crop_folders{c}));
        end
        for t = train_inds
            copyfile(strcat(crop_folder,filesep,files(t).name),strcat(strcat(train_folder,filesep,crop_folders{c},filesep,files(t).name)));
        end
        test_inds = ceil(length(files)/2)+1:length(files);
        val_folder = strcat(cam_folder,filesep,'val');
        if ~exist(val_folder,'dir')
            mkdir(val_folder);
        end
        if ~exist(strcat(val_folder,filesep,crop_folders{c}),'dir')
            mkdir(strcat(val_folder,filesep,crop_folders{c}));
        end
        for t = test_inds
            copyfile(strcat(crop_folder,filesep,files(t).name),strcat(strcat(val_folder,filesep,crop_folders{c},filesep,files(t).name)));
        end
    end

end