%% 17th feb 2018

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VERSION 12 NOTES: need to change the LED values from 9 by 9 to 5 by 5,
% and size of the images from 200x200 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% size of images
tic 

pixels_x = 200;
pixels_y = 200;

sampling_pitch = 0.5; % um

% create a uniform grid with the above values
x_vector = ((1:pixels_x) - floor(pixels_x/2) - 1)* sampling_pitch;
y_vector = ((1:pixels_y) - floor(pixels_y/2) - 1)* sampling_pitch;

% calculation of the spatial location for each pixel, using the correct
% sampling interval like in a ccd camera.
[x,y] = meshgrid(x_vector, y_vector);

% difference in refractive index between bead and background
delta_n = 0.5;
% wavelength
lamda = 0.5;
% bead radius
bead_rad = 3;
% number of images
test_images = 10;
% number of training images
train_images = 30;

% absorption of the bead
absorption = 0.1;
% desired low pass filtered image size
des_imag_sz = 28;
% Generate random number of spheres to be used generated: should be bwtn 1
% and 4


% LED parameters
ledoffset = 10;
  
ledradx = 2;
ledrady = 2; 
totleds = (2*ledradx+1)*(2*ledrady+1);

test_image = zeros(test_images, totleds, des_imag_sz^2);
%training_image = zeros(training_images, totleds, des_imag_sz^2);
final_test_images = zeros(des_imag_sz,des_imag_sz, test_images*totleds,1);
% counter for the total number of images created 
mm=1;
test_labels = zeros(test_images,1);
%% Generate and check distances btwn 4 (4,2) sets of values
for kk = 1:test_images
    
    num_spheres = randi(5);
    % preallocations

    circle = zeros(pixels_x,pixels_y,num_spheres);
    bead_absorption = zeros(pixels_x,pixels_y,num_spheres);
    bead_thickness = zeros(pixels_x,pixels_y,num_spheres);
    bead_phase = zeros(pixels_x,pixels_y,num_spheres);
    bead = zeros(pixels_x,pixels_y,num_spheres);

    fbead =  zeros(pixels_x, pixels_y, num_spheres);
    
    tot = zeros();

    bead_location_x = zeros();
    bead_location_y = zeros(); 
    

    while bead_location_x(1,1) ==0 && bead_location_y(1,1)==0
   
        count = 1;
        diff_x = zeros();
        diff_y = zeros();
   
        % generate 4 rows and 2 columns of random numbers, and make sure they
        % fit in the meshgrid created above. Column 1 are my x-values, and
        % column 2 are the y-values
        c = (2* rand(num_spheres,2)- 1) * max(x_vector)*0.5;
   
        % make a for-loop to measure the difference between all the columns, row
        % by row:
        % column 1 = x1, x2, x3,x4
        % column 1 = y1, y2, y3,y4
        % The for-loop below does the following for the 4 values created above
        % diff_x = x1-x2, x1-x3, x1-x4, x2-x3, x2-x4, x3-x4
        % diff_y = y1-y2, y1-x3, y1-y4, y2-y3, y2-y4, y3-y4
        
        
        % remove bug when number of spheres generated is 1 
        if num_spheres ==1
            bead_location_x = c(:,1);
            bead_location_y = c(:,2);
        else
            for i = 1:size(c,1)-1
            a = c(i,1);
            b = c(i,2);
                for k = (i+1):length(c)
                    diff_x(count) = abs(abs(a) - abs(c(k,1)));
                    diff_y(count) = abs(abs(b) - abs(c(k,2)));
           
                    count = count+1;
                end
            end
        % combine the difference between the values calculated above into 1
        % matrix.
        
        tot = [diff_x; diff_y]';
   
        % check that the matrix created above, all the 6 differences created
        % above are larger than the radius of the sphere specified upfront
        if tot(:,:)>2*bead_rad
            bead_location_x = c(:,1);
            bead_location_y = c(:,2);
        else
            bead_location_x = 0;
            bead_location_y = 0;
        end
            
        end  
   
    end
    % using the non-overlapping distance above, create 4 spheres and make
    % the complex with a given phase
    final_bead = zeros(pixels_x, pixels_x,test_images);
    for ll = 1:length(bead_location_x)
       
        % create a circular binary mask for the location of the sphere
        % Use the equation of a circle as Dr.Roarke uses
        circle(:,:,ll) = sqrt((x-bead_location_x(ll,1)).^2 + (y-bead_location_y(ll,1)).^2)< bead_rad;
        % define the absorption/ amplitude of the bead. Am using Dr.Roarke's code
        % here again
        bead_absorption(:,:,ll) = 1 - absorption.*double(circle(:,:,ll));
        
        % define thickness of the bead along the optical axis. Am using Dr.Roarke's code
        % here again as i haven't derived the formula yet on my own
        % the bead thickness
        bead_thickness(:,:,ll) = real(2*sqrt(bead_rad^2-(x-bead_location_x(ll,1)).^2-(y-bead_location_y(ll,1)).^2));
        
        % the phase can now be defined as:
        bead_phase(:,:,ll) = exp(2*1i*pi*delta_n*2*bead_thickness(:,:,ll)/lamda);
         % complex bead is the
         
        bead(:,:,ll) = bead_absorption(:,:,ll).*bead_phase(:,:,ll);
        
    end
    
    % add the 4 sphere together so as to create 1 image with 2 spheres
    if num_spheres ==1
        final_bead(:,:,kk) = bead(:,:,ll);
    else 
        for p = 1:i+1
        final_bead(:,:,kk) = final_bead(:,:,kk) + bead(:,:,p);
        end
    end
    
    %imagesc(abs(final_bead(:,:,1))); axis image
    %final_bead(:,:,kk) = abs(final_bead(:,:,kk)).^2;
    
    % define some parameters for shifting the LED over the array
    tot_kspace_sz = size(final_bead, 1);
    centrekx = (tot_kspace_sz/2)+1;
    centreky = (tot_kspace_sz/2)+1;
    
    
    
    % perform fft of images
    
    fbead(:,:,kk) = fftshift(fft2(fftshift(final_bead(:,:,kk))));
    
    % define led array to be used in the simulation
    
    count = 1;
    for rr = -ledradx:ledradx
        for ss = -ledrady:ledrady
            
            % define led offset in fourier space
            offsetx = ledoffset.*rr;
            offsety = ledoffset.*ss;
            
            % define the value of fbead to be downsampled by taking
            % different portions of fbead above for the different images
            fbead1 = fbead(:,:,kk);
            % downsample to simulate image formation by a microscope
            fphase_mask_down = fbead1(centrekx + offsetx - floor(des_imag_sz/2):centrekx + offsetx + floor(des_imag_sz/2)-1, centreky + offsety - floor(des_imag_sz/2):centreky + offsety + floor(des_imag_sz/2)-1);
            
        
            % Inverse fourier transform to the image plane
            phase_mask_down = ifftshift(ifft2(ifftshift(fphase_mask_down)));
            
            % form image now
            image_i = abs(phase_mask_down).^2;
            %imagesc(image_i); axis image; pause(1);
            % save all the images created a stretched out column
       
            test_image (kk,count,:) = reshape(image_i, [des_imag_sz^2 1]);

            count = count+1;
            % 
        end
    end
%     for loop for creating labels

% Save the number of spheres created for per the main for loop
 test_labels(kk) = num_spheres;
 
 disp(kk)
 %%%%%%% above is the end of for loop for 1 image:
end
   
% re = reshape(test_image(1,3, :), [30, 30]);
% re1 = reshape(test_image(2,3, :), [30, 30]);
% figure; imagesc(re);figure; imagesc(re1); axis image
%% create the images (unstretched)
% Create a 3D matrix with the unwrapped images : x-axis = 30, y-axis = 30, 
% z-axis = total number of images created
for bb = 1:kk
    for cc = 1:count-1
        final_test_images(:,:,mm) = reshape(test_image(bb,cc, :), [des_imag_sz, des_imag_sz]);
        mm = mm+1;
    end
end        
% imagesc(final(:,:,25)); axis image
% imagesc(final(:,:,25)); axis image
% imagesc(final(:,:,26)); axis image

test_labels_2 = zeros(totleds,length(test_labels));
%%  create the associated labels (number of spheres per image) 
for aa = 1:length(test_labels)
   test_label_1 = test_labels(aa);
   for jjj = 1:mm-1
       test_labels_2(:,aa) = repmat(test_label_1, totleds, 1); 
   end
end
% store the labels finally as one long column
final_test_labels = test_labels_2(:)/totleds;
final_test_labels_changed = final_test_labels - 1;
% save to disk
save final_test_labels_changed.txt final_test_labels_changed -ascii
save final_test_labels.txt final_test_labels -ascii



%% Test that the labels created and the number of spheres in is same
imagesc(final_test_images(:,:,27)); axis image
spheres_test_number = final_test_labels(27)

%% Need to save the training images and labels as text files for classification 
% Increase the dynamic range of the uint8 data by setting the values above
% a certain threshold as the same constant
% 1: NORMALIZE THE DATA by thresholding each of the 81 images to same max
disp('norm division 1');
for bb = 1:size(test_image, 1)
   test_image(bb,:,:) = test_image(bb, :,:)./max(max(test_image(bb,:,:)));
end

% ADD GAUSSIAN NOISE (SIMULATING NOISE AT THE CCD SENSOR)
sensor_noise_fac = .0025;

test_image = test_image + sensor_noise_fac.*randn(size(test_image));

% Threshold negative values to 0
test_image(test_image < 0) = 0;

% set a maximum value for saturation
saturation_value = 0.7;
test_image(test_image > saturation_value) = saturation_value;

% scale the images now to uint8 scale: ranges from 0 to 255 = ((2^8)-1)
test_image = test_image./max(test_image(:)).*((2^8)-1);

%Use floor to prevent any -1 values
test_image = floor(test_image);

%%%%%%%% 
% reshape the images now to a 2D matrix, preserving the entries
test_image_2D = reshape(test_image, [test_images des_imag_sz^2*totleds]);

% write file to disk
fid = fopen('test_image_FP25_2k.txt', 'wt');
for ee = 1:test_images
    fprintf(fid, '%hd\t', squeeze(test_image_2D(ee, 1:des_imag_sz^2*totleds)));
    fprintf(fid, '\n');
end
fclose(fid);
toc

train_image = zeros(train_images, totleds, des_imag_sz^2);
%training_image = zeros(training_images, totleds, des_imag_sz^2);
final_train_images = zeros(des_imag_sz,des_imag_sz, train_images*totleds,1);
% counter for the total number of images created 
mm=1;
train_labels = zeros(train_images,1);
%% Generate and check distances btwn 4 (4,2) sets of values
for kk = 1:train_images
    
    num_spheres = randi(5);
    % preallocations

    circle = zeros(pixels_x,pixels_y,num_spheres);
    bead_absorption = zeros(pixels_x,pixels_y,num_spheres);
    bead_thickness = zeros(pixels_x,pixels_y,num_spheres);
    bead_phase = zeros(pixels_x,pixels_y,num_spheres);
    bead = zeros(pixels_x,pixels_y,num_spheres);

    fbead =  zeros(pixels_x, pixels_y, num_spheres);
    
    tot = zeros();

    bead_location_x = zeros();
    bead_location_y = zeros(); 
    

    while bead_location_x(1,1) ==0 && bead_location_y(1,1)==0
   
        count = 1;
        diff_x = zeros();
        diff_y = zeros();
   
        % generate 4 rows and 2 columns of random numbers, and make sure they
        % fit in the meshgrid created above. Column 1 are my x-values, and
        % column 2 are the y-values
        c = (2* rand(num_spheres,2)- 1) * max(x_vector)*0.8;
   
        % make a for-loop to measure the difference between all the columns, row
        % by row:
        % column 1 = x1, x2, x3,x4
        % column 1 = y1, y2, y3,y4
        % The for-loop below does the following for the 4 values created above
        % diff_x = x1-x2, x1-x3, x1-x4, x2-x3, x2-x4, x3-x4
        % diff_y = y1-y2, y1-x3, y1-y4, y2-y3, y2-y4, y3-y4
        
        
        % remove bug when number of spheres generated is 1 
        if num_spheres ==1
            bead_location_x = c(:,1);
            bead_location_y = c(:,2);
        else
            for i = 1:size(c,1)-1
            a = c(i,1);
            b = c(i,2);
                for k = (i+1):length(c)
                    diff_x(count) = abs(abs(a) - abs(c(k,1)));
                    diff_y(count) = abs(abs(b) - abs(c(k,2)));
           
                    count = count+1;
                end
            end
        % combine the difference between the values calculated above into 1
        % matrix.
        
        tot = [diff_x; diff_y]';
   
        % check that the matrix created above, all the 6 differences created
        % above are larger than the radius of the sphere specified upfront
        if tot(:,:)>2*bead_rad
            bead_location_x = c(:,1);
            bead_location_y = c(:,2);
        else
            bead_location_x = 0;
            bead_location_y = 0;
        end
            
        end  
   
    end
    % using the non-overlapping distance above, create 4 spheres and make
    % the complex with a given phase
    final_bead = zeros(pixels_x, pixels_x,train_images);
    for ll = 1:length(bead_location_x)
       
        % create a circular binary mask for the location of the sphere
        % Use the equation of a circle as Dr.Roarke uses
        circle(:,:,ll) = sqrt((x-bead_location_x(ll,1)).^2 + (y-bead_location_y(ll,1)).^2)< bead_rad;
        % define the absorption/ amplitude of the bead. Am using Dr.Roarke's code
        % here again
        bead_absorption(:,:,ll) = 1 - absorption.*double(circle(:,:,ll));
        
        % define thickness of the bead along the optical axis. Am using Dr.Roarke's code
        % here again as i haven't derived the formula yet on my own
        % the bead thickness
        bead_thickness(:,:,ll) = real(2*sqrt(bead_rad^2-(x-bead_location_x(ll,1)).^2-(y-bead_location_y(ll,1)).^2));
        
        % the phase can now be defined as:
        bead_phase(:,:,ll) = exp(2*1i*pi*delta_n*2*bead_thickness(:,:,ll)/lamda);
         % complex bead is the
         
        bead(:,:,ll) = bead_absorption(:,:,ll).*bead_phase(:,:,ll);
        
    end
    
    % add the 4 sphere together so as to create 1 image with 2 spheres
    if num_spheres ==1
        final_bead(:,:,kk) = bead(:,:,ll);
    else 
        for p = 1:i+1
        final_bead(:,:,kk) = final_bead(:,:,kk) + bead(:,:,p);
        end
    end
    
    %imagesc(abs(final_bead(:,:,1))); axis image
    %final_bead(:,:,kk) = abs(final_bead(:,:,kk)).^2;
    
    % define some parameters for shifting the LED over the array
    tot_kspace_sz = size(final_bead, 1);
    centrekx = (tot_kspace_sz/2)+1;
    centreky = (tot_kspace_sz/2)+1;
    
    
    
    % perform fft of images
    
    fbead(:,:,kk) = fftshift(fft2(fftshift(final_bead(:,:,kk))));
    
    % define led array to be used in the simulation
    
    count = 1;
    for rr = -ledradx:ledradx
        for ss = -ledrady:ledrady
            
            % define led offset in fourier space
            offsetx = ledoffset.*rr;
            offsety = ledoffset.*ss;
            
            % define the value of fbead to be downsampled by taking
            % different portions of fbead above for the different images
            fbead1 = fbead(:,:,kk);
            % downsample to simulate image formation by a microscope
            fphase_mask_down = fbead1(centrekx + offsetx - floor(des_imag_sz/2):centrekx + offsetx + floor(des_imag_sz/2)-1, centreky + offsety - floor(des_imag_sz/2):centreky + offsety + floor(des_imag_sz/2)-1);
            
        
            % Inverse fourier transform to the image plane
            phase_mask_down = ifftshift(ifft2(ifftshift(fphase_mask_down)));
            
            % form image now
            image_i = abs(phase_mask_down).^2;
            %imagesc(image_i); axis image; pause(1);
            % save all the images created a stretched out column
       
            train_image (kk,count,:) = reshape(image_i, [des_imag_sz^2 1]);

            count = count+1;
            % 
        end
    end
%     for loop for creating labels

% Save the number of spheres created for per the main for loop
 train_labels(kk) = num_spheres;
 
 disp(kk)
 %%%%%%% above is the end of for loop for 1 image:
end
   
% re = reshape(train_image(1,3, :), [30, 30]);
% re1 = reshape(train_image(2,3, :), [30, 30]);
% figure; imagesc(re);figure; imagesc(re1); axis image
%% create the images (unstretched)
% Create a 3D matrix with the unwrapped images : x-axis = 30, y-axis = 30, 
% z-axis = total number of images created
for bb = 1:kk
    for cc = 1:count-1
        final_train_images(:,:,mm) = reshape(train_image(bb,cc, :), [des_imag_sz, des_imag_sz]);
        mm = mm+1;
    end
end        
% imagesc(final(:,:,25)); axis image
% imagesc(final(:,:,25)); axis image
% imagesc(final(:,:,26)); axis image

train_labels_2 = zeros(totleds,length(train_labels));
%%  create the associated labels (number of spheres per image) 
for aa = 1:length(train_labels)
   train_label_1 = train_labels(aa);
   for jjj = 1:mm-1
       train_labels_2(:,aa) = repmat(train_label_1, totleds, 1); 
   end
end
% store the labels finally as one long column
final_train_labels = train_labels_2(:);
final_train_labels_changed = final_train_labels - 1;
% save to disk
save final_train_labels_changed.txt final_train_labels_changed -ascii
save final_train_labels.txt final_train_labels -ascii



%% train that the labels created and the number of spheres in is same
imagesc(final_train_images(:,:,27)); axis image
spheres_train_number = final_train_labels(27)

%% Need to save the training images and labels as text files for classification 
% Increase the dynamic range of the uint8 data by setting the values above
% a certain threshold as the same constant
% 1: NORMALIZE THE DATA by thresholding each of the 81 images to same max
disp('norm division 2');
for bb = 1:size(train_image, 1)
   train_image(bb,:,:) = train_image(bb, :,:)./max(max(train_image(bb,:,:)));
end

% ADD GAUSSIAN NOISE (SIMULATING NOISE AT THE CCD SENSOR)
sensor_noise_fac = .0025;

train_image = train_image + sensor_noise_fac.*randn(size(train_image));

% Threshold negative values to 0
train_image(train_image < 0) = 0;

% set a maximum value for saturation
saturation_value = 0.7;
train_image(train_image > saturation_value) = saturation_value;

% scale the images now to uint8 scale: ranges from 0 to 255 = ((2^8)-1)
train_image = train_image./max(train_image(:)).*((2^8)-1);

%Use floor to prevent any -1 values
train_image = floor(train_image);

%%%%%%%% 
% reshape the images now to a 2D matrix, preserving the entries
train_image_2D = reshape(train_image, [train_images des_imag_sz^2*totleds]);

% write file to disk
fid = fopen('train_image_FP25_10k.txt', 'wt');
for ee = 1:train_images
    fprintf(fid, '%hd\t', squeeze(train_image_2D(ee, 1:des_imag_sz^2*totleds)));
    fprintf(fid, '\n');
end
fclose(fid);
toc
