clc;
close all;

% Define the path to the dataset folder
dataset_folder = 'C:\Users\PMYLS\Desktop\dataset';

% Get a list of all image files in the dataset folder
image_files = dir(fullfile(dataset_folder, '*.jpg'));

% Initialize variables to store evaluation metrics
TP_count = 0; % True Positives
FP_count = 0; % False Positives
TN_count = 0; % True Negatives
FN_count = 0; % False Negatives

% Loop over each image in the dataset
for i = 1:length(image_files)
    % Read the image
    filename = fullfile(dataset_folder, image_files(i).name);
    Im1 = imread(filename);
    
    % Convert image to grayscale if it's not already
    if size(Im1, 3) > 1
        Im1_gray = rgb2gray(Im1);
    else
        Im1_gray = Im1;
    end

    % Convert image to YCbCr color space
    Im_YCbCr = rgb2ycbcr(Im1);

    % Extract Y, Cb, and Cr channels
    Y = Im_YCbCr(:,:,1);
    Cb = Im_YCbCr(:,:,2);
    Cr = Im_YCbCr(:,:,3);

    % Thresholding to detect potential fire regions
    fire_mask = (Cr > 1.2 * Cb + 10) & (Y > 100) | (Im1_gray > 220); % Adjust the threshold for grayscale images

    % Apply median filtering to remove noise
    fire_mask = medfilt2(fire_mask, [5, 5]);

    % Thresholding to detect potential smoke regions
    smoke_mask = Y < 150 | (Im1_gray < 100); % Adjust the threshold for grayscale images

    % Apply median filtering to remove noise
    smoke_mask = medfilt2(smoke_mask, [5, 5]);

    % Refine fire mask using area thresholding
    fire_mask = bwareaopen(fire_mask, 100);

    % Refine smoke mask using area thresholding
    smoke_mask = bwareaopen(smoke_mask, 100);

    % Combine fire and smoke masks
    combined_mask = fire_mask | smoke_mask;

    % Find bounding box around combined mask
    props = regionprops(combined_mask, 'BoundingBox');
    bbox = props.BoundingBox;
    
    % Check if fire and smoke were detected
    fire_detected = ~isempty(regionprops(fire_mask)); % Check if any fire region was detected
    smoke_detected = any(smoke_mask(:)); % Check if any smoke was detected
    
    % Plotting red, green, blue channels, and chrominance components
    figure;
    subplot(2, 3, 1);
    imshow(Im1);
    title('Original Image');
    
    subplot(2, 3, 2);
    R = Im1; R(:, :, 2:3) = 0;
    imshow(R);
    title('Red Channel');
    
    subplot(2, 3, 3);
    G = Im1; G(:, :, [1 3]) = 0;
    imshow(G);
    title('Green Channel');
    
    subplot(2, 3, 4);
    B = Im1; B(:, :, 1:2) = 0;
    imshow(B);
    title('Blue Channel');
    
    subplot(2, 3, 5);
    imshow(Y);
    title('Y Component');
    
    subplot(2, 3, 6);
    imshow(Cr);
    title('Chrominance Red');

    % Display the binary masks for fire and smoke detection
    figure;
    subplot(3, 1, 1);
    imshow(fire_mask);
    title('Fire Detection (Binary)');
    
    subplot(3,1, 2);
    imshow(smoke_mask);
    title('Smoke Detection (Binary)');
    
    % Display the original image with fire and smoke detection status
    subplot(3, 1, 3);
    imshow(Im1);
    title('Final Outcome');
    hold on;
    if fire_detected
        text(10, 20, 'Fire Detected', 'Color', 'r', 'FontSize', 8, 'FontWeight', 'bold');
    else
        text(10, 20, 'No Fire Detected', 'Color', 'r', 'FontSize', 8, 'FontWeight', 'bold');
    end
    if smoke_detected
        text(10, 70, 'Smoke Detected', 'Color', 'r', 'FontSize', 8, 'FontWeight', 'bold');
    else
        text(10, 70, 'No Smoke Detected', 'Color', 'r', 'FontSize', 8, 'FontWeight', 'bold');
    end
    hold off;
    
    % Update counts based on ground truth and detection results
    % (In this example, ground truth is assumed to be known)
    % Replace the ground truth checks with actual ground truth extraction methods if available
    % Example: ground_truth_fire = is_fire_present(Im1);
    %          ground_truth_smoke = is_smoke_present(Im1);
    ground_truth_fire = true; % Example: ground_truth_fire = is_fire_present(Im1);
    ground_truth_smoke = false; % Example: ground_truth_smoke = is_smoke_present(Im1);
    
    if ground_truth_fire && fire_detected
        TP_count = TP_count + 1; % True Positive (fire detected when fire is present)
    elseif ~ground_truth_fire && fire_detected
        FP_count = FP_count + 1; % False Positive (fire detected when fire is not present)
    elseif ground_truth_fire && ~fire_detected
        FN_count = FN_count + 1; % False Negative (fire not detected when fire is present)
    else
        TN_count = TN_count + 1; % True Negative (no fire and no detection)
    end
    
    if ground_truth_smoke && smoke_detected
        TP_count = TP_count + 1; % True Positive (smoke detected when smoke is present)
    elseif ~ground_truth_smoke && smoke_detected
        FP_count = FP_count + 1; % False Positive (smoke detected when smoke is not present)
    elseif ground_truth_smoke && ~smoke_detected
        FN_count = FN_count + 1; % False Negative (smoke not detected when smoke is present)
    else
        TN_count = TN_count + 1; % True Negative (no smoke and no detection)
    end
end

% Calculate evaluation metrics
total_samples = length(image_files);
accuracy = (TP_count + TN_count) / total_samples;
precision = TP_count / (TP_count + FP_count);
recall = TP_count / (TP_count + FN_count);
f1_score = 2 * (precision * recall) / (precision + recall);

% Display the evaluation metrics
fprintf('Evaluation Metrics:\n');
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1_score);