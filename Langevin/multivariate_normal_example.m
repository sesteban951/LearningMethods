%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mulitivariate gaussian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Define mean and covariance
mu1 = [2;         
       2];         
sigma1 = [1.0 0.0;  
          0.0 0.5]; 
mu2 = [-2;         
        2]; 
sigma2 = [2.0 0.0;  
          0.0 1.0]; 
mu3 = [0;         
      -2]; 
sigma3 = [0.5 0.2;  
          0.2 1.0]; 
weights = [0.3; 0.4; 0.3]; % weights for the Gaussian components

% pack the means and covariances into arrays
n_dists = 3;
mu_list = zeros(2, 1, n_dists);
mu_list(:,:, 1) = mu1;
mu_list(:,:, 2) = mu2;
mu_list(:,:, 3) = mu3;

sigma_list = zeros(2, 2, n_dists);
sigma_list(:,:, 1) = sigma1;
sigma_list(:,:, 2) = sigma2;
sigma_list(:,:, 3) = sigma3;

% compute the mean of the means
mu_avg = zeros(2, 1);
sigma_avg = zeros(2, 2);
for i = 1:n_dists
    mu_avg = mu_avg + weights(i) * mu_list(:, :, i);
    sigma_avg = sigma_avg + weights(i) * sigma_list(:, :, i);
end

% Create a grid of (x, y) points
n_pts = 50;
x_range = linspace(mu_avg(1) - 5*sqrt(sigma_avg(1,1)), mu_avg(1) + 5*sqrt(sigma_avg(1,1)), n_pts);
y_range = linspace(mu_avg(2) - 5*sqrt(sigma_avg(2,2)), mu_avg(2) + 5*sqrt(sigma_avg(2,2)), n_pts);
[X, Y] = meshgrid(x_range, y_range);

% Pack into points
pos = [X(:), Y(:)];

% Compute the Gaussian PDF
Z_GMM = zeros(size(pos, 1), 1);
Z_Bayesian = zeros(size(pos, 1), 1);
for i = 1:size(pos, 1)
    
    % Compute the probability density for each point
    x = pos(i, :)';
    z_GMM = normal_pdf_GMM(x, mu_list, sigma_list, weights);
    z_Bayesian = normal_pdf_Bayesian(x, mu_list, sigma_list);

    % Reshape the result back to the grid
    Z_GMM(i) = z_GMM;
    Z_Bayesian(i) = z_Bayesian;
end

% Reshape Z to match the grid dimensions of X and Y
Z_GMM = reshape(Z_GMM, size(X));
Z_Bayesian = reshape(Z_Bayesian, size(X));

% Plot the surface
figure;

subplot(1,2,1);
surf(X, Y, Z_GMM, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); zlabel('Probability Density');
title('Gaussian Mixture Model');
colormap turbo;
colorbar;
view(45, 45); % 3D view angle

subplot(1,2,2);
surf(X, Y, Z_Bayesian, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); zlabel('Probability Density');
title('Bayesian Fusion of Gaussians');
colormap turbo;
colorbar;
view(45, 45); % 3D view angle

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% gaussian mixture model
function p = normal_pdf_GMM(x, mu_list, sigma_list, weights)
    
    % number of distributions
    n_dists = size(mu_list, 3);

    % average out the PDFs
    p = 0;
    for i = 1:n_dists

        mu = mu_list(:, :, i);
        sigma = sigma_list(:, :, i);

        % compute the inverse and det of the covariance matrix
        inv_sigma = inv(sigma);
        det_sigma = det(sigma);

        % compute the difference
        diff = x - mu;

        % compute the probability density
        d = length(x);
        p_i = (1 / ((2 * pi)^(d/2) * sqrt(det_sigma))) * exp(-0.5 * diff' * inv_sigma * diff) ;
        p = p + weights(i) * p_i;
    end
end

% Bayesian fusion of normal distributions
function p = normal_pdf_Bayesian(x, mu_list, sigma_list)

    % number of distributions
    n_dists = size(mu_list, 2); % Assuming mu_list is (dim, n_dists)
    dim = size(mu_list, 1);

    % Compute fused covariance
    Sigma_inv_sum = zeros(dim);
    mu_weighted_sum = zeros(dim, 1);

    % compute the equivalent Bayesian fusion
    for i = 1:n_dists
        Sigma_i = sigma_list(:, :, i);
        mu_i = mu_list(:, i); % Adjusted indexing for 2D mu_list
        Sigma_inv = inv(Sigma_i); % Consider replacing with a numerically stable method

        Sigma_inv_sum = Sigma_inv_sum + Sigma_inv;
        mu_weighted_sum = mu_weighted_sum + Sigma_inv * mu_i;
    end
    Sigma_fused = inv(Sigma_inv_sum); % Consider replacing with a numerically stable method
    mu_fused = Sigma_fused * mu_weighted_sum;

    % Compute the PDF of the fused Gaussian at x
    diff = x - mu_fused;
    d = length(x);
    norm_const = 1 / ((2*pi)^(d/2) * sqrt(det(Sigma_fused)));
    p = norm_const * exp(-0.5 * (diff' * (Sigma_inv_sum) * diff)); % Avoid explicit inversion
end
