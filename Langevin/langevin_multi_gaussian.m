%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Langevin with Multiple Gaussians
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc;

% select the number of distributions
n_dists = 20;
mu_xlim = 2.5;
mu_ylim = 2.5;

% vector field settings
n_pts = 40;

% create containers for langevin dynamics
n_trajectories = 100;

% Langevin parameters
alpha = 0.005;          % time step
num_steps = 150;        % number of steps
max_step_length = 1.0;  % maximum step length (to prevent instability)
annealing = 1;          % annealing effect on or off, linear in num steps (for Langevin dynamics)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate random means and covariances
mu_list = zeros(2, 1, n_dists);
Sigma_list = zeros(2, 2, n_dists);
for i = 1:n_dists
    % sample a random mean
    mu_list(:, :, i) = sample_vector(-mu_xlim, mu_xlim, -mu_ylim, mu_ylim);
    
    % sample a random covariance matrix
    % Sigma_list(:, :, i) = sample_pos_def_matrix(2);
    Sigma_list(:, :, i) = 0.2 * eye(2);
end

% sample some weights
weights = sample_simplex(n_dists);
% weights = ones(n_dists, 1) / n_dists;

% compute the mean of the means
mu_avg = zeros(2, 1);
for i = 1:n_dists
    mu_avg = mu_avg + weights(i) * mu_list(:, :, i);
end

% create a grid of (x, y) points
xlim_max = mu_avg(1) + 3.0;
xlim_min = mu_avg(1) - 3.0;
ylim_max = mu_avg(2) + 3.0;
ylim_min = mu_avg(2) - 3.0;
x_range = linspace(xlim_min, xlim_max, n_pts);
y_range = linspace(ylim_min, ylim_max, n_pts);
[X, Y] = meshgrid(x_range, y_range);

% pack into points
grid_pts = [X(:), Y(:)];

% compute the Gaussian PDF
Z = zeros(size(X));
Z_grad = zeros(size(X, 1), size(X, 2), 2);

% evaluate PDF at each grid point
for i = 1:size(X, 1)
    for j = 1:size(X, 2)
        
        % evluate at this point
        x_vec = [X(i, j); Y(i, j)];
        
        % compute the probability density
        Z_point = normal_pdf_GMM(x_vec, mu_list, Sigma_list, weights);
        Z(i, j) = Z_point;

        % compute the gradient of the probability density
        Z_point_grad = normal_pdf_GMM_grad(x_vec, mu_list, Sigma_list, weights);
        Z_point_grad = Z_point_grad / norm(Z_point_grad); % normalize the gradient
        Z_grad(i, j, :) = Z_point_grad;
    end
end

% Reshape Z to match the grid dimensions of X and Y
Z = reshape(Z, size(X));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Langevin Dynamics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Langevin container
X_trajs = zeros(num_steps, 2, n_trajectories); % initialize Langevin trajectory

% populate the intial conditions
for i = 1:n_trajectories
    % sample a random point in the domain
    x0 = sample_vector(xlim_min, xlim_max, ylim_min, ylim_max);

    % insert into the Langevin container
    X_trajs(1, :, i) = x0';
end

% simulate the Langevin dynamics (realized with Euler-Maruyama)
for j = 1:n_trajectories
    for i = 2:num_steps
        % compute the gradient of the probability density at the current point
        xk = X_trajs(i-1, :, j)';
        grad = normal_pdf_GMM_grad(xk, mu_list, Sigma_list, weights);

        % annealing effect
        if annealing == 1
            % compute the annealing factor
            sigma = 1 - (i / num_steps);
            noise_vec = normrnd(0, sigma^2, 2, 1);
        else
            noise_vec = randn(2, 1);
        end

        % compute the step length and saturate if necessary
        step_direction = alpha * grad + sqrt(2 * alpha) * noise_vec;
        step_direction_length = norm(step_direction);
        if step_direction_length > max_step_length
            step_direction = (step_direction / step_direction_length) * max_step_length;
        end

        % take Langevin dynamics step
        xk = xk + step_direction;

        % save the new point
        X_trajs(i, :, j) = xk';
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot the surface
figure(1);
set(gcf, 'WindowState', 'maximized');

% plot the GMM
subplot(1, 2, 1);
hold on;
surf(X, Y, Z, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); zlabel('Probability Density');
title('Gaussian Mixture Model');
colormap turbo;
colorbar;
view(45, 45); % 3D view angle

% plot the gradient field
subplot(1, 2, 2);
hold on;

imagesc(x_range, y_range, Z);
xline(0); yline(0);
axis xy; % ensures y increases upwards
axis equal tight;
x_max = max(X_trajs(:, 1, :), [], 'all');
x_max = max(x_max, max(x_range));
x_min = min(X_trajs(:, 1, :), [], 'all');
x_min = min(x_min, min(x_range));
y_min = min(X_trajs(:, 2, :), [], 'all');
y_min = min(y_min, min(y_range));
y_max = max(X_trajs(:, 2, :), [], 'all');
y_max = max(y_max, max(y_range));
axis([x_min x_max y_min y_max]);

xlabel('x'); ylabel('y');
title('2D Heat Map of Multivariate Gaussian');
colorbar;

% plot the gradient field
quiver(X, Y, Z_grad(:, :, 1), Z_grad(:, :, 2), 'r');

% animate the trajecotories
pause(0.5);
for i = 1:num_steps
    % plot the Langevin trajectories
    pts = [];
    for j = 1:n_trajectories
        pt = plot(X_trajs(i, 1, j), X_trajs(i, 2, j), 'go', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
        pts = [pts; pt];
    end

    % set the title
    title(['Langevin Dynamics Trajectories (Step ' num2str(i) ' / ' num2str(num_steps) ')']);

    drawnow;

    if i ~= num_steps
        % delete the previous points
        delete(pts);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% PDF of the gaussian mixture model (GMM)
function p = normal_pdf_GMM(x, mu_list, sigma_list, weights)
    
    % number of distributions
    n_dists = size(mu_list, 3);

    % average out the PDFs
    p = 0;
    for i = 1:n_dists

        % get the mean and covariance
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

% log gradient of the GMM
function grad_p = normal_pdf_GMM_grad(x, mu_list, sigma_list, weights)

    % number of distributions
    n_dists = size(mu_list, 3);

    % dimensionality of x
    d = length(x);

    % initialize numerator (gradient of PDF)
    grad_pdf = zeros(d, 1);

    % compute total PDF value using helper function
    p = normal_pdf_GMM(x, mu_list, sigma_list, weights);

    % compute gradient of the PDF
    for i = 1:n_dists

        % get mean and covariance
        mu = mu_list(:, :, i);
        sigma = sigma_list(:, :, i);
        
        % inverse and determinant
        inv_sigma = inv(sigma);
        det_sigma = det(sigma);
        
        % compute the difference
        diff = x - mu;

        % component PDF
        coeff = 1 / ((2 * pi)^(d/2) * sqrt(det_sigma));
        exp_term = exp(-0.5 * diff' * inv_sigma * diff);
        p_i = coeff * exp_term;

        % gradient of component
        grad_p_i = p_i * (-inv_sigma * diff);

        % accumulate weighted gradients
        grad_pdf = grad_pdf + weights(i) * grad_p_i;
    end

    % final log-gradient (handle division by zero)
    if p > 0
        grad_p = grad_pdf / p;
    else
        grad_p = zeros(d, 1);
    end
end

% sample a random vector
function v = sample_vector(xmin, xmax, ymin, ymax)
    % random vector with uniform distribution
    v = [(rand(1) * (xmax - xmin)) + xmin; 
         (rand(1) * (ymax - ymin)) + ymin];
end

% sample a positive definite symmetric matrix
function M = sample_pos_def_matrix(n)
    % Generate a random matrix
    A = rand(n);
    % Create a symmetric matrix
    M = A' * A;
end

% sample a random point in the simplex
function x = sample_simplex(n)
    % Generate n+1 sorted uniform random numbers between 0 and 1
    u = sort([0, rand(1, n-1), 1]);
    % The differences between consecutive elements give a point in the simplex
    x = diff(u)';
end
