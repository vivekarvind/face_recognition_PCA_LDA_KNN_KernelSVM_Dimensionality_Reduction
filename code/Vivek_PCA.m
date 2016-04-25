function reduc_mat = Vivek_PCA(input_data,k_size)

    % Subtract mean
    input_data=transpose(input_data);
    [M,N_val] = size(input_data);
    reduc_mat=zeros(N_val,k_size);
    mn_val = mean(input_data,2);
    input_data = input_data - repmat(mn_val,1,N_val);

    % Construct Covariance Matrix
    covar = 1 / (N_val-1) * (input_data) * (input_data');

    % Construct a diagonal matrix
    [PC_covariance, V_covariance] = eig(covar);
    V_covariance = diag(V_covariance);

    % Sort Variances
    [junk, rindices] = sort(-1*V_covariance);

    % Project the Data
    PC_covariance = PC_covariance(:,1:k_size);
    X = PC_covariance' * input_data;
    reduc_mat=transpose(X);
end