function [] = Task5_Vivek_KERNEL_SVM()

    training_data_no = 8;
    p = struct();
    accuracy = zeros(1, 5);
    perm = randperm(10);
    
    for i=1:5
        testing_data_no = 10 - training_data_no;
        test_id = testing_data_no*i-1:testing_data_no*i;
        train_id = setxor(1:10, test_id);
        p.training_data = perm(train_id);
        p.testing_data = perm(test_id);
        p.number_of_classes = 40;
        fprintf('Cross Validation %d:\n', i);
        training_set = p.training_data;
        testing_set = p.testing_data;
        classes = p.number_of_classes;
        tic;
        
        % Load the data
        [train_faces, train_labels] = Vivek_Load_Data(classes, training_set, [112 92]);
        [test_faces, test_labels] = Vivek_Load_Data(classes, testing_set, [112 92]);

        % Select the Kernel
        train_kernel_faces = (1+train_faces'*train_faces'').^2;
        test_kernel_faces = (1+test_faces'*train_faces'').^2;
        train_kernel_faces = train_kernel_faces';
        test_kernel_faces = test_kernel_faces';
        [dimensions, numbers] = size(train_kernel_faces);
        w_vector = zeros(dimensions, classes);
        b_term = zeros(1, classes);

        % Training (One-Vs-Rest)
        for class_id = 1:classes
            class_label = -ones(size(train_labels));
            class_label(logical(train_labels == class_id)) = 1;
            X_vector = train_kernel_faces;
            y_labels = class_label;
            C = classes;
            K_func = X_vector' * X_vector;
            [dim_1, ~] = size(X_vector);
            sample_number = size(K_func, 2);
            Y_vector = y_labels' * y_labels;  H = K_func .* Y_vector;
            positive_x = X_vector(:, y_labels == 1);  
            negative_x = X_vector(:, y_labels == -1);
            f = -ones(sample_number, 1)';    
            A=[]; 
            b_1=[]; 
            A_eq = y_labels; 
            b_eq = 0;
            lb_args = zeros(sample_number, 1); 
            ub_args = C / sample_number * ones(sample_number,1);
            options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off'); 
            alpha = quadprog(H, f, A, b_1, A_eq, b_eq, lb_args, ub_args, [], options)';
            w_3 = (alpha .* y_labels);  
            w_1 = repmat(w_3, dim_1,1); 
            w_2 = sum(X_vector .* w_1, 2);
            positive_margin = w_2' * positive_x;  
            negative_margin = w_2' * negative_x;
            bias_term = -(max(negative_margin) + min(positive_margin))/2;
            w_vector(:,class_id) = w_2;
            b_term(:,class_id) = bias_term;
        end

        %Testing
        function_value = w_vector' * test_kernel_faces + repmat(b_term,size(w_vector' * test_kernel_faces, 2), 1)';
        [~, predict_labels] = max(function_value);
        positives = (predict_labels == test_labels);
        acc_individual = sum(positives) / numel(test_labels);
        fprintf('Individual Accuracy: %.2f%%\n', 100 * acc_individual);
        toc;
        accuracy(i) = acc_individual;
    end
    fprintf('Overall Accuracy of Kernel SVM is: %.2f%%\n', 100 * mean(accuracy));
end

