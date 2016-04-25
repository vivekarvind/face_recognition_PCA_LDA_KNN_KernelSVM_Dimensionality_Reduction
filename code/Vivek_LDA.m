function [train_data] = Vivek_LDA(train_data)
    data = '/Users/VA/Documents/MS/Acads/Fall15/ML/PA3/Dataset';
    image_size = [92,112];
    features = image_size(1)*image_size(2);
    directory = dir(data);
    file_name = directory(~ismember({directory.name},{'.','..'}));
    data_matrix = [];
    class_matrix = [];
    w = [];
    s_mat1 = zeros(features,features);
    s_mat2 = zeros(features,features);
    s_mat3 = zeros(features,features);
    s_mat4 = zeros(features,features);

    %Create a map
    classwise_vec = containers.Map('KeyType','int64', 'ValueType','any');
    classwise_vec_mean = containers.Map('KeyType','int64', 'ValueType','any');

    %Create Data Matrix
    for i=1:400
        img_path = strcat(data,'/',file_name(i).name);
        img_class = strsplit(file_name(i).name,'_');
        class_matrix = [class_matrix; img_class(1)];
        img_read = imread(img_path);
        image_vec = (img_read(:))';
        data_matrix = [data_matrix; image_vec];   
    end

    data_matrix_mean = mean(data_matrix);

    %Compute mean vectors
    for i = 1:40
        test_1 = strcat('s',num2str(i));
        classwise_features = data_matrix(find(strcmp(class_matrix,test_1)),:);
        classwise_vec(i) = classwise_features;
        classwise_vec_mean(i) = mean(classwise_features);   
    end

    %Compute intraclass scatter matrix
    for i = 1:40
        class_feature = classwise_vec(i);
        for j = 1:10
            image_feature = class_feature(j,:);
            image_feature = double(image_feature');
            s_mat5 = ((image_feature - classwise_vec_mean(i)') * (image_feature - classwise_vec_mean(i)')');
            s_mat1 = s_mat1 + s_mat5;
        end
        s_mat2 = s_mat2 + s_mat1;
    end

    %Compute interclass scatter matrix
    for i = 1:40
        class_mean = classwise_vec_mean(i)';
        s_1 = (10 * ((class_mean - data_matrix_mean') * (class_mean - data_matrix_mean')'));
        s_mat3 = s_mat3 + s_1;
    end
    s_mat4 = s_mat4 + s_mat3;


    %Compute Eigen Values
    eigen_inverse = pinv(s_mat2) * (s_mat4);
    [eigen_vector, eigen_values] = eig(eigen_inverse);
    [a1,b1] = size(eigen_values);
    eigen_len = a1*b1;

    %Sort Eigen Values with Eigen Vectors
    eigen_sorted=diag(sort(diag(eigen_values),'descend'));
    [c, index]=sort(diag(eigen_values),'descend'); 
    eigen_vec_sort=eigen_vector(:,index); 
    eigen_vals_sort=sort(real(diag(eigen_values)),'descend');

    %Form W Vector
    for i = 1:length(eigen_vals_sort)
        if(real(eigen_vals_sort(i)) > 0.2)
            w(:,i) = eigen_vec_sort(:,i);
        end
    end

    display(w)
end