function [] = Task1_Vivek_PCA_KNN()
    % Loading Images
    data = imageSet('/Users/VA/Documents/MS/Acads/Fall15/ML/PA3/data/','recursive');

    % 5 Fold Cross Validation and Feature Extraction
    [part1, part2, part3, part4, part5] = partition(data,[0.2 0.2 0.2 0.2 0.2]);
    data_parts = {part1, part2, part3, part4, part5};
 
    % 5 Fold Cross Validation
    fprintf('\nKNN using 5 Fold Cross Validation');
    for i=1:5
        fprintf('\n\nCross Validation %d:\n', i);
        k=1;
        for j=1:5       
            if(j==i)
                [test_file, test_data] = Vivek_Extract_Feature(data_parts{j});
            else
                [train_file{k}, train_cell{k}] = Vivek_Extract_Feature(data_parts{j});
                k=k+1;
            end
        end
    
        train_data = [train_cell{1};train_cell{2};train_cell{3};train_cell{4}];
        trainfile = [train_file{1};train_file{2};train_file{3};train_file{4}];
    
        % Applying LDA on Feature Matrix
        train_data = Vivek_LDA(train_data);
        test_data = Vivek_LDA(test_data);
        
        fprintf('\nPREDICTIONS:\n')
        % Classification
        for test_id = 1: size(test_data,1)
            min_value = realmax;  
            f1 = test_data(test_id,:);
            test_path = char(test_file(test_id));
            for train_id = 1: size(train_data,1)
                f2 = train_data(train_id,:);
                Difference = f1 - f2;
                Dist_value = sqrt(Difference * Difference');
                if(Dist_value<min_value)
                    min_value = Dist_value;
                    id = train_id;
                end
            end
    
            train_path = char(trainfile(id));
            [train_path_str, class_name, extension1] = fileparts(train_path);
            train_class = strsplit(train_path_str,'\');        
            %classname = strsplit(classpath(0),'/');
            [test_path_str, test_file_name, extension2] = fileparts(test_path);
            test_class = strsplit(test_path_str,'\');
            fprintf('\n%d. File %s from class path %s belongs to %s',test_id,test_file_name, char(test_class(1)),char(train_class(1)));
        end
    end
end
 