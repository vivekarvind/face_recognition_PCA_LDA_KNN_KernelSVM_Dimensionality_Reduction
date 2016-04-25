function [face_data, lables] = Vivek_Load_Data(subjects, subset, face_size)
    data = '/Users/VA/Documents/MS/Acads/Fall15/ML/PA3/data/';
    face_set = length(subset);
    face_data = zeros(prod(face_size), face_set*subjects);
    lables = zeros(1, face_set*subjects);
    k = 1;
    for i=1:subjects
        for j = 1:length(subset)
            img_name = sprintf('%ss%d/%d.pgm', data, i, subset(j));
            data_unit = read_img(img_name, face_size);
            data_unit = data_unit(:);
            data_unit = (data_unit - mean(data_unit))/std(data_unit);
            face_data(:, k) = data_unit;
            lables(k) = i;
            k = k + 1;
        end
    end

    function [img] = read_img(img_name, img_size)
        img = double(imread(img_name));
        img = imresize(img, img_size, 'bilinear');
    end
end
