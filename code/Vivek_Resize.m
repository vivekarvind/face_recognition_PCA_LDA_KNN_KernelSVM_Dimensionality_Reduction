function [n_count,f_value] = Vivek_Resize(input_data)
    f_value = zeros(size(input_data,2)*input_data(1).Count,2576);
    count = 1 ;
    for i = 1:size(input_data,2)
        for j = 1:input_data(i).Count
            [f_value(count,:)] = reshape(imresize(read(input_data(i),j),0.5),1,2576);
            [n_count(count,1)] = input_data(i).ImageLocation(j);
            count = count + 1;
        end
    end
end