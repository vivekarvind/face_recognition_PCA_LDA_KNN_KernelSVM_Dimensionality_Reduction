function [n,f] = Vivek_Extract_Feature(input_arg)
f = zeros(size(input_arg,2)*input_arg(1).Count,10304);
count = 1 ;
for i = 1:size(input_arg,2)
    for j = 1:input_arg(i).Count
        [f(count,:)] = reshape(read(input_arg(i),j),1,10304);
        [n(count,1)] = input_arg(i).ImageLocation(j);
        count = count + 1;
    end
end
end