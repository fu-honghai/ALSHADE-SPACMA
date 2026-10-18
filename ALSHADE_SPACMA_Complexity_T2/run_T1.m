clear; clc; close all;

dims = [10, 30, 50, 100];
I_fno = 18;
pop_size = 200000;   % 200000 次评价
xmax = 100.0;
xmin = -100.0;

for dim = dims
    fprintf('Running T1 for Function 18 at D = %d ...\n', dim);

    popold = repmat(xmin .* ones(1, dim), pop_size, 1) + ...
             rand(pop_size, dim) .* ...
             (repmat(xmax .* ones(1, dim) - xmin .* ones(1, dim), pop_size, 1));

    tic;
    fitness = cec17_func(popold', I_fno); %#ok<NASGU>
    T1_time = toc;

    fprintf('T1(D=%d) = %.8f seconds\n', dim, T1_time);

    save(sprintf('T1_CEC2017_D%d.mat', dim), 'T1_time');
    writematrix(T1_time, sprintf('T1_CEC2017_D%d.txt', dim));
end

disp('All T1 experiments finished successfully.');