clear; clc; close all;

tic
for i = 1:1000000
    x = 0.55 + i;
    x = x + x;
    x = x / 2;
    x = x * x;
    x = sqrt(x);
    x = log(x);
    x = exp(x);
    x = x / (x + 2);
end
T0_time = toc;

fprintf('T0 = %.8f seconds\n', T0_time);
save('T0_time.mat', 'T0_time');
writematrix(T0_time, 'T0_time.txt');