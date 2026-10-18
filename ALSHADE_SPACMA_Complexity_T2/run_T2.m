clear; clc; close all;

%% ========================= 基本设置 =========================
Function_name = 18;
dims = [10, 30, 50, 100];
lb = -100;
ub = 100;
MaxFEs = 200000;         % 复杂度测试统一 200000 次函数评价
runs = 5;                % T2 重复 5 次
alg_name = 'ALSHADE_SPACMA';

result_dir = 'Complexity_Results_ALSHADE_SPACMA';
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

%% ========================= 读取 T0 =========================
T0_file = 'T0_time.mat';
if ~exist(T0_file, 'file')
    error('未找到 %s，请先运行 T0 脚本。', T0_file);
end

S0 = load(T0_file);
fn0 = fieldnames(S0);
T0 = S0.(fn0{1});

fprintf('Loaded T0 = %.8f seconds\n\n', T0);

%% ========================= 汇总矩阵 =========================
% [D, T1, T2_1, T2_2, T2_3, T2_4, T2_5, T2_hat, (T2_hat-T1)/T0]
summary_matrix = zeros(length(dims), 9);

for id = 1:length(dims)
    dim = dims(id);

    %% ---------- 读取该维度的 T1 ----------
    T1_file = sprintf('T1_CEC2017_D%d.mat', dim);
    if ~exist(T1_file, 'file')
        error('未找到 %s，请先运行对应维度的 T1 脚本。', T1_file);
    end

    S1 = load(T1_file);
    fn1 = fieldnames(S1);
    T1 = S1.(fn1{1});

    fprintf('============================================\n');
    fprintf('Algorithm = %s | Function = %d | Dimension = %d\n', alg_name, Function_name, dim);
    fprintf('Loaded T1 = %.8f seconds\n', T1);
    fprintf('============================================\n');

    %% ---------- 跑 5 次完整算法，得到 5 个 T2 ----------
    RunTime_All = zeros(runs, 1);

    for run = 1:runs
        rng(10000 * dim + run, 'twister');

        tic;
        [BestX, Best_score, curve, BestSol] = ...
            ALSHADE_SPACMA_T2(MaxFEs, lb, ub, dim, Function_name); %#ok<NASGU>
        RunTime_All(run) = toc;

        fprintf('T2 run %d = %.8f seconds\n', run, RunTime_All(run));
    end

    %% ---------- 计算 T2 均值和复杂度指标 ----------
    T2_hat = mean(RunTime_All);
    complexity_value = (T2_hat - T1) / T0;

    fprintf('Mean(T2) = %.8f seconds\n', T2_hat);
    fprintf('(T2_hat - T1) / T0 = %.8f\n\n', complexity_value);

    %% ---------- 保存单个维度结果 ----------
    save(fullfile(result_dir, sprintf('T2_ALSHADE_SPACMA_CEC2017_F18_D%d.mat', dim)), ...
        'RunTime_All', 'T2_hat', 'T1', 'T0', 'complexity_value', 'Function_name', 'dim', 'alg_name');

    writematrix(RunTime_All, ...
        fullfile(result_dir, sprintf('T2_ALSHADE_SPACMA_CEC2017_F18_D%d_runs.txt', dim)), ...
        'Delimiter', 'tab');

    summary_matrix(id, :) = [dim, T1, RunTime_All(:).', T2_hat, complexity_value];
end

%% ========================= 保存总汇总 =========================
summary_table = array2table(summary_matrix, ...
    'VariableNames', {'D', 'T1', 'T2_1', 'T2_2', 'T2_3', 'T2_4', 'T2_5', 'T2_hat', 'ComplexityValue'});

disp(summary_table);

save(fullfile(result_dir, 'Complexity_Summary_ALSHADE_SPACMA_CEC2017_F18.mat'), ...
    'summary_matrix', 'summary_table', 'T0', 'alg_name', 'Function_name');

writetable(summary_table, ...
    fullfile(result_dir, 'Complexity_Summary_ALSHADE_SPACMA_CEC2017_F18.csv'));

writematrix(summary_matrix, ...
    fullfile(result_dir, 'Complexity_Summary_ALSHADE_SPACMA_CEC2017_F18.txt'), ...
    'Delimiter', 'tab');

disp('All ALSHADE_SPACMA complexity experiments finished successfully.');