function [BestX, Best_score, curve, BestSol] = ALSHADE_SPACMA_T2(MaxFEs, lb, ub, dim, Function_name)

    % 统一参数
    fhd = @cec17_func;
    L_Rate = 0.80;
    val_2_reach = 1e-8;
    RecordFEsFactor = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, ...
                       0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    progress = numel(RecordFEsFactor);

    % tanh 缩减参数
    k_param = 8;

    % 分阶段档案裁剪参数
    prune_params = struct();
    prune_params.k = 7;
    prune_params.beta_default = 2.0;
    prune_params.beta_min = 0.5;
    prune_params.beta_max = 2.5;
    prune_params.lambda_min = 0.30;
    prune_params.lambda_max = 0.70;

    % 边界处理
    if isscalar(lb)
        lb_vec = lb * ones(1, dim);
    else
        lb_vec = lb(:)';
    end

    if isscalar(ub)
        ub_vec = ub * ones(1, dim);
    else
        ub_vec = ub(:)';
    end

    lu = [lb_vec; ub_vec];
    optimum = Function_name * 100.0;

    % 单次运行核心
    [bsf_error_val, errorVals] = run_single_lshade_spacma( ...
        dim, MaxFEs, lu, fhd, Function_name, optimum, ...
        val_2_reach, L_Rate, progress, RecordFEsFactor, 1, ...
        k_param, prune_params);

    % 为了兼容 T2 主脚本接口，构造统一输出
    BestX = [];
    Best_score = bsf_error_val;
    curve = errorVals(:)';

    BestSol = struct();
    BestSol.Position = [];
    BestSol.Cost = bsf_error_val;
    BestSol.sol = struct('xx', [], 'yy', [], 'zz', []);
    BestSol.BestCost = curve;
    BestSol.MaxIt = MaxFEs;
end