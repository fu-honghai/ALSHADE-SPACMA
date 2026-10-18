function [bsf_error_val, errorVals] = run_single_lshade_spacma(...
    problem_size, max_nfes, lu, fhd, func, optimum, ...
    val_2_reach, L_Rate, progress, RecordFEsFactor, run_id, ...
    k_param, prune_params)

    % 参数设置
    p_best_rate = 0.11;
    arc_rate = 1.4;
    memory_size = 5;
    pop_size = 18 * problem_size;
    max_pop_size = pop_size;
    min_pop_size = 4.0;
    First_calss_percentage = 0.5;

    % 初始化变量
    nfes = 0;
    run_funcvals = [];
    
    % 初始化种群
    popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, problem_size) .* ...
        (repmat(lu(2, :) - lu(1, :), pop_size, 1));
    pop = popold;
    
    % 初始适应度
    fitness = feval(fhd, pop', func);
    fitness = fitness';
    
    % 初始化最佳解
    bsf_fit_var = 1e+30;
    bsf_index = 0;
    bsf_solution = zeros(1, problem_size);
    
    % 评估初始种群
    for i = 1:pop_size
        nfes = nfes + 1;
        if (fitness(i) < bsf_fit_var && isreal(pop(i, :)) && ...
                sum(isnan(pop(i, :))) == 0 && min(pop(i, :)) >= -100 && max(pop(i, :)) <= 100)
            bsf_fit_var = fitness(i);
            bsf_solution = pop(i, :);
            bsf_index = i;
        end
        if nfes > max_nfes, break; end
    end
    
    run_funcvals = [run_funcvals; ones(pop_size, 1) * bsf_fit_var];
    
    % 初始化记忆和存档
    memory_sf = 0.5 .* ones(memory_size, 1);
    memory_cr = 0.5 .* ones(memory_size, 1);
    memory_pos = 1;
    
    archive.NP = round(arc_rate * pop_size); % 确保是整数
    archive.pop = zeros(0, problem_size);
    archive.funvalues = zeros(0, 1);
    memory_1st_class_percentage = First_calss_percentage .* ones(memory_size, 1);
    
    % 初始化CMA-ES参数
    sigma = 0.5;
    xmean = rand(problem_size, 1);
    mu = pop_size / 2;
    weights = log(mu + 1/2) - log(1:mu)';
    mu = floor(mu);
    weights = weights / sum(weights);
    mueff = sum(weights)^2 / sum(weights.^2);
    
    cc = (4 + mueff/problem_size) / (problem_size + 4 + 2*mueff/problem_size);
    cs = (mueff + 2) / (problem_size + mueff + 5);
    c1 = 2 / ((problem_size + 1.3)^2 + mueff);
    cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((problem_size + 2)^2 + mueff));
    damps = 1 + 2 * max(0, sqrt((mueff - 1)/(problem_size + 1)) - 1) + cs;
    
    pc = zeros(problem_size, 1);
    ps = zeros(problem_size, 1);
    B = eye(problem_size, problem_size);
    Dv = ones(problem_size, 1);
    C = B * diag(Dv.^2) * B';
    invsqrtC = B * diag(Dv.^-1) * B';
    eigeneval = 0;
    chiN = problem_size^0.5 * (1 - 1/(4*problem_size) + 1/(21*problem_size^2));
    
    % 主循环
    Hybridization_flag = 1;
    
    while nfes < max_nfes
        pop = popold;
        [temp_fit, sorted_index] = sort(fitness, 'ascend');
        
        mem_rand_index = ceil(memory_size * rand(pop_size, 1));
        mu_sf = memory_sf(mem_rand_index);
        mu_cr = memory_cr(mem_rand_index);
        mem_rand_ratio = rand(pop_size, 1);
        
        % 生成交叉率
        cr = normrnd(mu_cr, 0.1);
        term_pos = find(mu_cr == -1);
        cr(term_pos) = 0;
        cr = min(cr, 1);
        cr = max(cr, 0);

        % 生成缩放因子（前半固定，后半柯西）
        if nfes <= max_nfes/2
            sf = 0.45 + 0.1 * rand(pop_size, 1);
            pos = find(sf <= 0);
            while ~isempty(pos)
                sf(pos) = 0.45 + 0.1 * rand(length(pos), 1);
                pos = find(sf <= 0);
            end
        else
            sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
            pos = find(sf <= 0);
            while ~isempty(pos)
                sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                pos = find(sf <= 0);
            end
        end
        sf = min(sf, 1);
        
        % 生成混合类选择概率
        Class_Select_Index = (memory_1st_class_percentage(mem_rand_index) >= mem_rand_ratio);
        if Hybridization_flag == 0
            Class_Select_Index = or(Class_Select_Index, ~Class_Select_Index);
        end
        
        r0 = 1:pop_size;
        popAll = [pop; archive.pop];
        [r1, r2] = gnR1R2(pop_size, size(popAll, 1), r0);
        r1 = r1(:); r2 = r2(:); % 安全列向量
        
        pNP = max(round(p_best_rate * pop_size), 2);
        randindex = ceil(rand(1, pop_size) .* pNP);
        randindex = max(1, randindex);
        pbest = pop(sorted_index(randindex), :);
        
        vi = [];
        temp = [];
        if sum(Class_Select_Index) ~= 0
            vi(Class_Select_Index, :) = pop(Class_Select_Index, :) + ...
                sf(Class_Select_Index, ones(1, problem_size)) .* ...
                (pbest(Class_Select_Index, :) - pop(Class_Select_Index, :) + ...
                pop(r1(Class_Select_Index), :) - popAll(r2(Class_Select_Index), :));
        end
        
        if sum(~Class_Select_Index) ~= 0
            for k = 1:sum(~Class_Select_Index)
                temp(:, k) = xmean + sigma * B * (Dv .* randn(problem_size, 1));
            end
            vi(~Class_Select_Index, :) = temp';
        end
        
        if ~isreal(vi)
            Hybridization_flag = 0;
            continue;
        end
        
        vi = boundConstraint(vi, pop, lu);
        
        % 交叉
        mask = rand(pop_size, problem_size) > cr(:, ones(1, problem_size));
        rows = (1:pop_size)';
        cols = floor(rand(pop_size, 1) * problem_size) + 1;
        jrand = sub2ind([pop_size, problem_size], rows, cols);
        mask(jrand) = false;
        ui = vi;
        ui(mask) = pop(mask);
        
        children_fitness = feval(fhd, ui', func);
        children_fitness = children_fitness';
        
        % 更新最优解
        for i = 1:pop_size
            nfes = nfes + 1;
            if (children_fitness(i) < bsf_fit_var && isreal(ui(i, :)) && ...
                    sum(isnan(ui(i, :))) == 0 && min(ui(i, :)) >= -100 && max(ui(i, :)) <= 100)
                bsf_fit_var = children_fitness(i);
                bsf_solution = ui(i, :);
                bsf_index = i;
            end
            if nfes > max_nfes, break; end
        end
        
        run_funcvals = [run_funcvals; ones(pop_size, 1) * bsf_fit_var];
        
        dif = abs(fitness - children_fitness);
        
        % 选择
        Child_is_better_index = (fitness > children_fitness);
        goodCR = cr(Child_is_better_index == 1);
        goodF =  sf(Child_is_better_index == 1);
        dif_val = dif(Child_is_better_index == 1);
        dif_val_Class_1 = dif(and(Child_is_better_index,  Class_Select_Index) == 1);
        dif_val_Class_2 = dif(and(Child_is_better_index, ~Class_Select_Index) == 1);
        
        archive = updateArchive(archive, popold(Child_is_better_index == 1, :), ...
                                fitness(Child_is_better_index == 1));
        
        % ==== NEW(1): 档案容量控制——分阶段 β(t), λ(t) 裁剪 ====
        if size(archive.pop, 1) > archive.NP
            t_now = nfes / max_nfes;
            archive = prune_archive_weighted_phase(archive, archive.NP, t_now, prune_params, lu);
        end
        % ======================================================
        
        [fitness, Child_is_better_index] = min([fitness, children_fitness], [], 2);
        popold = pop;
        popold(Child_is_better_index == 2, :) = ui(Child_is_better_index == 2, :);
        
        % 记忆更新
        num_success_params = numel(goodCR);
        if num_success_params > 0
            sum_dif = sum(dif_val);
            dif_val = dif_val / sum_dif;
            memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
            if max(goodCR) == 0 || memory_cr(memory_pos) == -1
                memory_cr(memory_pos) = -1;
            else
                memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
            end
            if Hybridization_flag == 1
                memory_1st_class_percentage(memory_pos) = ...
                    memory_1st_class_percentage(memory_pos) * L_Rate + ...
                    (1 - L_Rate) * (sum(dif_val_Class_1) / ...
                    (sum(dif_val_Class_1) + sum(dif_val_Class_2)));
                memory_1st_class_percentage(memory_pos) = ...
                    min(memory_1st_class_percentage(memory_pos), 0.8);
                memory_1st_class_percentage(memory_pos) = ...
                    max(memory_1st_class_percentage(memory_pos), 0.2);
            end
            memory_pos = memory_pos + 1;
            if memory_pos > memory_size, memory_pos = 1; end
        end
        
        % 非线性种群缩减（tanh）
        t = nfes / max_nfes;
        weight = 0.5 * (1 + tanh(k_param * (t - 0.5)));
        plan_pop_size = round(max_pop_size - (max_pop_size - min_pop_size) * weight);
        plan_pop_size = max(min(plan_pop_size, max_pop_size), min_pop_size);
        
        if pop_size > plan_pop_size
            reduction_ind_num = pop_size - plan_pop_size;
            if pop_size - reduction_ind_num < min_pop_size
                reduction_ind_num = pop_size - min_pop_size;
            end
            
            pop_size = pop_size - reduction_ind_num;
            for r = 1:reduction_ind_num
                [~, indBest] = sort(fitness, 'ascend');
                worst_ind = indBest(end);
                popold(worst_ind, :) = [];
                pop(worst_ind, :) = [];
                fitness(worst_ind, :) = [];
                Child_is_better_index(worst_ind, :) = [];
            end
            
            archive.NP = round(arc_rate * pop_size);
            
            % ==== NEW(2): 种群缩减后再次做分阶段裁剪 ====
            if size(archive.pop, 1) > archive.NP
                t_now2 = nfes / max_nfes;
                archive = prune_archive_weighted_phase(archive, archive.NP, t_now2, prune_params, lu);
            end
            % =================================================
            
            % 更新CMA参数
            mu = pop_size / 2;
            weights = log(mu + 1/2) - log(1:mu)';
            mu = floor(mu);
            weights = weights / sum(weights);
            mueff = sum(weights)^2 / sum(weights.^2);
        end
        
        % CMA-ES自适应
        if Hybridization_flag == 1
            [~, popindex] = sort(fitness);
            xold = xmean;
            xmean = popold(popindex(1:mu), :)' * weights;
            
            % 路径更新
            ps = (1 - cs) * ps + ...
                sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma;
            hsig = sum(ps.^2) / (1 - (1 - cs)^(2 * nfes / pop_size)) / problem_size < 2 + 4/(problem_size + 1);
            pc = (1 - cc) * pc + ...
                hsig * sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma;
            
            % 协方差矩阵更新
            artmp = (1/sigma) * (popold(popindex(1:mu), :)' - repmat(xold, 1, mu));
            C = (1 - c1 - cmu) * C + ...
                c1 * (pc * pc' + (1 - hsig) * cc * (2 - cc) * C) + ...
                cmu * artmp * diag(weights) * artmp';
            
            % 步长自适应
            sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1));
            
            % B 和 D 的更新
            if nfes - eigeneval > pop_size / (c1 + cmu) / problem_size / 10
                eigeneval = nfes;
                C = triu(C) + triu(C, 1)';
                if any(isnan(C(:))) || any(~isfinite(C(:))) || ~isreal(C)
                    Hybridization_flag = 0;
                    continue;
                end
                [B, Dv] = eig(C);
                Dv = sqrt(diag(Dv));
                invsqrtC = B * diag(Dv.^-1) * B';
            end
        end
    end
    
    % 违反约束检查与输出
    if max(bsf_solution) > 100, fprintf('Run %d: Above Max\n', run_id); end
    if min(bsf_solution) < -100, fprintf('Run %d: Below Min\n', run_id); end
    if ~isreal(bsf_solution),    fprintf('Run %d: Complex\n', run_id);   end
    
    bsf_error_val = abs(bsf_fit_var - optimum);
    if bsf_error_val < val_2_reach, bsf_error_val = 0; end
    if any(isnan(bsf_solution)), fprintf('Run %d: NaN\n', run_id); end
    fprintf('Run %d: best-so-far error value = %1.8e\n', run_id, bsf_error_val);
    
    % 进度点误差
    errorVals = zeros(progress, 1);
    for w = 1:progress
        idx = min(RecordFEsFactor(w) * max_nfes, length(run_funcvals));
        bestold = run_funcvals(idx) - optimum;
        if abs(bestold) > 1e-8
            errorVals(w) = abs(bestold);
        else
            errorVals(w) = 0;
        end
    end
end