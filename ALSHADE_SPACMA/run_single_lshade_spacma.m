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
    check_fes = unique(max(1, ceil(RecordFEsFactor(:) * max_nfes)));
    checkpoint_best = nan(progress,1);
    checkpoint_ptr = 1;
    
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
    
    % Initial evaluations: record only after actually spending the FE budget.
    while checkpoint_ptr <= progress && check_fes(checkpoint_ptr) <= nfes
        checkpoint_best(checkpoint_ptr) = bsf_fit_var;
        checkpoint_ptr = checkpoint_ptr + 1;
    end
    
    % 初始化记忆和存档
    memory_sf = 0.5 .* ones(memory_size, 1);
    memory_cr = 0.5 .* ones(memory_size, 1);
    memory_pos = 1;
    
    archive.NP = round(arc_rate * pop_size); % 确保是整数
    archive.pop = zeros(0, problem_size);
    archive.funvalues = zeros(0, 1);
    memory_1st_class_percentage = First_calss_percentage .* ones(memory_size, 1);
    
    % dCMA-lite channel state (final ALSHADE-SPACMA configuration)
    dcl.sigma = 0.5;
    dcl.var_ema_beta = 0.20;      % eta
    dcl.min_var = 1e-12;          % c_min
    dcl.bound_clamp = true;
    dcl.success_target = 0.20;    % p_target
    dcl.success_lr = 0.10;        % c_s
    dcl.success_state = dcl.success_target;  % p_s(0)

    % c_0: coordinate-wise sample variance of the initial population
    dcl.var_diag = var(popold, 0, 1)';
    if any(~isfinite(dcl.var_diag)) || all(dcl.var_diag <= 0)
        dcl.var_diag = ones(problem_size, 1);
    end
    dcl.var_diag = max(dcl.var_diag, dcl.min_var);

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
        
        % Second channel: dCMA-lite directly replaces full CMA-ES.
        dCMA_Channel_Index = ~Class_Select_Index;
        if any(dCMA_Channel_Index)
            notsel = find(dCMA_Channel_Index);
            sqrt_diag = sqrt(max(dcl.var_diag, dcl.min_var))';
            z = randn(numel(notsel), problem_size);
            step = z .* repmat(dcl.sigma .* sqrt_diag, numel(notsel), 1);
            vi(notsel, :) = pop(notsel, :) + step;

            if dcl.bound_clamp
                vi(notsel, :) = min(max(vi(notsel, :), ...
                    repmat(lu(1, :), numel(notsel), 1)), ...
                    repmat(lu(2, :), numel(notsel), 1));
            end
        end

        if ~isreal(vi) || any(~isfinite(vi(:)))
            % Numerical safeguard: use DE channel for this generation.
            Class_Select_Index = true(pop_size, 1);
            dCMA_Channel_Index = false(pop_size, 1);
            vi = pop + sf(:, ones(1, problem_size)) .* ...
                (pbest - pop + pop(r1, :) - popAll(r2, :));
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
        
        % Do not evaluate more individuals than the remaining FE budget.
        n_eval = min(pop_size, max_nfes - nfes);
        children_fitness = inf(pop_size, 1);
        if n_eval > 0
            fnew = feval(fhd, ui(1:n_eval, :)', func);
            children_fitness(1:n_eval) = fnew(:);
        end
        
        % Best-so-far trajectory is recorded at exact evaluation checkpoints.
        for i = 1:n_eval
            nfes = nfes + 1;
            if isfinite(children_fitness(i)) && children_fitness(i) < bsf_fit_var
                bsf_fit_var = children_fitness(i);
                bsf_solution = ui(i, :);
                bsf_index = i;
            end
            while checkpoint_ptr <= progress && nfes >= check_fes(checkpoint_ptr)
                checkpoint_best(checkpoint_ptr) = bsf_fit_var;
                checkpoint_ptr = checkpoint_ptr + 1;
            end
        end

        dif = abs(fitness - children_fitness);
        
        % 选择
        Child_is_better_index = (fitness > children_fitness);
        DE_Success_Index = Child_is_better_index & Class_Select_Index;
        dCMA_Success_Index = Child_is_better_index & dCMA_Channel_Index;
        goodCR = cr(DE_Success_Index);
        goodF = sf(DE_Success_Index);
        dif_val = dif(DE_Success_Index);
        dif_val_Class_1 = dif(DE_Success_Index);
        dif_val_Class_2 = dif(dCMA_Success_Index);
        
        archive.pop = [archive.pop; popold(Child_is_better_index == 1, :)];
        archive.funvalues = [archive.funvalues; fitness(Child_is_better_index == 1)];
        if ~isempty(archive.pop)
            [archive.pop, unique_idx] = unique(archive.pop, 'rows', 'stable');
            archive.funvalues = archive.funvalues(unique_idx);
        end
        
        % ==== NEW(1): 档案容量控制——分阶段 β(t), λ(t) 裁剪 ====
        if size(archive.pop, 1) > archive.NP
        
            t_now = nfes / max_nfes;
            archive = prune_archive_weighted_phase(archive, archive.NP, t_now, prune_params, lu);
        end
        % ======================================================
        
        [fitness, Child_is_better_index] = min([fitness, children_fitness], [], 2);
        popold = pop;
        popold(Child_is_better_index == 2, :) = ui(Child_is_better_index == 2, :);
        
        % DE F/CR memory and FCP channel-allocation update.
        has_channel_success = any(DE_Success_Index | dCMA_Success_Index);
        if has_channel_success
            if ~isempty(goodCR)
                sum_dif = sum(dif_val);
                if sum_dif <= 0, sum_dif = eps; end
                w_dif = dif_val / sum_dif;
                memory_sf(memory_pos) = (w_dif' * (goodF .^ 2)) / max(eps, (w_dif' * goodF));
                if max(goodCR) == 0 || memory_cr(memory_pos) == -1
                    memory_cr(memory_pos) = -1;
                else
                    memory_cr(memory_pos) = (w_dif' * (goodCR .^ 2)) / max(eps, (w_dif' * goodCR));
                end
            end

            if Hybridization_flag == 1
                s1 = sum(dif_val_Class_1);
                s2 = sum(dif_val_Class_2);
                denom = s1 + s2;
                if denom <= 0, denom = eps; end
                memory_1st_class_percentage(memory_pos) = ...
                    memory_1st_class_percentage(memory_pos) * L_Rate + ...
                    (1 - L_Rate) * (s1 / denom);
                memory_1st_class_percentage(memory_pos) = ...
                    min(max(memory_1st_class_percentage(memory_pos), 0.2), 0.8);
            end

            memory_pos = memory_pos + 1;
            if memory_pos > memory_size, memory_pos = 1; end
        end

        % dCMA-lite update before population reduction: ui/pop/logical masks
        % all still have the same pre-reduction population size.
        % dCMA-lite diagonal variance update from successful dCMA offspring.
        if any(dCMA_Success_Index)
            delta = ui(dCMA_Success_Index, :) - pop(dCMA_Success_Index, :);
            h = mean(abs(delta), 1)';
            h2 = max(h .^ 2, dcl.min_var);
            dcl.var_diag = (1 - dcl.var_ema_beta) * dcl.var_diag + ...
                           dcl.var_ema_beta * h2;
            dcl.var_diag = max(dcl.var_diag, dcl.min_var);
        end

        % Success-rate-controlled bidirectional step-size adaptation.
        % Ntrial=0: p_s and sigma unchanged.
        % Ntrial>0, Nsucc=0: r_g=0 and normal update is applied.
        dCMA_trials_g = sum(dCMA_Channel_Index);
        dCMA_success_g = sum(dCMA_Success_Index);
        if dCMA_trials_g > 0
            r_g = dCMA_success_g / dCMA_trials_g;
            dcl.success_state = (1 - dcl.success_lr) * dcl.success_state + ...
                                dcl.success_lr * r_g;
            dcl.sigma = dcl.sigma * exp(dcl.success_lr * ...
                        (dcl.success_state - dcl.success_target));
            dcl.sigma = max(dcl.sigma, sqrt(dcl.min_var));
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
    
    % Exact CEC2017 FE checkpoint errors.
    errorVals = abs(checkpoint_best - optimum);
    errorVals(errorVals < val_2_reach) = 0;
    assert(nfes == max_nfes, 'Incorrect FE budget: %d / %d', nfes, max_nfes);
end
