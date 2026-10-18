function archive = prune_archive_weighted_phase(archive, Amax, t, P, lu)
% 分阶段（β(t), λ(t)）+ 密度感知 的加权无放回档案裁剪
% - 输入：
%   archive.pop [n x d], archive.funvalues [n x 1]
%   lu    : [2 x d] 决策变量上下界；第1行为下界，第2行为上界
%   Amax  : 目标容量
%   t     : nfes/max_nfes ∈ [0,1]
%   P     : 参数结构体，字段：
%           k, beta_min, beta_max, lambda_min, lambda_max
% - 实现：
%   rank_norm = (rank-1)/(n-1)   （越大越差）
%   dens_norm = normalize( 1./(kNN_mean_dist) )
%   score = λ(t) * (rank_norm.^β(t)) + (1-λ(t)) * dens_norm
%   以 score 作为删除权重，做加权无放回采样删 (n-Amax) 个
%
% - 鲁棒性：
%   若权重全零或出现数值异常，退化为“先按 rank_norm，再按 dens_norm”的确定性 Top-K。

    % 采用优化问题的预定义搜索边界，禁止根据档案样本范围动态归一化。
    d = size(archive.pop, 2);
    if nargin < 5 || ~isnumeric(lu) || ~isequal(size(lu), [2, d]) || ...
            any(~isfinite(lu(:))) || any(lu(2,:) <= lu(1,:))
        error('ArchivePruning:InvalidBounds', ...
            '必须提供 2 x d 的有效上下界 lu，且每一维上界严格大于下界。');
    end
    n = size(archive.pop, 1);
    if n <= Amax, return; end
    K = n - Amax;

    % 动态 β(t), λ(t)
    beta_t = P.beta_min + (P.beta_max - P.beta_min) * min(max(t,0),1);
    lam_t  = P.lambda_min + (P.lambda_max - P.lambda_min) * min(max(t,0),1);

    % 1) 适应度排名（升序：越小越好） -> [0,1] 归一（越大越差）
    [~, idx] = sort(archive.funvalues, 'ascend');
    ranks = zeros(n,1); ranks(idx) = (1:n)';
    if n > 1
        rank_norm = (ranks - 1) / (n - 1);
    else
        rank_norm = 0;
    end

    % 2) 拥挤度：kNN 平均距离 -> 新密度 = 1/距离 -> [0,1] 归一（越大越拥挤/越该删）
    k = P.k;
    k = min(max(1,k), max(1,n-1));       % 安全 k
    % 将每维映射至 [0,1] 后，再计算档案个体之间的欧氏距离。
    lower = lu(1,:);
    span = lu(2,:) - lower;
    archive_pop_norm = bsxfun(@rdivide, bsxfun(@minus, archive.pop, lower), span);
    D = pdist2(archive_pop_norm, archive_pop_norm, 'euclidean');
    D(1:n+1:end) = inf;
    crowd = zeros(n,1);
    for i = 1:n
        [m, ~] = mink(D(i,:), k);
        crowd(i) = mean(m);
    end
    dens = 1 ./ (crowd + eps);
    if max(dens) > min(dens)
        dens_norm = (dens - min(dens)) / (max(dens) - min(dens) + eps);
    else
        dens_norm = zeros(n,1);
    end

    % 3) 分阶段混合 + 选择压力
    score = lam_t .* (rank_norm .^ beta_t) + (1 - lam_t) .* dens_norm;

    % 4) 加权无放回采样删除 K 个（鲁棒兜底）
    del = false(n,1);
    if any(score > 0) && all(score >= 0) && isfinite(sum(score))
        w = score(:);
        w(~isfinite(w) | w < 0) = 0;
        if sum(w) == 0
            % 兜底：确定性 Top-K（按 rank_norm，再按 dens_norm）
            del = fallback_topk_delete_(rank_norm, dens_norm, K);
        else
            w = w / sum(w);
            for tdel = 1:K
                wsum = sum(w);
                if wsum <= 0
                    % 再兜底
                    del = fallback_topk_delete_(rank_norm, dens_norm, K);
                    break;
                end
                w = w / wsum;
                pick = randsample(n, 1, true, w);
                while del(pick)
                    % 跳过已删
                    w(pick) = 0;
                    wsum2 = sum(w);
                    if wsum2 <= 0
                        del = fallback_topk_delete_(rank_norm, dens_norm, K);
                        break;
                    end
                    w = w / wsum2;
                    pick = randsample(n, 1, true, w);
                end
                del(pick) = true;
                w(pick) = 0; % 移除已删除元素
            end
        end
    else
        % 兜底：确定性 Top-K
        del = fallback_topk_delete_(rank_norm, dens_norm, K);
    end

    archive.pop(del, :) = [];
    archive.funvalues(del, :) = [];
end

function del = fallback_topk_delete_(rank_norm, dens_norm, K)
% 确定性兜底：先按 rank_norm（差者优先删），再按 dens_norm（拥挤者优先）
    n = numel(rank_norm);
    score_det = 0.7 * rank_norm + 0.3 * dens_norm;
    [~, ord] = sort(score_det, 'descend');
    ord = ord(1:min(K, n));
    del = false(n,1);
    del(ord) = true;
end