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
