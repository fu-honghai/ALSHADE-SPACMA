% ALSHADE-SPACMA (tanh + staged archive pruning + dCMA-lite)
% CEC2017; D = 10,30,50,100; 51 independent runs.
% For quick smoke test change dims/functions/runs to [10]/[1 3]/2.
clc;
format long g;
dims = [10 30 50 100];
functions = 1:30;
runs = 51;
base_seed = 20261019;
use_parallel = true;
L_Rate = 0.80;
val_2_reach = 1e-8;
RecordFEsFactor = [0.01 0.02 0.03 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
progress = numel(RecordFEsFactor);
k_param = 8;
prune_params = struct('k',7,'beta_min',0.5,'beta_max',2.5, ...
    'lambda_min',0.30,'lambda_max',0.70);
alg_name = 'ALSHADE_SPACMA';
output_dir = fullfile(pwd, 'CEC2017_Results_ALSHADE_SPACMA');
if ~exist(output_dir,'dir'), mkdir(output_dir); end
if exist('cec17_func','file') == 0
    error('cec17_func is missing. Use MATLAB Windows x64 or compile supplied CEC2017 MEX for your system.');
end
use_parallel = use_parallel && license('test','Distrib_Computing_Toolbox');
if use_parallel
    try
        if isempty(gcp('nocreate')), parpool('local'); end
    catch ME
        warning('Parallel pool unavailable, switching to serial: %s', ME.message);
        use_parallel = false;
    end
end
for D = dims
    max_nfes = 10000 * D;
    lu = [-100*ones(1,D); 100*ones(1,D)];
    all_Best_fitness = nan(30,runs);
    results_stat = nan(30,5); % [Best Worst Median Mean Std]
    for func = functions
        optimum = func*100;
        all_outcomes = nan(1,runs);
        all_errorvals = nan(progress,runs);
        fprintf('CEC2017: D=%d F=%d | MaxFEs=%d | runs=%d\n',D,func,max_nfes,runs);
        if use_parallel
            parfor run_id = 1:runs
                rng(base_seed + 100000*D + 1000*func + run_id,'twister');
                [one_error, one_curve] = run_single_lshade_spacma(D,max_nfes,lu, ...
                    @cec17_func,func,optimum,val_2_reach,L_Rate,progress, ...
                    RecordFEsFactor,run_id,k_param,prune_params);
                all_outcomes(run_id) = one_error;
                all_errorvals(:,run_id) = one_curve;
            end
        else
            for run_id = 1:runs
                rng(base_seed + 100000*D + 1000*func + run_id,'twister');
                [one_error, one_curve] = run_single_lshade_spacma(D,max_nfes,lu, ...
                    @cec17_func,func,optimum,val_2_reach,L_Rate,progress, ...
                    RecordFEsFactor,run_id,k_param,prune_params);
                all_outcomes(run_id) = one_error;
                all_errorvals(:,run_id) = one_curve;
            end
        end
        all_Best_fitness(func,:) = all_outcomes;
        results_stat(func,:) = [min(all_outcomes),max(all_outcomes), ...
            median(all_outcomes),mean(all_outcomes),std(all_outcomes)];
        save(fullfile(output_dir,sprintf('%s_CEC2017_Problem%d_D%d.mat', ...
            alg_name,func,D)),'all_outcomes','all_errorvals','max_nfes','RecordFEsFactor');
        writematrix(all_errorvals,fullfile(output_dir,sprintf('%s_%d_%d.txt',alg_name,func,D)), ...
            'Delimiter','tab');
    end
    writematrix(all_Best_fitness,fullfile(output_dir,sprintf('%s_%d.txt',alg_name,D)), ...
        'Delimiter','tab');
    writematrix(results_stat,fullfile(output_dir,sprintf('results_stat_%s_%d.txt',alg_name,D)), ...
        'Delimiter','tab');
    save(fullfile(output_dir,sprintf('%s_CEC2017_D%d_summary.mat',alg_name,D)), ...
        'all_Best_fitness','results_stat','runs','max_nfes');
end
fprintf('CEC2017 experiments finished. Results: %s\n', output_dir);
