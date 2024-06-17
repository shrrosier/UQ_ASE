% this script takes as input 3 matrices: Yv, Ys and X, where 
% - Yv (num_experiments x num_nodes) is the change in ice speed between the
% start and end of each simulation, at each node in the global mesh
% - Ys (num_experiments x num_nodes) is the change in ice surface elevation
% between the start and end of each simulation, at each node in the global 
% mesh
% - X (num_experiments x num_parameters) is the set of model parameters
% (randomly sampled) that were used to run each ice sheet model experiment

clearvars;

load('model_results.mat','Yv','Ys','X');

num_exp =size(Ys,1);

seq = randperm(num_exp);

% I do SVD for the k-largest singular values, where k captures at least XX%
% of the total variation in Yv and Ys (pct*100). This is repeated for 
% different thresholds of pct and pct becomes a part of the RNN network
% optimization
pct = [0.8 0.85 0.9 0.92 0.95]; 

% do the full svd to calculate how many components are required for each
% increment in pct
[~,Ss1,~] = svd(Ys); 
[~,Sv1,~] = svd(Yv);

for ii = 1:numel(pct)

    val = pct(ii);
    sec_nComp = find((cumsum(diag(Ss1).^2)./sum(diag(Ss1).^2))>val,1,'first');

    sec_pct1 = diag(Ss1).^2./sum(diag(Ss1).^2);
    sec_pct = sec_pct1(1:sec_nComp);

    vel_nComp = find((cumsum(diag(Sv1).^2)./sum(diag(Sv1).^2))>val,1,'first');

    vel_pct1 = diag(Sv1).^2./sum(diag(Sv1).^2);
    vel_pct = vel_pct1(1:vel_nComp);

    predictors = X;

    [Us,Ss,Vs] = svds(Ys,sec_nComp);

    [~,Sv,Vv] = svds(Yv,vel_nComp);

    val_idx = floor(num_exp*0.8);
    test_idx = floor(num_exp*0.1) + val_idx;

    Bs = Ss*Vs';
    sec_reproj = (Bs*Bs')\Bs; % equivalent to inv(B*B')*B
    sec_hat = Ys*sec_reproj';

    Bv = Sv*Vv';
    vel_reproj = (Bv*Bv')\Bv;
    vel_hat = Yv*vel_reproj';

    data = [sec_hat(seq,:) vel_hat(seq,:)];
    predictors2 = predictors(seq,:);

    T_train = data(1:val_idx,:);
    T_val = data(val_idx+1:test_idx,:);
    T_test = data(test_idx+1:end,:);

    X_train = predictors2(1:val_idx,:);
    X_val = predictors2(val_idx+1:test_idx,:);
    X_test = predictors2(test_idx+1:end,:);

    fname1 = sprintf('C:/Users/shrro/PycharmProjects/RNN_surrogate/mat_files/data_N0k%.2g',val*100);
    fname2 = sprintf('C:/Users/shrro/PycharmProjects/RNN_surrogate/mat_files/SVD_N0k%.2g',val*100);

    save(fname1,'X_test','X_val','X_train','T_train','T_val','T_test');

    save(fname2, 'Vv', 'Sv', 'Bv', 'vel_reproj', 'Vs', 'Ss', 'Bs', 'sec_reproj', 'vel_pct', 'sec_pct','seq');

end

