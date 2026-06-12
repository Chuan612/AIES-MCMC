% rng(2026)
clear; clc; close all
load data3.mat;

% -----------------------------
% Synthetic impedance benchmark
% -----------------------------
nd = 99;
nm = 99;
true_imp = Vp' .* Rho';

nfilt = 3;
cutofffr = 0.35;
[b, a_filter] = butter(nfilt, cutofffr);
true_imp = filtfilt(b, a_filter, true_imp);

dt = 0.001;
freq = 45;
ntw = 64;
[w, ~] = RickerWavelet(freq, dt, ntw);
w = w(:)';

r_true = diff(true_imp) ./ (true_imp(1:end-1) + true_imp(2:end));
d = conv(r_true, w, 'same');

sigma = sqrt(0.1 * var(d));
d_obs = d + sigma * randn(size(d));
% sigma = 0.1*var(Seis);  % var(noise) = var(data)/SNR
% Seis = Seis+ sqrt(sigma)*randn(size(Seis));
% Errvar = diag(sigma);
% sigmaerr = kron(Errvar, eye(nd));
% InvCovErr = pinv(sigmaerr);

% Prior model
cutofffr = 0.04;
[b, a_filter] = butter(nfilt, cutofffr);
m0 = filtfilt(b, a_filter, true_imp);
% m0 = m0-2.5;

% Spatial correlation matrix
corrlength = 3* dt;
trow = repmat(0:dt:(nd - 1) * dt, nd, 1);
tcol = repmat((0:dt:(nd - 1) * dt)', 1, nd);
tdis = abs(trow - tcol);
sigmatime = exp(-(tdis ./ corrlength) .^ 2);
sigma0 = max(var(true_imp - m0, 1), 1.0e-6);
sigmaprior = sigma0 * sigmatime;
sigmaprior = (sigmaprior + sigmaprior') / 2 + 1.0e-8 * eye(nd);
inv_sigmaprior = pinv(sigmaprior);

% -----------------------------
% Aggressively optimized AIES parameters
% -----------------------------
nwalkers = 200;
if mod(nwalkers, 2) ~= 0
    error('nwalkers must be even.');
end

ndim = nd;
a = 2;
niter = 10000;
burnin = round(0.1 * niter);
nsaved = niter - burnin;

walkers = mvnrnd(m0, sigmaprior, nwalkers);
logp = log_posterior_batch(walkers, d_obs, w, sigma, m0, inv_sigmaprior);

logp_hist = zeros(niter, 1);
posterior_mean_accum = zeros(1, nd);
saved_samples = zeros(max(nsaved, 1), nd);
saved_count = 0;
accept_hist = zeros(niter, 1);

for iter = 1:niter
    perm_idx = randperm(nwalkers);
    left_idx = perm_idx(1:(nwalkers / 2));
    right_idx = perm_idx((nwalkers / 2 + 1):end);

    [walkers, logp, acc_left] = stretch_move_batch( ...
        walkers, logp, left_idx, right_idx, a, ndim, d_obs, w, sigma, m0, inv_sigmaprior);

    [walkers, logp, acc_right] = stretch_move_batch( ...
        walkers, logp, right_idx, left_idx, a, ndim, d_obs, w, sigma, m0, inv_sigmaprior);

    logp_hist(iter) = mean(logp);
    accept_hist(iter) = 0.5 * (acc_left + acc_right);

    if iter > burnin
        saved_count = saved_count + 1;
        walker_mean = mean(walkers, 1);
        posterior_mean_accum = posterior_mean_accum + walker_mean;
        saved_samples(saved_count, :) = walker_mean;
    end

    if mod(iter, 100) == 0
        fprintf('Iteration %d / %d, acceptance %.2f%%\n', ...
            iter, niter, 100 * accept_hist(iter));
    end
end

fprintf('Average Acceptance Rate: %.2f%%\n', 100 * mean(accept_hist));

% Time = (1:nd) * dt;
postImp = walkers';
Impmap = zeros(nm,1);
if saved_count > 0
    m_post = saved_samples(1:saved_count, :);
    m_est = posterior_mean_accum / saved_count;
else
    m_post = mean(walkers, 1);
    m_est = m_post;
end
m_est =mean(postImp,2);
Impv = [7:0.001:11];
for j=1:nm


[f1(j,:),xi1]= ksdensity(postImp(j,:),Impv ,'Kernel', 'epanechnikov','Bandwidth',0.1);
f1(j,:) = f1(j,:)/sum(f1(j,:));
[~,Impmapind]=max(f1(j,:));
Impmap(j,1)=xi1(Impmapind);
end


for i=1:nm
    ImpP05(i,1) = quantile(postImp(i,:),0.05);
    ImpP95(i,1) = quantile(postImp(i,:),0.95);
end
% figure(1)
% figSize = [8 14];
% set(gcf, 'Unit', 'Centimeters', 'Position', [10, 5, figSize]);
% plot(samples, Time, 'b', 'LineWidth', 1);
% hold on;
% plot(m0, Time, 'g', 'LineWidth', 2);
% plot(true_imp, Time, 'k', 'LineWidth', 2);
% plot(m_est, Time, 'r', 'LineWidth', 2);
% axis tight; grid on; box on; set(gca, 'YDir', 'reverse');
% xlabel('Impedance(Km/s*g/cm3)'); ylabel('Time(s)');
% title('Faster AIES-based Impedance Inversion');
ntr = 1;
true_imp = true_imp';
m0 = m0';
figSize=[8 14];% A4: Single Column With Large Size
figure(1)
set(gcf,'Unit','Centimeters','Position',[10,2,figSize]);
% plot(postImp, Time, 'k', 'LineWidth', 1);
% hold on;
plot( postImp(:,1:end),Time,  'Color', [0.602, 0.602, 0.602], 'LineWidth', 0.5)
hold on;
plot(true_imp(:,ntr), Time, 'k', 'LineWidth', 1.5);
plot(m0(:,ntr), Time, 'Color',[ 0    0.4471    0.7412], 'LineWidth', 1);
% plot(postVp,Time, 'color',[0.3020 0.7451 0.9333], 'LineWidth', 1);
plot(m_est(:,ntr), Time, 'r', 'LineWidth', 1.5);
plot(Impmap, Time, 'Color',[ 0.0588    1.0000    1.0000], 'LineWidth', 1.5);
plot(ImpP05, Time, 'r--', 'LineWidth', 1);
plot(ImpP95, Time, 'r--', 'LineWidth', 1);
% plot(VpP05, Time, 'color',[0.8510 0.3255 0.0980], 'LineWidth', 1);
% plot(VpP95, Time, 'color',[0.8510 0.3255 0.0980], 'LineWidth', 1);
axis tight; grid on; box on; set(gca, 'YDir', 'reverse');
xlabel('Impedance (km/s*g/cm^3)'); ylabel('Time (s)');
defaultAxes();

figSize=[16 10];
figure(2);
set(gcf,'Unit','Centimeters','Position',[10,2,figSize]);
semilogx(1:10000,-logp_hist, 'color', [0    0.4471    0.7412], 'LineWidth', 1.2);
xlabel('AIES iteration');
ylabel('Negative mean log posterior');
grid on;
defaultAxes
% set(gca, 'YDir', 'reverse');

figure(3);
set(gcf,'Unit','Centimeters','Position',[10,2,figSize]);
plot(100 * accept_hist, 'color', [0.4660, 0.6740, 0.1880], 'LineWidth', 1.5);
xlabel('AIES Iteration');
ylabel('Acceptance rate (%)');
grid on;
defaultAxes
figSize=[16 12];% A4: Single Column With Large Size
figure(4)
set(gcf,'Unit','Centimeters','Position',[10,2,figSize]);
subplot(131)
plot(Vp(:,1), Time, 'k', 'LineWidth', 2);
% plot(VpP05, Time, 'color',[0.8510 0.3255 0.0980], 'LineWidth', 1);
% plot(VpP95, Time, 'color',[0.8510 0.3255 0.0980], 'LineWidth', 1);
axis tight; grid on; box on; set(gca, 'YDir', 'reverse');
xlabel('P-wave velocity (km/s)'); ylabel('Time (s)');
set(gca, 'XTick', [3.8,4.2,4.6]);             
set(gca, 'XTickLabel', {'3.8','4.2','4.6'});  
defaultAxes
subplot(132)
plot(Rho(:,1), Time, 'k', 'LineWidth', 2);
% plot(Vslp, Time, 'r--', 'LineWidth', 1);
% plot(Vsup, Time, 'r--', 'LineWidth', 1);
axis tight; grid on; box on; set(gca, 'YDir', 'reverse');
xlabel('Density (g/cm^3)'); 
set(gca, 'YTickLabel', []);
defaultAxes
subplot(133)
plot(true_imp, Time, 'k', 'LineWidth', 2);
% plot(Rholp, Time, 'r--', 'LineWidth', 1);
% plot(Rhoup, Time, 'r--', 'LineWidth', 1);
axis tight; grid on; box on; set(gca, 'YDir', 'reverse');
xlabel('Impedance(km/s*g/cm^3)'); 
set(gca, 'XTick', [8.5,9.5,10.5]);             
set(gca, 'XTickLabel', {'8.5','9.5','10.5'});  
set(gca, 'YTickLabel', []);
defaultAxes

for i=1:nd
    ImpP05(i,1) = quantile(postImp(:,i),0.05);
    ImpP95(i,1) = quantile(postImp(:,i),0.95);
end
Coverage_Ratio = [sum(true_imp>=ImpP05 & true_imp<ImpP95)/nm*100];
Imp_min = min(true_imp);
Imp_max = max(true_imp);
rmseImp_mean = sqrt(sum((m_est - true_imp) .^ 2) / nm);
rrmseImp_mean = rmseImp_mean/(Imp_max-Imp_min);
rmseImp_map = sqrt(sum((Impmap - true_imp) .^ 2) / nm);
rrmseImp_map = rmseImp_map/(Imp_max-Imp_min);


ccImp = corrcoef(m_est, true_imp);
ccImp_mean = ccImp(1, 2);
ccImp = corrcoef(Impmap, true_imp);
ccImp_map = ccImp(1, 2);

function [walkers, logp, acceptance_rate] = stretch_move_batch( ...
    walkers, logp, move_idx, ref_idx, a, ndim, d_obs, w, sigma, m0, inv_sigmaprior)

    nmove = numel(move_idx);
    ref_pick = ref_idx(randi(numel(ref_idx), nmove, 1));

    current = walkers(move_idx, :);
    reference = walkers(ref_pick, :);
    z = ((a - 1) * rand(nmove, 1) + 1) .^ 2 / a;
    proposal = reference + (current - reference) .* z;

    logp_new = log_posterior_batch(proposal, d_obs, w, sigma, m0, inv_sigmaprior);
    log_alpha = (ndim - 1) * log(z) + (logp_new - logp(move_idx));
    accept = log(rand(nmove, 1)) < min(0, log_alpha);

    accepted_idx = move_idx(accept);
    walkers(accepted_idx, :) = proposal(accept, :);
    logp(accepted_idx) = logp_new(accept);
    acceptance_rate = mean(accept);
end

function lp = log_posterior_batch(models, d_obs, w, sigma, m0, inv_sigmaprior)
    refl = diff(models, 1, 2) ./ (models(:, 1:end-1) + models(:, 2:end));
    d_syn = conv2(refl, w, 'same');

    residual = d_syn - d_obs;
    misfit = sum(residual .^ 2, 2);
    ll = -0.5 *misfit / (sigma ^ 2);

    dm = models - m0;
    prior = -0.5 * sum((dm * inv_sigmaprior) .* dm, 2);

    lp = ll + prior;
end
