%% 高精度梁最优代表坐标系识别示例
% 该脚本展示了如何在存在小幅弹性变形和测量噪声的情况下，
% 通过鲁棒的分阶段优化流程同时估计梁的刚体运动与弹性变形。
%
% 主要特点：
%   1. 以SVD刚体配准作为稳健初始值。
%   2. 采用带二阶正则的变形场求解器，提高物理一致性。
%   3. 使用Huber权重的交替最小化策略抑制异常点。
%   4. 通过自定义Gauss-Newton微调进一步压缩残差。
%
% 注意：脚本不依赖优化工具箱，所有求解均基于线性代数运算。
%
% 作者：GPT-Coaex 项目示例
% 日期：2024

clear; clc; close all;

%% ================= 1. 场景构造 =================
fprintf('===============================================================\n');
fprintf('  高精度梁坐标系识别 (鲁棒正则化版本)\n');
fprintf('===============================================================\n\n');

params = struct();
params.L = 1.0;               % 梁长度 (m)
params.n_nodes = 31;          % 网格节点数
params.s = linspace(0, params.L, params.n_nodes)';

% 未变形梁参考坐标
s_i = zeros(params.n_nodes, 3);
s_i(:, 1) = params.s;

% 真实刚体运动 (平移+欧拉角)
r_true = [0.3; 0.2; 0.1];
theta_true = [0.18; -0.11; 0.25];
R_true = eulerXYZ2rotm(theta_true);

fprintf('【真实场景】\n');
fprintf('  梁长          : %.2f m\n', params.L);
fprintf('  节点数        : %d\n', params.n_nodes);
fprintf('  真实平移      : [% .4f, % .4f, % .4f] m\n', r_true);
fprintf('  真实欧拉角    : [% .4f, % .4f, % .4f] rad\n\n', theta_true);

% 生成物理一致的弹性变形
elastic_true = generateBeamDeformation(params.s, params.L);

% 合成测量 (包含微小噪声)
noise_level = 5e-6;           % 高精度测量噪声
r_pi = zeros(params.n_nodes, 3);
for i = 1:params.n_nodes
    def_point = s_i(i, :)' + elastic_true(i, :)';
    r_pi(i, :) = (r_true + R_true * def_point)';
end
r_pi = r_pi + noise_level * randn(size(r_pi));

max_def = max(vecnorm(elastic_true, 2, 2));
fprintf('  最大真实变形  : %.4f m (%.2f%% 梁长)\n', max_def, 100 * max_def / params.L);
fprintf('  噪声标准差    : %.1e m\n\n', noise_level);

%% ================= 2. 参数设置 =================
opts = struct();
opts.max_iter       = 80;      % 交替优化最大迭代
opts.tol            = 1e-11;   % 相对收敛阈值
opts.lambda_bend    = 2e-3;    % 二阶弯曲正则
opts.lambda_grad    = 2e-4;    % 一阶平滑正则
opts.lambda_axial   = 5e-4;    % 轴向伸长抑制
opts.lambda_tikh    = 1e-8;    % 数值稳定项
opts.huber_delta    = 5e-4;    % Huber阈值
opts.min_weight     = 1e-2;    % 权重下限
opts.verbose        = true;    % 打印细节
opts.gn_max_iter    = 12;      % 微调迭代
opts.gn_tol         = 1e-12;   % 微调收敛

%% ================= 3. 初始估计 =================
[r_init, R_init] = solveRigidTransform(s_i, r_pi);
init_rmse = computeRMSE(s_i, r_pi, r_init, R_init);
fprintf('【阶段0】SVD刚体初始化\n');
fprintf('  初始RMSE      : %.3e m\n\n', init_rmse);

%% ================= 4. 鲁棒正则化估计 =================
[r_opt, R_opt, u_opt, solver_info] = estimateBeamMotion(s_i, r_pi, r_init, R_init, opts);

fprintf('\n【求解统计】\n');
fprintf('  迭代次数      : %d\n', solver_info.iterations);
fprintf('  最终成本      : %.3e\n', solver_info.cost(end));
fprintf('  最终RMSE      : %.3e m\n\n', solver_info.rmse);

%% ================= 5. 精度评估 =================
pos_error = norm(r_opt - r_true);
R_err = R_true' * R_opt;
angle_error = acos(max(-1,min(1,(trace(R_err) - 1) / 2)));

theta_opt = rotm2eulerXYZ(R_opt);

deform_error_vec = u_opt - elastic_true;
deform_rms = sqrt(mean(sum(deform_error_vec.^2, 2)));
deform_max = max(vecnorm(deform_error_vec, 2, 2));
corr_matrix = corrcoef(vecnorm(u_opt,2,2), vecnorm(elastic_true,2,2));
corr_value = corr_matrix(1,2);

fprintf('【位置精度】\n');
fprintf('  估计平移      : [% .8f, % .8f, % .8f] m\n', r_opt);
fprintf('  误差范数      : %.3e m\n', pos_error);
fprintf('  相对误差      : %.3e %%\n\n', 100 * pos_error / norm(r_true));

fprintf('【姿态精度】\n');
fprintf('  估计欧拉角    : [% .8f, % .8f, % .8f] rad\n', theta_opt);
fprintf('  姿态角误差    : %.3e rad (%.6f°)\n\n', angle_error, angle_error * 180 / pi);

fprintf('【变形精度】\n');
fprintf('  RMS误差       : %.3e m\n', deform_rms);
fprintf('  最大误差      : %.3e m\n', deform_max);
fprintf('  相关系数      : %.6f\n\n', corr_value);

%% ================= 6. 收敛性判定 =================
criteria = {
    solver_info.rmse < 1e-8,        '最终RMSE < 1e-8 m';
    pos_error           < 1e-8,     '平移误差 < 1e-8 m';
    angle_error         < 1e-6,     '姿态误差 < 1e-6 rad';
    deform_rms          < 5e-7,     '变形RMS < 5e-7 m';
    deform_max          < 2e-6,     '变形最大误差 < 2e-6 m';
    solver_info.converged,          '迭代过程收敛';
    corr_value          > 0.9999,   '变形相关系数 > 0.9999'
    };

fprintf('【收敛判据】\n');
passed = 0;
for i = 1:size(criteria, 1)
    if criteria{i, 1}
        fprintf('  ✓ %s\n', criteria{i, 2});
        passed = passed + 1;
    else
        fprintf('  ✗ %s\n', criteria{i, 2});
    end
end
fprintf('  成功率        : %.0f%% (%d/%d)\n\n', 100 * passed / size(criteria,1), passed, size(criteria,1));

%% ================= 7. 可视化 =================
visualizeBeamResult(s_i, r_pi, elastic_true, u_opt, r_true, R_true, r_opt, R_opt, solver_info);

fprintf('\n执行完成！\n');

%% =====================================================================
%%                               函数区
%% =====================================================================

function [r_opt, R_opt, u_opt, info] = estimateBeamMotion(s_i, r_pi, r0, R0, opts)
%ESTIMATEBEAMMOTION  交替优化求解刚体+弹性变形
%
% 输入：
%   s_i  - 未变形梁节点 (n×3)
%   r_pi - 测量点 (n×3)
%   r0, R0 - 初始刚体运动
%   opts - 参数结构体
% 输出：
%   r_opt, R_opt, u_opt - 估计结果
%   info - 求解统计信息

n = size(s_i, 1);
weights = ones(n, 1);
r = r0;
R = R0;
u = zeros(n, 3);

info.cost = [];
info.iterations = 0;
info.converged = false;

for iter = 1:opts.max_iter
    % 计算等效观测变形 (刚体坐标系)
    d = zeros(n, 3);
    for i = 1:n
        d(i, :) = (R' * (r_pi(i, :)' - r))' - s_i(i, :);
    end

    % 解正则化弹性场
    u_prev = u;
    u = solveRegularizedDeformation(d, weights, s_i(:, 1), opts);

    % 更新刚体运动
    s_def = s_i + u;
    [r, R] = weightedRigidTransform(s_def, r_pi, weights);

    % 计算残差与Huber权重
    residuals = r + (R * s_def')' - r_pi;
    weights = computeHuberWeights(residuals, opts.huber_delta, opts.min_weight);

    cost = sum(weights .* sum(residuals.^2, 2));
    info.cost(end + 1, 1) = cost;

    if opts.verbose && (iter == 1 || mod(iter, 10) == 0)
        rmse_iter = sqrt(mean(sum(residuals.^2, 2)));
        fprintf('  迭代 %3d | 成本: %.3e | RMSE: %.3e\n', iter, cost, rmse_iter);
    end

    delta_u = norm(u - u_prev, 'fro');
    denom = max(1, norm(u_prev, 'fro'));
    if delta_u / denom < opts.tol
        info.converged = true;
        break;
    end
end

info.iterations = iter;

% Gauss-Newton 精细化
s_def = s_i + u;
[r_refined, R_refined] = refineRigidMotion(r, R, s_def, r_pi, opts);

residuals = r_refined + (R_refined * s_def')' - r_pi;
info.rmse = sqrt(mean(sum(residuals.^2, 2)));
info.cost(end + 1, 1) = sum(vecnorm(residuals, 2, 2).^2);

r_opt = r_refined;
R_opt = R_refined;
u_opt = u;
end

function u = solveRegularizedDeformation(d, weights, s_coord, opts)
%SOLVEREGULARIZEDDEFORMATION  变形场正则化求解器
n = size(d, 1);
w = max(weights, opts.min_weight);

D1 = buildFirstDifference(s_coord);
D2 = buildSecondDifference(s_coord);

L1 = D1' * D1;
L2 = D2' * D2;
W = spdiags(w, 0, n, n);
I = speye(n);

A_common = W + opts.lambda_bend * L2 + opts.lambda_grad * L1 + opts.lambda_tikh * I;

A_x = A_common + opts.lambda_axial * L1;
A_y = A_common;
A_z = A_common;

A_x = enforceDirichlet(A_x, 1);
A_y = enforceDirichlet(A_y, 1);
A_z = enforceDirichlet(A_z, 1);

b_x = W * d(:, 1);
b_y = W * d(:, 2);
b_z = W * d(:, 3);

b_x(1) = 0; b_y(1) = 0; b_z(1) = 0;

u = zeros(n, 3);

u(:, 1) = A_x \ b_x;
u(:, 2) = A_y \ b_y;
u(:, 3) = A_z \ b_z;
end

function A = enforceDirichlet(A, idx)
%ENFORCEDIRICHLET  设置Dirichlet边界 (u(idx)=0)
A(idx, :) = 0;
A(:, idx) = 0;
A(idx, idx) = 1;
end

function D1 = buildFirstDifference(s_coord)
% 构建一阶差分矩阵
n = numel(s_coord);
rows = n - 1;
D1 = spalloc(rows, n, 2 * rows);
for i = 1:rows
    h = s_coord(i + 1) - s_coord(i);
    D1(i, i) = -1 / h;
    D1(i, i + 1) = 1 / h;
end
end

function D2 = buildSecondDifference(s_coord)
% 构建二阶差分矩阵
n = numel(s_coord);
rows = n - 2;
D2 = spalloc(rows, n, 3 * rows);
for i = 1:rows
    h1 = s_coord(i + 1) - s_coord(i);
    h2 = s_coord(i + 2) - s_coord(i + 1);
    h = (h1 + h2) / 2;
    D2(i, i) = 1 / h^2;
    D2(i, i + 1) = -2 / h^2;
    D2(i, i + 2) = 1 / h^2;
end
end

function [r, R] = weightedRigidTransform(source, target, weights)
%WEIGHTEDRIGIDTRANSFORM  加权刚体配准 (Kabsch)
w = max(weights, eps);
W = w / sum(w);
centroid_s = W' * source;
centroid_t = W' * target;

S = source - centroid_s;
T = target - centroid_t;

H = (S .* W)';
H = H * T;

[U, ~, V] = svd(H);
R = V * U';
if det(R) < 0
    V(:, 3) = -V(:, 3);
    R = V * U';
end

r = centroid_t' - R * centroid_s';
end

function [r, R] = solveRigidTransform(source, target)
%SOLVERIGIDTRANSFORM  标准Kabsch刚体配准
centroid_s = mean(source, 1);
centroid_t = mean(target, 1);

S = source - centroid_s;
T = target - centroid_t;

H = S' * T;
[U, ~, V] = svd(H);
R = V * U';
if det(R) < 0
    V(:, 3) = -V(:, 3);
    R = V * U';
end

r = centroid_t' - R * centroid_s';
end

function weights = computeHuberWeights(residuals, delta, min_w)
%COMPUTEHUBERWEIGHTS  基于Huber函数的点权重
r_norm = vecnorm(residuals, 2, 2);
weights = ones(size(r_norm));
mask = r_norm > delta;
weights(mask) = delta ./ r_norm(mask);
weights = max(weights, min_w);
end

function [r_new, R_new] = refineRigidMotion(r, R, source, target, opts)
%REFINERIGIDMOTION  Gauss-Newton 微调
r_new = r;
R_new = R;

for iter = 1:opts.gn_max_iter
    residuals = zeros(size(source));
    J = zeros(3 * size(source,1), 6);
    rhs = zeros(3 * size(source,1), 1);
    idx = 1;
    for i = 1:size(source, 1)
        s = source(i, :)';
        predicted = r_new + R_new * s;
        res = predicted - target(i, :)';
        residuals(i, :) = res';

        J_block = [eye(3), -R_new * skewMatrix(s)];
        J(idx:idx+2, :) = J_block;
        rhs(idx:idx+2) = -res;
        idx = idx + 3;
    end

    delta = J \ rhs;
    dr = delta(1:3);
    domega = delta(4:6);

    r_new = r_new + dr;
    R_new = R_new * expSO3(domega);

    if norm(delta) < opts.gn_tol
        break;
    end
end

% 确保R_new仍为正交矩阵
[U, ~, V] = svd(R_new);
R_new = U * V';
if det(R_new) < 0
    U(:, 3) = -U(:, 3);
    R_new = U * V';
end
end

function M = skewMatrix(v)
%SKEWMATRIX  反对称矩阵
M = [   0   -v(3)  v(2);
        v(3)   0   -v(1);
       -v(2)  v(1)   0  ];
end

function R = expSO3(omega)
%EXPSO3  so(3)到SO(3)的指数映射
theta = norm(omega);
if theta < 1e-12
    R = eye(3);
    return;
end
k = omega / theta;
K = skewMatrix(k);
R = eye(3) + sin(theta) * K + (1 - cos(theta)) * (K * K);
end

function rmse = computeRMSE(source, target, r, R)
%COMPUTERMSE  均方根误差
pred = r' + (R * source')';
res = pred - target;
rmse = sqrt(mean(sum(res.^2, 2)));
end

function elastic = generateBeamDeformation(s, L)
%GENERATEBEAMDEFORMATION  物理可行的梁变形场
n = numel(s);
xi = s / L;

elastic = zeros(n, 3);

% 悬臂梁在端部集中力下的典型弯曲形状 (y方向)
elastic(:, 2) = 0.05 * (xi.^2) .* (3 - 2 * xi);

% 第一扭转模态 (绕x轴) 引起的z方向摆动
elastic(:, 3) = 0.03 * sin(pi * xi);

% 轻微轴向拉伸 (x方向)
elastic(:, 1) = -0.0015 * xi + 0.0004 * xi.^2;

% 考虑沿梁的轻微扭转 (旋转后再叠加)
for i = 1:n
    twist = 0.12 * xi(i);
    R_twist = [1, 0, 0;
               0, cos(twist), -sin(twist);
               0, sin(twist),  cos(twist)];
    elastic(i, :) = (R_twist * elastic(i, :)')';
end

% 根部固定条件
elastic(1, :) = 0;
end

function visualizeBeamResult(s_i, r_pi, deform_true, deform_est, r_true, R_true, r_est, R_est, info)
%VISUALIZEBEAMRESULT  可视化对比
n = size(s_i, 1);
param = s_i(:, 1);

rigid_true = zeros(n, 3);
rigid_est = zeros(n, 3);
def_true = zeros(n, 3);
def_est = zeros(n, 3);
for i = 1:n
    rigid_true(i, :) = (r_true + R_true * s_i(i, :)')';
    rigid_est(i, :)  = (r_est  + R_est  * s_i(i, :)')';
    def_true(i, :)   = (r_true + R_true * (s_i(i, :)' + deform_true(i, :)'))';
    def_est(i, :)    = (r_est  + R_est  * (s_i(i, :)' + deform_est(i, :)'))';
end

figure('Name', 'Beam Reconstruction', 'NumberTitle', 'off', 'Position', [80, 80, 1200, 800]);

subplot(2,2,1); hold on; grid on; axis equal;
plot3(s_i(:,1), s_i(:,2), s_i(:,3), 'k--', 'LineWidth', 1.2, 'DisplayName', '未变形');
plot3(r_pi(:,1), r_pi(:,2), r_pi(:,3), 'b.', 'MarkerSize', 12, 'DisplayName', '测量');
plot3(def_true(:,1), def_true(:,2), def_true(:,3), 'g-', 'LineWidth', 2, 'DisplayName', '真实变形');
plot3(def_est(:,1), def_est(:,2), def_est(:,3), 'r--', 'LineWidth', 2, 'DisplayName', '估计变形');
legend('Location', 'best');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('三维几何对比');
view(3);

subplot(2,2,2); hold on; grid on;
plot(param, vecnorm(deform_true,2,2), 'g-', 'LineWidth', 2, 'DisplayName', '真实范数');
plot(param, vecnorm(deform_est,2,2), 'r--', 'LineWidth', 2, 'DisplayName', '估计范数');
legend('Location', 'best');
xlabel('s (m)'); ylabel('||u|| (m)');
title('变形范数比较');

subplot(2,2,3); hold on; grid on;
plot(param, deform_est(:,1) - deform_true(:,1), 'r-', 'DisplayName', '\Delta u_x');
plot(param, deform_est(:,2) - deform_true(:,2), 'g-', 'DisplayName', '\Delta u_y');
plot(param, deform_est(:,3) - deform_true(:,3), 'b-', 'DisplayName', '\Delta u_z');
legend('Location', 'best');
xlabel('s (m)'); ylabel('误差 (m)');
title('分量误差');

subplot(2,2,4); hold on; grid on;
plot(1:numel(info.cost), info.cost, 'LineWidth', 2);
xlabel('迭代'); ylabel('成本');
title('优化收敛历程');
end

function R = eulerXYZ2rotm(theta)
%EULERXYZ2ROTM  XYZ顺序欧拉角到旋转矩阵
cx = cos(theta(1)); sx = sin(theta(1));
cy = cos(theta(2)); sy = sin(theta(2));
cz = cos(theta(3)); sz = sin(theta(3));
Rx = [1 0 0; 0 cx -sx; 0 sx cx];
Ry = [cy 0 sy; 0 1 0; -sy 0 cy];
Rz = [cz -sz 0; sz cz 0; 0 0 1];
R = Rz * Ry * Rx;
end

function euler = rotm2eulerXYZ(R)
%ROTM2EULERXYZ  旋转矩阵到XYZ欧拉角
if abs(R(3,1)) < 1
    euler(2) = -asin(R(3,1));
    euler(1) = atan2(R(3,2), R(3,3));
    euler(3) = atan2(R(2,1), R(1,1));
else
    % 接近奇异，采用备用方案
    euler(3) = 0;
    if R(3,1) <= -1
        euler(2) = pi/2;
        euler(1) = atan2(R(1,2), R(1,3));
    else
        euler(2) = -pi/2;
        euler(1) = atan2(-R(1,2), -R(1,3));
    end
end
euler = euler(:);
end

