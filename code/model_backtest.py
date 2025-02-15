from ysquant.api.stock.daily_sharry_reader import DailySharryReader
from daily2monthly import transform_to_monthly

import pandas as pd
import numpy as np
import os
import time
import gc
import bottleneck as bn
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import List, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import SGDRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from scipy.optimize import minimize
from itertools import product
from tqdm import tqdm

# import backtrader as bt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置显示中文字体
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

# %matplotlib inline

# GLM用函数：二阶样条基函数
def spline_basis(z, knots):
    # 基函数的第一部分是 1 和 z
    basis = [np.ones_like(z), z]

    # 添加二阶样条基函数
    for c in knots:
        basis.append(np.maximum(z - c, 0) ** 2)

    # 将基函数堆叠为矩阵
    return np.column_stack(basis)

# GLM用函数：Huber损失函数+GroupLasso正则化
class GroupLassoHuberRegressor:
    def __init__(self, epsilon=0.5, lambda_=0.1, max_iter=1000):
        self.epsilon = epsilon  # Huber 损失函数的阈值参数
        self.lambda_ = lambda_  # Group Lasso 正则化参数
        self.max_iter = max_iter  # 最大迭代次数
        self.coef_ = None       # 模型系数

    def huber_loss(self, y_true, y_pred):
        residual = y_true - y_pred
        mask = np.abs(residual) <= self.epsilon
        return np.where(mask, 0.5 * residual**2, self.epsilon * (np.abs(residual) - 0.5 * self.epsilon))

    def group_lasso_penalty(self, theta, group_sizes):
        penalty = 0
        start = 0
        for size in group_sizes:
            end = start + size
            group = theta[start:end]
            penalty += np.sqrt(np.sum(group**2))
            start = end
        return self.lambda_ * penalty

    def objective(self, theta, X, y, group_sizes):
        # 目标函数：Huber + GroupLasso
        y_pred = X @ theta
        loss = np.sum(self.huber_loss(y, y_pred))
        penalty = self.group_lasso_penalty(theta, group_sizes)
        return loss + penalty

    def fit(self, X, y, group_sizes):
        # 初始化参数
        theta_init = np.zeros(X.shape[1])

        # 最小化目标函数
        result = minimize(
            fun=self.objective,
            x0=theta_init,
            args=(X, y, group_sizes),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter}
        )

        # 保存最优参数
        self.coef_ = result.x
        return self

    def predict(self, X):
        return X @ self.coef_

class HuberElasticNet(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, l1_ratio=0.5, delta=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha  # Regularization strength
        self.l1_ratio = l1_ratio  # L1 vs L2 ratio
        self.delta = delta  # Huber loss delta
        self.max_iter = max_iter
        self.tol = tol

    def huber_loss(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * error**2
        linear_loss = self.delta * np.abs(error) - 0.5 * self.delta**2
        return np.where(is_small_error, squared_loss, linear_loss).mean()

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_features = X.shape[1]
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.max_iter):
            y_pred = np.dot(X, self.coef_) + self.intercept_
            grad = self._compute_gradient(X, y, y_pred)
            self.coef_ -= self.tol * grad['coef']
            self.intercept_ -= self.tol * grad['intercept']

            # Check convergence
            if np.linalg.norm(grad['coef']) < self.tol:
                break

        return self

    def _compute_gradient(self, X, y, y_pred):
        error = y - y_pred
        is_small_error = np.abs(error) <= self.delta
        grad_loss = np.where(is_small_error, error, self.delta * np.sign(error))

        grad_coef = -np.dot(X.T, grad_loss) / len(y)
        grad_intercept = -grad_loss.mean()

        # Add Elastic Net regularization
        grad_coef += self.alpha * self.l1_ratio * np.sign(self.coef_)  # L1
        grad_coef += self.alpha * (1 - self.l1_ratio) * self.coef_  # L2

        return {'coef': grad_coef, 'intercept': grad_intercept}

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.dot(X, self.coef_) + self.intercept_

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, input_dim=6, output_dim=1, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, output_dim)
        self.device = device
        self.d_feat = input_dim

    def forward(self, src):
        # src [N, F*T] --> [N, T, F]
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])

        return output.squeeze()
    
class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        """
        初始化神经网络
        :param input_size: 输入特征的维度
        :param output_size: 输出特征的维度
        :param hidden_size: 每个隐藏层的神经元数量
        :param num_layers: 全连接层的数量（不包括输入和输出层）
        """
        super(Net, self).__init__()

        layers = []
        if num_layers == 0:
            # 如果没有隐藏层，直接连接输入到输出
            layers.append(nn.Linear(input_size, output_size))
        else:
            # 输入层到第一个隐藏层
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LeakyReLU())

            # 隐藏层之间
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.LeakyReLU())

            # 最后一层连接到输出
            layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据
        :return: 输出结果
        """
        return self.network(x)
    
# 所有用到的模型
class models:

    @staticmethod
    def OLS_Huber(X_train, X_valid, y_train, y_valid, X_test):

        # 定义参数网格
        param_grid = {
            'epsilon': [1.1, 1.35, 1.5],  # Huber损失函数的阈值参数
            'alpha': [0.0001, 0.001, 0.01]  # 正则化强度
        }

        # 初始化最优参数和最优R2分数
        best_params = None
        best_r2 = -np.inf
        best_model = None

        # 遍历所有参数组合
        for epsilon, alpha in product(param_grid['epsilon'], param_grid['alpha']):
            # 创建HuberRegressor模型
            huber = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000)

            # 在训练集上训练模型
            huber.fit(X_train, y_train)

            # 在验证集上预测
            y_valid_pred = huber.predict(X_valid)

            # 计算验证集上的R2分数
            r2 = r2_score(y_valid, y_valid_pred)

            # 如果当前参数组合的R2分数更高，则更新最优参数和模型
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'epsilon': epsilon, 'alpha': alpha}
                best_model = huber

        # 输出最优参数和验证集上的R2分数
        # print("最优参数:", best_params)
        # print("验证集的R2:", best_r2)

        # 使用最优模型预测测试集
        y_pred = best_model.predict(X_test)

        return y_pred

    @staticmethod
    def PCR(X_train, X_valid, y_train, y_valid, X_test):

        # 定义主成分个数
        numpc =np.arange(1, X_train.shape[1]//2, 5).tolist()

        # 初始化最优参数和最优R2分数
        best_params = None
        best_r2 = -np.inf
        best_pca_model = None
        best_ols_model = None

        # 遍历所有参数组合
        for i, n in enumerate(numpc):
            # 创建PCA模型
            pca_model = PCA(n_components=n)

            # 数据降维
            X_train_pca = pca_model.fit_transform(X_train)
            X_valid_pca = pca_model.transform(X_valid)

            # 线性回归
            ols_model = LinearRegression()

            # 在训练集上训练模型
            ols_model.fit(X_train_pca, y_train)

            # 在验证集上预测
            y_valid_pred = ols_model.predict(X_valid_pca)

            # 计算验证集上的R2分数
            r2 = r2_score(y_valid, y_valid_pred)

            # 如果当前参数组合的R2分数更高，则更新最优参数和模型
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'n_components': n}
                best_pca_model = pca_model
                best_ols_model = ols_model

        # 输出最优参数和验证集上的R2分数
        # print("最优参数:", best_params)
        # print("验证集的R2:", best_r2)

        # 使用最优模型预测测试集
        X_test_pca = best_pca_model.transform(X_test)
        y_pred = best_ols_model.predict(X_test_pca)

        return y_pred

    @staticmethod
    def PLS(X_train, X_valid, y_train, y_valid, X_test):

        # 定义主成分个数
        numpc =np.arange(1, X_train.shape[1]//2, 5).tolist()

        # 初始化最优参数和最优R2分数
        best_params = None
        best_r2 = -np.inf
        best_model = None

        # 遍历所有参数组合
        for i, n in enumerate(numpc):
            # 创建PLSRegression模型
            pls_model = PLSRegression(n_components=n)

            # 在训练集上训练模型
            pls_model.fit(X_train, y_train)

            # 在验证集上预测
            y_valid_pred = pls_model.predict(X_valid)

            # 计算验证集上的R2分数
            r2 = r2_score(y_valid, y_valid_pred)

            # 如果当前参数组合的R2分数更高，则更新最优参数和模型
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'n_components': n}
                best_model = pls_model

        # 输出最优参数和验证集上的R2分数
        # print("最优参数:", best_params)
        # print("验证集的R2:", best_r2)

        # 使用最优模型预测测试集
        y_pred = best_model.predict(X_test)

        return y_pred.reshape(-1)

    @staticmethod
    def GLM_with_Spline_GroupLasso(X_train, X_valid, y_train, y_valid, X_test):
        best_score = -np.inf
        best_model = None
        best_knots = None

        num_knots_candidates = [3, 4, 5, 6]  # 候选节点数量
        lambda_candidates = [0.01, 0.1, 1.0]  # 候选正则化参数

        for num_knots in num_knots_candidates:
            # 基于分位数确定节点位置
            knots = np.quantile(X_train, np.linspace(0, 1, num_knots + 2)[1:-1])

            for lambda_ in lambda_candidates:

                # 生成样条基函数
                X_train_spline = spline_basis(X_train, knots)
                X_val_spline = spline_basis(X_valid, knots)

                # 定义分组大小（每个特征对应的样条基函数数量）
                group_sizes = [len(knots) + 2] * X_train.shape[1]

                # 训练模型
                glm_model = GroupLassoHuberRegressor(lambda_=lambda_)
                glm_model.fit(X_train_spline, y_train, group_sizes)

                # 评估模型
                y_val_pred = glm_model.predict(X_val_spline)
                score = r2_score(y_valid, y_val_pred)

                if score > best_score:
                    best_score = score
                    best_model = glm_model
                    best_knots = knots

            # 使用最优模型预测测试集
            X_test_spline = spline_basis(X_test, best_knots)
            y_pred = best_model.predict(X_test_spline)

        return y_pred

    @staticmethod
    def ElasticNet_Huber(X_train, X_valid, y_train, y_valid, X_test):
        best_r2 = -np.inf
        best_model = None
        best_knots = None

        alpha_rates = [3, 4, 5, 6]  # 候选节点数量
        l1_ratios = [0, 0.3, 0.6, 0.9]  # 候选正则化参数

        for alpha, l1_rate in product(alpha_rates, l1_ratios):
            # 初始化 Elastic Net 模型
            # 设置 alpha 为正则化强度，l1_ratio 为 Lasso的比例，为0就表示都是ridge，为1表示都lasso
            # 若alpha为，那就是最简单的最小二乘法ols
            model = HuberElasticNet(alpha=alpha, l1_ratio=l1_rate)

            # 在训练集上训练模型
            model.fit(X_train, y_train)

            # 在验证集上预测
            y_valid_pred = model.predict(X_valid)

            # 计算验证集上的R2分数
            r2 = r2_score(y_valid, y_valid_pred)

            # 如果当前参数组合的R2分数更高，则更新最优参数和模型
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'alpha': alpha, 'l1_ratio': l1_rate}
                best_model = model

        # 输出最优参数和验证集上的R2分数
        # print("最优参数:", best_params)
        # print("验证集的R2:", best_r2)

        # 使用最优模型预测测试集
        y_pred = best_model.predict(X_test)

        return y_pred

    @staticmethod
    def RandomForest(X_train, X_valid, y_train, y_valid, X_test):
        best_r2 = -np.inf
        best_model = None
        best_knots = None

        tree_nums = [150, 350, 550]  # 候选节点数量
        max_depths = [3, 6, 9]  # 候选正则化参数

        for num, depth in product(tree_nums, max_depths):
            # 初始化 Elastic Net 模型
            # 设置 alpha 为正则化强度，l1_ratio 为 Lasso的比例，为0就表示都是ridge，为1表示都lasso
            # 若alpha为，那就是最简单的最小二乘法ols
            model = RandomForestRegressor(n_estimators = num,
                                  max_depth = depth,
                                  random_state = 42,)

            # 在训练集上训练模型
            model.fit(X_train, y_train)

            # 在验证集上预测
            y_valid_pred = model.predict(X_valid)

            # 计算验证集上的R2分数
            r2 = r2_score(y_valid, y_valid_pred)

            # 如果当前参数组合的R2分数更高，则更新最优参数和模型
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'num': num, 'depth': depth}
                best_model = model

        # 输出最优参数和验证集上的R2分数
        # print("最优参数:", best_params)
        # print("验证集的R2:", best_r2)

        # 使用最优模型预测测试集
        y_pred = best_model.predict(X_test)

        return y_pred

    @staticmethod
    def GBRT(X_train, X_valid, y_train, y_valid, X_test):
        best_r2 = -np.inf
        best_model = None
        best_knots = None

        tree_nums = [150, 350, 550]  # 候选节点数量
        max_depths = [3, 6, 9]  # 候选正则化参数

        for num, depth in product(tree_nums, max_depths):
            # 初始化 Elastic Net 模型
            # 设置 alpha 为正则化强度，l1_ratio 为 Lasso的比例，为0就表示都是ridge，为1表示都lasso
            # 若alpha为，那就是最简单的最小二乘法ols
            model = GradientBoostingRegressor(
                n_estimators=num,
                learning_rate=0.03,
                max_depth=depth,
                min_samples_split=2,
                min_samples_leaf=1,
                loss='huber',
                random_state=42
            )

            # 在训练集上训练模型
            model.fit(X_train, y_train)

            # 在验证集上预测
            y_valid_pred = model.predict(X_valid)

            # 计算验证集上的R2分数
            r2 = r2_score(y_valid, y_valid_pred)

            # 如果当前参数组合的R2分数更高，则更新最优参数和模型
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'num': num, 'depth': depth}
                best_model = model

        # 输出最优参数和验证集上的R2分数
        # print("最优参数:", best_params)
        # print("验证集的R2:", best_r2)

        # 使用最优模型预测测试集
        y_pred = best_model.predict(X_test)

        return y_pred

    @staticmethod
    def XGB(X_train, X_valid, y_train, y_valid, X_test):
        best_r2 = -np.inf
        best_model = None
        best_knots = None

        # tree_nums = [150, 350, 550]  # 候选节点数量
        max_depths = [3, 6, 9]  
        lams = [3, 5, 7]  # 候选正则化参数

        for lam, depth in product(lams, max_depths):
            params = {
                'tree_method': 'gpu_hist',  # 使用 GPU 加速
                'predictor': 'gpu_predictor',  # 使用 GPU 进行预测
                'n_estimators': 1000,
                'learning_rate': 0.03,
                'max_depth': depth,
                'subsample': 0.7,
                'colsample_bytree': 0.6,
                'lambda': lam,
                'random_state': 42
            }
            model = XGBRegressor(**params)

            # 在训练集上训练模型
            model.fit(X_train, y_train)

            # 在验证集上预测
            y_valid_pred = model.predict(X_valid)

            # 计算验证集上的R2分数
            r2 = r2_score(y_valid, y_valid_pred)

            # 如果当前参数组合的R2分数更高，则更新最优参数和模型
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'lambda_l2': lam, 'depth': depth}
                best_model = model

        # 输出最优参数和验证集上的R2分数
        # print("最优参数:", best_params)
        # print("验证集的R2:", best_r2)

        # 使用最优模型预测测试集
        y_pred = best_model.predict(X_test)

        return y_pred
    
    @staticmethod
    def LGBM(X_train, X_valid, y_train, y_valid, X_test):
        best_r2 = -np.inf
        best_model = None
        best_knots = None

        # tree_nums = [150, 350, 550]  # 候选节点数量
        max_depths = [3, 6, 9]  
        lams = [3, 5, 7]  # 候选正则化参数

        for lam, depth in product(lams, max_depths):
            params={
                # 'device': 'gpu',
                'boosting_type': 'rf',
                'bagging_fraction': 0.8, # 样本随机采样 
                'bagging_freq': 1, # 每轮都进行bagging
                'n_estimators': 1000,
                'learning_rate': 0.03,
                'max_depth': depth,
                'min_child_samples': 1000,
                'lambda_l2': lam,
                'feature_fraction': 0.3,
                'random_state': 42
            }
            model = LGBMRegressor(**params)

            # 在训练集上训练模型
            model.fit(X_train, y_train)

            # 在验证集上预测
            y_valid_pred = model.predict(X_valid)

            # 计算验证集上的R2分数
            r2 = r2_score(y_valid, y_valid_pred)

            # 如果当前参数组合的R2分数更高，则更新最优参数和模型
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'lambda_l2': lam, 'depth': depth}
                best_model = model

        # 输出最优参数和验证集上的R2分数
        # print("最优参数:", best_params)
        # print("验证集的R2:", best_r2)

        # 使用最优模型预测测试集
        y_pred = best_model.predict(X_test)

        return y_pred

    @staticmethod
    def Neural_Net(X_train, X_valid, y_train, y_valid, X_test, num_layers=3):

        # 将数据存储为 TensorDataset，并创建 DataLoader
        X_train = torch.tensor(X_train, dtype=torch.float32).cuda()
        y_train = torch.tensor(y_train, dtype=torch.float32).cuda()
        X_valid = torch.tensor(X_valid, dtype=torch.float32).cuda()
        y_valid = torch.tensor(y_valid, dtype=torch.float32).cuda()
        X_test = torch.tensor(X_test, dtype=torch.float32).cuda()
        dataset = TensorDataset(X_train, y_train)
        batch_size = 1024
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        dataset = TensorDataset(X_valid, y_valid)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 初始化神经网络
        hidden_size = 128  # 每个隐藏层的神经元数量
        output_size = 1
        input_size = len(X_train[0])
        model = Net(input_size, output_size, hidden_size, num_layers).cuda()

        # 定义损失函数和优化器
        criterion = nn.MSELoss().cuda()
        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)

        patience = 10  # 如果验证集效果在 patience 个 epoch 内没有改善，就停止训练
        best_val_loss = float('inf')  # 记录最佳验证损失
        epochs_no_improve = 0  # 未改善的 epoch 计数
        best_model = None  # 最佳模型保存路径

        # 训练模型
        num_epochs = 50
        for epoch in range(num_epochs):
            # 训练模式
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()  # 清空梯度
                outputs = model(batch_X)  # 前向传播
                loss = criterion(outputs, batch_y)  # 计算损失
                loss.backward()  # 反向传播
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5.0)
                max_norm = 5.0  # 梯度的最大范数
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                # 打印梯度
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f'{name} gradient: {param.grad.norm()}')
                        # time.sleep(1)
                optimizer.step()  # 更新参数

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证模式
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 检查验证集损失是否改善
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0

                # 保存当前最佳模型
                best_model = model
                print(f"Best model saved at epoch {epoch + 1} with Val Loss: {val_loss:.4f}")
            else:
                epochs_no_improve += 1

            # 提前停止
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        y_pred = best_model(X_test)
        y_pred = y_pred.squeeze().detach().cpu().numpy()
        return y_pred
    
    @staticmethod
    def Transformer_model(X_train, X_valid, y_train, y_valid, X_test):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将数据存储为 TensorDataset，并创建 DataLoader
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_valid = torch.tensor(X_valid, dtype=torch.float32)
        y_valid = torch.tensor(y_valid, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_train, y_train)
        batch_size = 1024
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataset = TensorDataset(X_valid, y_valid)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = Transformer(input_dim=len(X_train[0]), 
                            output_dim=1, 
                            d_model=128, 
                            nhead=4, 
                            num_layers=2, 
                            dropout=0.35, 
                            device=device).to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)

        patience = 10  # 如果验证集效果在 patience 个 epoch 内没有改善，就停止训练
        best_val_loss = float('inf')  # 记录最佳验证损失
        epochs_no_improve = 0  # 未改善的 epoch 计数
        best_model = None  # 最佳模型保存路径

        # 训练模型
        num_epochs = 50
        for epoch in range(num_epochs):
            # 训练模式
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()  # 清空梯度
                outputs = model(batch_X)  # 前向传播
                loss = criterion(outputs, batch_y)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # 验证模式
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 检查验证集损失是否改善
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0

                # 保存当前最佳模型
                best_model = model
                print(f"Best model saved at epoch {epoch + 1} with Val Loss: {val_loss:.4f}")
            else:
                epochs_no_improve += 1

            # 提前停止
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # 将测试集封装为 DataLoader
        test_dataset = TensorDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch[0].to(device)  # 获取数据并移动到 GPU
                batch_pred = best_model(batch)
                predictions.extend(batch_pred.squeeze().detach().cpu().numpy())
                del batch, batch_pred
                gc.collect()
                torch.cuda.empty_cache()
        
        # y_pred = best_model(X_test)
        # print(y_pred.size())
        # y_pred = y_pred.squeeze().detach().cpu().numpy()
        return np.array(predictions)

# 训练集、验证集、测试集划分器，滚动划分
class TimeSeriesSplitter:
    def __init__(self, train_period=36, valid_period=6, test_period=6, roll_freq=6, freq='m'):
        """
        初始化TimeSeriesSplitter类

        参数:
        train_period (int): 训练集的周期长度（月数/日数）
        valid_period (int): 验证集的周期长度（月数/日数）
        test_period (int): 测试集的周期长度（月数/日数）
        roll_freq (int): 滚动频率（月数/日数）
        """
        self.train_period = train_period
        self.valid_period = valid_period
        self.test_period = test_period
        self.roll_freq = roll_freq
        self.freq = freq

    def split(self, data: pd.DataFrame):
        """
        划分数据集为训练集、验证集和测试集

        返回:
        list: 包含每次划分的训练集、验证集和测试集索引的列表
        """
        splits = []
        start_date = data['date'].min()
        end_date = data['date'].max()

        # 从数据集的起点开始向未来滚动
        while True:
            # 计算训练集的起始和结束日期
            train_start = start_date

            if self.freq == 'm':
                # 如果是datetime格式，要用pd.DateOffset
                # 如果是pd.Period格式，直接加减整数
                # train_end = train_start + pd.DateOffset(months=self.train_period)
                train_end = train_start + self.train_period

                # 计算验证集的起始和结束日期
                # valid_start = train_end + pd.DateOffset(months=1)
                # valid_end = valid_start + pd.DateOffset(months=self.valid_period)
                valid_start = train_end + 1
                valid_end = valid_start + self.valid_period

                # 计算测试集的起始和结束日期
                # test_start = valid_end + pd.DateOffset(months=1)
                # test_end = test_start + pd.DateOffset(months=self.test_period)
                test_start = valid_end + 1
                test_end = test_start + self.test_period
                
                # 向前滚动
                # start_date = start_date + pd.DateOffset(months=self.roll_freq)
                start_date = start_date + self.roll_freq
                
            elif self.freq == 'd':
                 # 如果是datetime格式，要用pd.DateOffset
                # 如果是pd.Period格式，直接加减整数
                train_end = train_start + pd.DateOffset(days=self.train_period)
                # train_end = train_start + self.train_period

                # 计算验证集的起始和结束日期
                valid_start = train_end + pd.DateOffset(days=20)
                valid_end = valid_start + pd.DateOffset(days=self.valid_period)

                # 计算测试集的起始和结束日期
                test_start = valid_end + pd.DateOffset(days=20)
                test_end = test_start + pd.DateOffset(days=self.test_period)
                
                # 向前滚动
                start_date = start_date + pd.DateOffset(days=self.roll_freq)

            # 获取索引，前闭后开
            train_idx = data[(data['date'] >= train_start) & (data['date'] < train_end)].index.tolist()
            valid_idx = data[(data['date'] >= valid_start) & (data['date'] < valid_end)].index.tolist()
            test_idx = data[(data['date'] >= test_start) & (data['date'] < test_end)].index.tolist()

            # 将索引添加到结果列表中
            splits.append([train_idx, valid_idx, test_idx])

            # 检查测试集的截止日期是否超出数据范围
            if test_end > end_date:
                break  # 如果超出范围，停止滚动

        return splits

def norm_and_fillna(data):
    mu = np.nanmean(data, axis=0)
    sigma = np.nanstd(data, axis=0)
    data = (data - mu) / sigma
    data = np.nan_to_num(data, 0)
    return data

# 主函数
def backtest_main(data, train_period=36, valid_period=6, test_period=6, roll_freq=6, freq='m', model='OLS', n_layer=2):
    """
    主函数：加载数据、划分数据、训练模型、分组回测

    参数:
    data_path (str): 数据文件路径
    train_period (int): 训练集周期长度（月数/日数）
    valid_period (int): 验证集周期长度（月数/日数）
    test_period (int): 测试集周期长度（月数/日数）
    roll_freq (int): 滚动频率（月数/日数）
    freq(str): 数据频率('m'月频,'d'日频)
    """
    print('Backtest Model: ', model)
    # 加载数据
    # data = pd.read_csv(data_path, parse_dates=['date'], index_col='date')

    # 数据预处理
    # processed_data = preprocess(data)

    # 划分数据集
    splitter = TimeSeriesSplitter(train_period, valid_period, test_period, roll_freq, freq)
    splits = splitter.split(data)

    # 训练模型并预测
    results = []
    for train_idx, valid_idx, test_idx in splits:
    # for train_idx, valid_idx, test_idx in splits:
        # 获取训练集、验证集、测试集
        X_train = data.loc[train_idx].drop(columns=['target', 'stock_code', 'date'])
        y_train = data.loc[train_idx]['target'].to_numpy()
        X_valid = data.loc[valid_idx].drop(columns=['target', 'stock_code', 'date'])
        y_valid = data.loc[valid_idx]['target'].to_numpy()
        X_test = data.loc[test_idx].drop(columns=['target', 'stock_code', 'date'])
        # print('X_train shape', X_train.shape)
        # print('X_valid shape', X_valid.shape)
        # print('X_test shape', X_test.shape)
        # 删除全为nan的列
        X_train = X_train.dropna(axis=1, how='all')
        X_valid = X_valid.dropna(axis=1, how='all')
        X_test = X_test.dropna(axis=1, how='all')
        # print('X_train shape', X_train.shape)
        # print('X_valid shape', X_valid.shape)
        # print('X_test shape', X_test.shape)
        cols = list(set(X_train.columns) & set(X_valid.columns) & set(X_test.columns))
        X_train = norm_and_fillna(X_train[cols].to_numpy())
        X_valid = norm_and_fillna(X_valid[cols].to_numpy())
        X_test = norm_and_fillna(X_test[cols].to_numpy())
        # print('X_train shape', X_train.shape)
        # print('X_valid shape', X_valid.shape)
        # print('X_test shape', X_test.shape)
        # print('y_train shape', y_train.shape)
        # print('y_valid shape', y_valid.shape)
        # print('y_test shape', data.loc[test_idx]['target'].to_numpy().shape)
        # 在这里选择要用的模型
        # 训练模型并预测
        if model == 'OLS':
            predictions = models.OLS_Huber(X_train, X_valid, y_train, y_valid, X_test)
        elif model == 'PCR':
            predictions = models.PCR(X_train, X_valid, y_train, y_valid, X_test)
        elif model == 'PLS':
            predictions = models.PLS(X_train, X_valid, y_train, y_valid, X_test)
        elif model == 'ENET':
            predictions = models.ElasticNet_Huber(X_train, X_valid, y_train, y_valid, X_test)
        elif model == 'RF':
            predictions = models.RandomForest(X_train, X_valid, y_train, y_valid, X_test)
        elif model == 'GBRT':
            predictions = models.GBRT(X_train, X_valid, y_train, y_valid, X_test)
        elif model == 'XGB':
            predictions = models.XGB(X_train, X_valid, y_train, y_valid, X_test)
        elif model == 'LGBM':
            predictions = models.LGBM(X_train, X_valid, y_train, y_valid, X_test)
        elif model == 'NN':
            predictions = models.Neural_Net(X_train, X_valid, y_train, y_valid, X_test, num_layers=n_layer)
        elif model == 'Transformer':
            predictions = models.Transformer_model(X_train, X_valid, y_train, y_valid, X_test)
        elif model == 'GLM':
        # GLM跑得会很慢
            predictions = models.GLM_with_Spline_GroupLasso(X_train, X_valid, y_train, y_valid, X_test)
        else:
            raise ValueError('Unknown model type.')
        # print(predictions)
        # print(predictions.shape)
        # 计算测试集R2
        r2 = r2_score(data.loc[test_idx]['target'], predictions)
        # print(f"Test R2: {r2}")

        # 将预测结果保存
        results.append(pd.DataFrame({
            'date': data.loc[test_idx]['date'],
            'stock_code': data.loc[test_idx]['stock_code'],
            'predicted_return': predictions
        }))

    # 合并所有预测结果
    results_df = pd.concat(results)

    return results_df

def my_cs_rank(arr_2d):
    rank_ = bn.nanrankdata(arr_2d, axis=1)
    no_nan_counts = np.sum(~np.isnan(arr_2d), axis=1, keepdims=True)
    pct_rank = rank_ / no_nan_counts
    return pct_rank


if __name__ == "__main__":
    dsr = DailySharryReader()

    # 读数据
    factor_value_dir = 'data/'
    index_ = dsr.date_idx[(dsr.date_idx >= 20150101) & (dsr.date_idx <= 20241231)]
    columns_ = dsr.symbol_idx
    data_arr = np.zeros((len(os.listdir(factor_value_dir)), len(index_), len(columns_)))
    fa_list = []
    factors = os.listdir(factor_value_dir)
    for i, file in tqdm(enumerate(factors), total=len(factors)):
        df = pd.read_pickle(os.path.join(factor_value_dir, file))
        fa_list.append(file.split('.')[0])
        df = df.reindex(index=index_, columns=columns_)
        df = df.replace(0.0, np.nan)
        
        # 如果要转换成月频数据，my_cs_rank()要注释掉
        data_arr[i] = np.array(df)
        # data_arr[i] = my_cs_rank(np.array(df))    # (86, 2000+, 5000+)
        # data_arr[i] = np.where(np.isnan(data_arr[i]), 0.5, data_arr[i])
        
    n_fa = len(fa_list)

    adj_close = dsr.get_field_data('adj_factor') * dsr.get_field_data('close')
    stock_ret = adj_close.shift(-10) / adj_close - 1
    label = stock_ret.reindex(index=index_, columns=columns_)  # t+10 / t - 1

    daily_data = pd.DataFrame(data_arr.reshape(n_fa, -1).T, columns=fa_list)
    daily_data['stock_code'] = columns_ * len(index_)
    daily_data['date'] = index_.repeat(len(columns_))
    daily_data['target'] = label.values.reshape(1, -1).T
    daily_data = daily_data[['stock_code', 'date', 'target'] + fa_list]
    daily_data = daily_data.dropna(subset='target')
    daily_data['date'] = pd.to_datetime(daily_data['date'].astype(str), format='%Y%m%d')
    daily_data = daily_data.sort_values(by=['stock_code', 'date'])
    daily_data = daily_data.reset_index(drop=True)
    
    monthly_data = transform_to_monthly(daily_data)
    # del daily_data

    # coverage = 1 - daily_data.isna().sum(axis=0) / len(daily_data)
    # coverage.to_excel('/nas197/user_home/guozhaopeng/aa_results/factor_coverage.xlsx', index=True)

    for m in ['OLS', 'PCR', 'PLS', 'ENET', 'RF', 'GLM', 'GBRT', 'LGBM', 'XGB', 'Transformer', 'NN']:
        results = backtest_main(monthly_data.copy(), 12*3, 6, 6, 6, 'm', m) # 月频
        results.to_pickle(f'results/monthly_preds_{m}.pkl')
        
        results = backtest_main(daily_data.copy(), 30*12*3, 30*6, 30*6, 30*6, 'd', 'NN', n_layer) # 日频
        results.to_pickle(f'results/preds_{m}.pkl')
