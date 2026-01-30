import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from bayes_opt import BayesianOptimization, UtilityFunction
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from contextlib import redirect_stdout
import io, os, sys
sys.path.append('.')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (
    LeaveOneOut,
    cross_val_score,
    train_test_split,
    cross_val_predict,
)

def plot_final_evaluation(y_train_full, y_test, y_pred_train, y_pred_test, model_name, results_path, cv_r2, loo_r2, loo_mae):
    """
    Generates and saves a comprehensive evaluation plot for the final model.

    This function creates a two-panel figure:
    1. A scatter plot of True vs. Predicted values for both development and hold-out sets.
    2. A residual plot for the hold-out test set to diagnose model bias.
    """
    
    # --- 1. Calculate Performance Metrics ---
    train_mae = mean_absolute_error(y_train_full, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train_full, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # --- 2. Create Figure and Subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(f'Final Evaluation: {model_name}', fontsize=18, y=1.02)

    # --- 3. Panel 1: True vs. Predicted Scatter Plot ---
    sns.scatterplot(x=y_train_full.values, y=y_pred_train, ax=axes[0], 
                    label=f'Train (Dev Set)', alpha=0.7, ec='k', s=50) # ec is edge color
    sns.scatterplot(x=y_test.values, y=y_pred_test, ax=axes[0],
                    label=f'Hold-out Test Set', alpha=0.8, ec='k', s=80, marker='^')
    
    # Dynamically determine plot limits to make the y=x line fit perfectly
    all_values = np.concatenate([y_train_full, y_test, y_pred_train, y_pred_test])
    min_val, max_val = all_values.min(), all_values.max()
    buffer = (max_val - min_val) * 0.05  # Add 5% buffer
    axes[0].plot([min_val - buffer, max_val + buffer], 
                 [min_val - buffer, max_val + buffer], 
                 'r--', linewidth=2, label='Ideal (y=x)')
    
    axes[0].set_xlabel('True Values (kcal/mol)', fontsize=12)
    axes[0].set_ylabel('Predicted Values (kcal/mol)', fontsize=12)
    axes[0].set_title('Prediction Performance', fontsize=14)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_aspect('equal', 'box') # Make the plot square

    # Add metrics text box
    metrics_text = (f"Train R² (Full Dev Set): {train_r2:.4f}\n"
                    f"Hold-out Test R²: {test_r2:.4f}\n"
                    f"Train MAE (Full Dev Set): {train_mae:.4f}\n"
                    f"Hold-out Test MAE: {test_mae:.4f}\n"
                    f"5-Fold CV R²: {cv_r2:.4f}\n"
                    f"LOO CV R²: {loo_r2:.4f}\n"
                    f"LOO CV MAE: {loo_mae:.4f}"
                    )
    axes[0].text(0.95, 0.05, metrics_text,
                 transform=axes[0].transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    # --- 4. Panel 2: Residual Plot (on Test Set) ---
    residuals = y_test.values - y_pred_test
    sns.scatterplot(x=y_pred_test, y=residuals, ax=axes[1], alpha=0.7, ec='k', s=60)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2) # Line at zero error
    
    axes[1].set_xlabel('Predicted Values (kcal/mol)', fontsize=12)
    axes[1].set_ylabel('Residuals (True - Predicted)', fontsize=12)
    axes[1].set_title('Residual Analysis (Hold-out Test Set)', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # --- 5. Finalize and Save ---
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
    
    save_path = f'{results_path}/{model_name}_final_evaluation.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    
    print(f"✅ Final evaluation plot for {model_name} saved to {save_path}")
    
    # Print final metrics clearly to the console
    print(f"\n--- Final Performance Report for {model_name} ---")
    print(f"  Development Set (Train):")
    print(f"    - R² Score: {train_r2:.4f}")
    print(f"    - MAE:      {train_mae:.4f}")
    print(f"  Hold-out Test Set:")
    print(f"    - R² Score: {test_r2:.4f}")
    print(f"    - MAE:      {test_mae:.4f}")
    print(f"  5 Fold Cross-Validation:")
    print(f"    -5 CV R² Score: {cv_r2:.4f}")
    print(f"  Leave-One-Out Cross-Validation:")
    print(f"    -LOO R² Score: {loo_r2:.4f}")
    print(f"    -LOO MAE:      {loo_mae:.4f}")
    print("-------------------------------------------------")





def bayesian_optimization(model_name,results_path, model_func, param_space, init_points=10, n_iter=1000, early_stopping_rounds=100):
    """完全兼容最新版 bayesian-optimization 的带早停贝叶斯优化"""
    
    # 初始化优化器
    optimizer = BayesianOptimization(
        f=model_func,
        pbounds=param_space,
        random_state=42,
        verbose=2  # 设置为2可以看到更多调试信息
    )
    
    # 设置高斯过程参数（可选）
    optimizer.set_gp_params(
        alpha=1e-5,
        n_restarts_optimizer=10
    )
    
    # 创建获取函数对象
    utility = UtilityFunction(
        kind="ucb",
        kappa=2.5,
        xi=0.0
    )
    
    # 早停相关变量
    best_target = -np.inf
    no_improvement_count = 0
    target_history = []
    
    # 早停回调函数
    def _early_stop_callback(res):
        nonlocal best_target, no_improvement_count, target_history
        
        current_target = res["target"]
        target_history.append(current_target)
        
        if current_target > best_target:
            best_target = current_target
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if early_stopping_rounds and no_improvement_count >= early_stopping_rounds:
            print(f"⚠️ 早停触发：连续 {early_stopping_rounds} 次迭代未提升")
            return True  # 返回True停止优化
        
        return False  # 返回False继续优化
    
    # 执行初始探索
    optimizer.maximize(init_points=init_points, n_iter=0)
    # 执行优化
    for _ in range(n_iter):
        # 获取下一个点
        next_point = optimizer.suggest(utility)
        
        # 评估目标函数
        target = model_func(**next_point)
        
        # 注册结果
        optimizer.register(params=next_point, target=target)
        
        # 检查早停
        if _early_stop_callback({"target": target}):
            break
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(target_history)), target_history, 'b-', label='Best Score')
    plt.xlabel('Iteration')
    plt.ylabel('Target Value')
    plt.title(f'Bayesian Optimization Convergence (Stopped at {len(target_history)} iters)')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f'{results_path}/{model_name}_bayesian.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"优化完成，共执行 {len(target_history)} 次迭代（最佳得分: {best_target:.4f}）")
    return optimizer.max



# def bayesian_score(number, X, y, model):
#     R2_test_list = []
#     for i in range(1, number):
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=i)
#         model = model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         #############
#         y_pred = pd.Series(y_pred.flatten(), index=y_test.index)
#         ###########
#         # y_pred = pd.Series(y_pred, index=y_test.index)
#         R2_test = r2_score(y_test, y_pred)
#         R2_test_list.append(R2_test)
#     return np.mean(R2_test_list)

def bayesian_score_10_fold(model, X, y, cv_folds=5): # 参数顺序调整，名称更清晰
    cv_scores = cross_val_score(
            model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error') # 使用参数
    return np.mean(cv_scores)




class ModelTrainer:
    def __init__(self, X, y, results_path):
        self.X_full = X  # 全部特征数据
        self.y_full = y  # 全部目标数据
        self.results_path = results_path
        
        # 关键：在这里进行一次性的、最终的训练/测试集分割
        # X_train_full 将用于所有调参和训练
        # X_test 将仅用于最终评估
        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42)
        # if self.results_path is None:
        #     # 创建该目录
        #     os.makedirs(self.results_path, exist_ok=True)
    def evaluate_final_model(self, final_model, name):
        """
        Trains the final model on the full development set and evaluates it 
        on the unseen hold-out test set.
        """
        print(f"\n--- Evaluating final '{name}' model ---")
        
        # 1. Train on the full development set
        final_model.fit(self.X_train_full, self.y_train_full)
        
        # 2. Predict on both sets
        train_pred = final_model.predict(self.X_train_full)
        test_pred = final_model.predict(self.X_test)

        cv_r2 = float(
        cross_val_score(
            final_model,
            self.X_train_full,
            self.y_train_full,
            cv=5,
            scoring="r2",
        ).mean()
        )
        loo_model = cross_val_predict(
            final_model,
            self.X_full,
            self.y_full,
            cv=LeaveOneOut(),
        )
        loo_r2 = float(r2_score(self.y_full, loo_model))
        loo_mae = float(mean_absolute_error(self.y_full, loo_model))
        # 3. Use the new plotting function to visualize and report results
        plot_final_evaluation(
            self.y_train_full, 
            self.y_test, 
            train_pred, 
            test_pred, 
            name, 
            self.results_path,
            cv_r2,
            loo_r2,
            loo_mae
        )


    def train_linear_models(self):

        """训练线性模型"""
        print("\n训练线性模型...✨")
        models = {
            'Linear': LinearRegression(),
            'Lasso': LassoCV(cv=5, random_state=42),
            'Ridge': RidgeCV(cv=5),
            'ElasticNet': ElasticNetCV(cv=5, random_state=42)
        }
        
        for name, model in models.items():
            model.fit(self.X_train_full, self.y_train_full)
            self.evaluate_final_model(model, name)

    def train_decision_tree(self):
        print('\n训练决策树模型...')
        dt_space = {
            'max_depth': (3, 10),
            'min_samples_split': (5, 25),
            'min_samples_leaf': (5, 15),
            'max_features': (0.5, 1.0),
            'ccp_alpha': (0.001, 0.05)
        }
        
        def dt_eval(max_depth, min_samples_split, min_samples_leaf, max_features, ccp_alpha):
            params = {
                'max_depth': int(max_depth),
                'min_samples_split': int(min_samples_split),
                'min_samples_leaf': int(min_samples_leaf),
                'max_features': max_features,
                'ccp_alpha': ccp_alpha,  # 添加ccp_alpha参数
            }
            
            model = DecisionTreeRegressor(**params, random_state=42)
            return bayesian_score_10_fold(model, self.X_train_full, self.y_train_full)
        
        with redirect_stdout(io.StringIO()):
            best_dt = bayesian_optimization('Decision_Tree', self.results_path, dt_eval, dt_space)
        print(f"决策树最优参数是: {best_dt['params']} ")
        
        dt_model = DecisionTreeRegressor(
            max_depth=int(best_dt['params']['max_depth']),
            min_samples_split=int(best_dt['params']['min_samples_split']),
            min_samples_leaf=int(best_dt['params']['min_samples_leaf']),
            max_features=best_dt['params']['max_features'],
            ccp_alpha=best_dt['params']['ccp_alpha'],  # 添加ccp_alpha参数
            random_state=42
        )
        self.evaluate_final_model(dt_model, 'Decision_Tree')


    def train_svr(self):
        """训练支持向量机模型"""
        print('\n训练支持向量机模型...')
        svr_space = {
            'C': (0.1, 20),
            'gamma': (0.001, 5),
            'epsilon': (0.05, 1.5),
            'kernel_param': (0, 3)  # 连续参数代替离散选择
        }
        
        def svr_eval(C, gamma, epsilon, kernel_param):
            # 将连续参数映射到核函数
            if kernel_param < 1:
                kernel = 'linear'
            elif kernel_param < 2:
                kernel = 'poly'
            else:
                kernel = 'rbf'
            
            params = {
                'kernel': kernel,
                'C': C,
                'gamma': gamma if kernel != 'linear' else 'auto',
                'epsilon': epsilon,
            }
            
            model = SVR(**params)
            return bayesian_score_10_fold(model, self.X_train_full, self.y_train_full)
        
        with redirect_stdout(io.StringIO()):
            best_svr = bayesian_optimization('SVR', self.results_path, svr_eval, svr_space)
        
        # 需要从kernel_param解析出实际的kernel类型
        best_kernel_param = best_svr['params']['kernel_param']
        if best_kernel_param < 1:
            best_kernel = 'linear'
        elif best_kernel_param < 2:
            best_kernel = 'poly'
        else:
            best_kernel = 'rbf'
        
        print(f"支持向量机最优参数是: C={best_svr['params']['C']}, "
            f"gamma={best_svr['params']['gamma']}, "
            f"epsilon={best_svr['params']['epsilon']}, "
            f"kernel={best_kernel} (原始参数值: {best_kernel_param})")
        
        svr_model = SVR(
            kernel=best_kernel,  # 使用解析后的kernel类型
            C=best_svr['params']['C'],
            gamma=best_svr['params']['gamma'] if best_kernel != 'linear' else 'auto',
            epsilon=best_svr['params']['epsilon'],
        )
        self.evaluate_final_model(svr_model, 'SVR')


    def train_xgboost(self):
        """训练XGBoost模型"""
        print('\n训练XGBoost模型...')
        xgb_space = {
            'eta': (0.01, 1.3),
            'gamma': (0, 10),
            'reg_alpha': (0, 10),
            'reg_lambda': (0, 10),
            'max_depth': (3, 15),
            'min_child_weight': (1, 10),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.5, 1),
            'n_estimators': (50, 100)
        }
        def xgb_eval(eta, gamma, reg_alpha, reg_lambda, max_depth, min_child_weight, subsample, colsample_bytree, n_estimators):
            params = {
                'eta': eta,
                'gamma': gamma,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'max_depth': int(max_depth),
                'min_child_weight': int(min_child_weight),
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'n_estimators': int(n_estimators)
            }

            model = xgb.XGBRegressor(**params, random_state=42)

            # return bayesian_score(51, self.X, self.y, model)
            return bayesian_score_10_fold(model, self.X_train_full, self.y_train_full)
        with redirect_stdout(io.StringIO()):
            best_xgb = bayesian_optimization('XGBoost',self.results_path, xgb_eval, xgb_space)
        print(f"XGBoost最优参数是: {best_xgb['params']} ")

        xgb_model = xgb.XGBRegressor(
            eta=best_xgb['params']['eta'],
            gamma=best_xgb['params']['gamma'],
            reg_alpha=best_xgb['params']['reg_alpha'],
            reg_lambda=best_xgb['params']['reg_lambda'],
            max_depth=int(best_xgb['params']['max_depth']),
            min_child_weight=int(best_xgb['params']['min_child_weight']),
            subsample=best_xgb['params']['subsample'],
            colsample_bytree=best_xgb['params']['colsample_bytree'],
            n_estimators=int(best_xgb['params']['n_estimators']),
            random_state=42
        )
        self.evaluate_final_model(xgb_model, 'XGBoost')
        
    def train_random_forest(self):
        """训练随机森林模型"""
        print('\n训练随机森林模型...')
        n_features = self.X_train_full.shape[1]

        # 回归问题推荐参数空间
        rf_space = {
            'n_estimators': (100, 1000),  # 回归需要更多树
            'max_depth': (5, 50),         # 更深以捕捉连续值关系
            'min_samples_split': (2, 20), # 稍大防止过拟合
            'min_samples_leaf': (1, 10),  # 重要！回归通常需要>1
            'max_features': (1, max(1, int(n_features/3))),  # 回归常用n/3
            'min_impurity_decrease': (0.0, 0.5)  # 更大范围
        }
        def rf_eval(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, min_impurity_decrease):
            params = {
                'n_estimators': int(n_estimators),
                'max_depth': None if max_depth > 45 else int(max_depth),
                'min_samples_split': int(min_samples_split),
                'min_samples_leaf': int(min_samples_leaf),
                'max_features': 'auto' if max_features >= n_features else int(max_features),
                'min_impurity_decrease': min_impurity_decrease
            }

            model = RandomForestRegressor(**params, random_state=42)

            # return bayesian_score(51, self.X, self.y, model)
            return bayesian_score_10_fold(model, self.X_train_full, self.y_train_full)
        with redirect_stdout(io.StringIO()):
            best_rf = bayesian_optimization('Random_Forest',self.results_path, rf_eval, rf_space)
        print(f"随机森林最优参数是: {best_rf['params']} ")

        rf_model = RandomForestRegressor(
            n_estimators=int(best_rf['params']['n_estimators']),
            max_depth=None if best_rf['params']['max_depth'] > 45 else int(best_rf['params']['max_depth']),
            min_samples_split=int(best_rf['params']['min_samples_split']),
            min_samples_leaf=int(best_rf['params']['min_samples_leaf']),
            max_features='auto' if best_rf['params']['max_features'] >= n_features else int(best_rf['params']['max_features']),
            min_impurity_decrease=best_rf['params']['min_impurity_decrease'],
            random_state=42
        )
        self.evaluate_final_model(rf_model, 'Random_Forest')

    def train_knn(self):
        """训练K近邻模型"""
        print('\n训练K近邻模型...')
        knn_space = {
            'n_neighbors': (5, 30),
            'weights': (0, 1),
            'p': (1, 2)  # 这里发现你参数空间定义了p但没用到呢~
        }
        
        def knn_eval(n_neighbors, weights, p):
            params = {
                'n_neighbors': int(n_neighbors),
                'weights': 'uniform' if weights < 0.5 else 'distance',
                'p': int(p)  # 补上用上p参数
            }

            model = KNeighborsRegressor(**params)
            return bayesian_score_10_fold(model, self.X_train_full, self.y_train_full)
        
        with redirect_stdout(io.StringIO()):
            best_knn = bayesian_optimization('KNN', self.results_path, knn_eval, knn_space)
        
        print(f"K近邻最优参数是: {best_knn['params']} ")

        knn_model = KNeighborsRegressor(
            n_neighbors=int(best_knn['params']['n_neighbors']),
            weights='uniform' if best_knn['params']['weights'] < 0.5 else 'distance',
            p=int(best_knn['params']['p']),
        )
        self.evaluate_final_model(knn_model, 'KNN')


