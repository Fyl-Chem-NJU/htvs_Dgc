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
import io

def load_data(data_path):
    """加载和预处理数据"""
    basic_data = pd.read_csv('../data.csv')
    basic_data.set_index('name', inplace=True)
    
    ox = pd.read_csv(data_path)
    ox.set_index('name_index', inplace=True)
    ox.index.name = 'name'
    
    # 特征工程
    X = pd.concat([basic_data.loc[[i]] for i in ox.index])
    y = ox
    ts_cols = [col for col in X.columns if col.startswith('ts_')]
    X = X[ts_cols]
    
    # 标准化
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), 
                    index=y.index, 
                    columns=ts_cols)
    
    # 去除高相关特征
    corr = X.corr().abs()
    keep = []
    for i in range(len(corr.columns)):
        above = corr.iloc[:i,i]
        if keep: above = above[keep]
        if all(above < 0.95):
            keep.append(corr.columns[i])
    
    return X[keep], y
def load_lig_data(data_path):
    """加载和预处理数据"""
    basic_data = pd.read_csv('../lig_xtb_level_data.csv')
    basic_data.set_index('name', inplace=True)
    
    ox = pd.read_csv(data_path)
    ox.set_index('name_index', inplace=True)
    ox.index.name = 'name'
    
    # 特征工程
    X = pd.concat([basic_data.loc[[i]] for i in ox.index])
    y = ox
    lig_cols = [col for col in X.columns if col.startswith('lig_')]
    X = X[lig_cols]
    
    # 标准化
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), 
                    index=y.index, 
                    columns=lig_cols)
    
    # 去除高相关特征
    corr = X.corr().abs()
    keep = []
    for i in range(len(corr.columns)):
        above = corr.iloc[:i,i]
        if keep: above = above[keep]
        if all(above < 0.95):
            keep.append(corr.columns[i])
    
    return X[keep], y





def plot_results(y_train, y_test, y_pred_train, y_pred_test, model_name,r2_ave_scores, r2_cv_scores, mae_cv_scores, results_path):
    """绘制结果图表并自动保存"""
    plt.figure(figsize=(12, 6))
    
    # 计算指标
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
#     scores = np.mean(scores)
#     scores_std = np.std(scores)
    
    # 预测结果图
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_train.values.ravel(), y=y_pred_train.ravel(), 
                   label='Train', alpha=0.6)
    sns.scatterplot(x=y_test.values.ravel(), y=y_pred_test.ravel(), 
                   label='Test', alpha=0.6)
    plt.plot([5,38], [5,38], 'k--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    
    # 在图中添加指标文本（右下角版本）
    metrics_text = (f"Train R²: {train_r2:.4f}\n"
                f"Test R²: {test_r2:.4f}\n"
                f"CV MAE: {-mae_cv_scores.mean():.4f}")
    plt.text(0.95, 0.05, metrics_text,  # x=0.95,y=0.05 就是右下角啦~
            transform=plt.gca().transAxes,
            verticalalignment='bottom',  # 对齐方式也要调整哦~
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    
    plt.title(f'{model_name} Prediction')

    # 残差图
    # plt.subplot(1, 2, 2)
    # residuals = y_test.values.ravel() - y_pred_test.ravel()
    # sns.histplot(residuals, kde=True)
    # plt.xlabel('Residuals')
    # plt.title('Residual Distribution')
    
    plt.tight_layout()
    
    # 自动保存图表（不显示）
    plt.savefig(f'{results_path}/{model_name}_results.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形避免内存泄漏
    
    # 打印指标
    print(f"\n{model_name} 性能指标:")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"AVG R2: {r2_ave_scores:.4f}")
    print(f"CV R2: {r2_cv_scores.mean():.4f}") 
    print(f"CV MAE: {mae_cv_scores.mean():.4f}")


def plot_results_dis(y_train, y_test, y_pred_train, y_pred_test, model_name,r2_ave_scores, r2_cv_scores, mae_cv_scores, results_path):
    """绘制结果图表并自动保存"""
    plt.figure(figsize=(12, 6))
    
    # 计算指标
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
#     scores = np.mean(scores)
#     scores_std = np.std(scores)
    
    # 预测结果图
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_train.values.ravel(), y=y_pred_train.ravel(), 
                   label='Train', alpha=0.6)
    sns.scatterplot(x=y_test.values.ravel(), y=y_pred_test.ravel(), 
                   label='Test', alpha=0.6)
    plt.plot([2.75,3.00], [2.75,3.00], 'k--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    
    # 在图中添加指标文本（右下角版本）
    metrics_text = (f"Train R²: {train_r2:.4f}\n"
                f"Test R²: {test_r2:.4f}\n"
                f"CV MAE: {-mae_cv_scores.mean():.4f}")
    plt.text(0.95, 0.05, metrics_text,  # x=0.95,y=0.05 就是右下角啦~
            transform=plt.gca().transAxes,
            verticalalignment='bottom',  # 对齐方式也要调整哦~
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    
    plt.title(f'{model_name} Prediction')

    # 残差图
    # plt.subplot(1, 2, 2)
    # residuals = y_test.values.ravel() - y_pred_test.ravel()
    # sns.histplot(residuals, kde=True)
    # plt.xlabel('Residuals')
    # plt.title('Residual Distribution')
    
    plt.tight_layout()
    
    # 自动保存图表（不显示）
    plt.savefig(f'{results_path}/{model_name}_results.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形避免内存泄漏
    
    # 打印指标
    print(f"\n{model_name} 性能指标:")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"AVG R2: {r2_ave_scores:.4f}")
    print(f"CV R2: {r2_cv_scores.mean():.4f}") 
    print(f"CV MAE: {mae_cv_scores.mean():.4f}")