# -*- coding: utf-8 -*-
"""
特征标准化器模块
"""

import numpy as np
from typing import List, Tuple


class FeatureScalerG:
    """状态向量标准化器 (用于 g_t)"""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.valid_mask_ = None
        self.n_features_in_ = None
        self.n_features_out_ = None
    
    def fit(self, G: np.ndarray, feature_names: List[str]) -> Tuple['FeatureScalerG', List[str], List[str]]:
        """
        拟合标准化器，处理 std=0 的特征
        
        Returns:
            self, valid_feature_names, dropped_feature_names
        """
        self.n_features_in_ = G.shape[1]
        
        self.mean_ = G.mean(axis=0)
        self.std_ = G.std(axis=0)
        
        # 找出 std=0 的特征
        self.valid_mask_ = self.std_ > 1e-8
        dropped_indices = np.where(~self.valid_mask_)[0]
        dropped_names = [feature_names[i] for i in dropped_indices]
        valid_names = [feature_names[i] for i in range(len(feature_names)) if self.valid_mask_[i]]
        
        # 只保留有效特征的统计量
        self.mean_ = self.mean_[self.valid_mask_]
        self.std_ = self.std_[self.valid_mask_]
        self.n_features_out_ = len(self.mean_)
        
        return self, valid_names, dropped_names
    
    def transform(self, G: np.ndarray) -> np.ndarray:
        """标准化，仅保留有效特征"""
        G_valid = G[:, self.valid_mask_]
        return (G_valid - self.mean_) / self.std_
    
    def fit_transform(self, G: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
        self.fit(G, feature_names)
        return self.transform(G), [feature_names[i] for i in range(len(feature_names)) if self.valid_mask_[i]], \
               [feature_names[i] for i in range(len(feature_names)) if not self.valid_mask_[i]]
    
    def get_stats(self) -> dict:
        return {
            'n_features_in': int(self.n_features_in_),
            'n_features_out': int(self.n_features_out_),
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist()
        }


class FeatureScaler:
    """
    特征标准化器 (Z-score)
    对形状为 (N, Lx, F) 的输入，在 N*Lx 维度上汇总计算均值和标准差，
    对每个 feature 独立标准化
    """
    
    def __init__(self):
        self.mean_ = None  # (F,)
        self.std_ = None   # (F,)
        self.n_features_ = None
    
    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        """
        在训练集上拟合
        
        Args:
            X: (N, Lx, F) 输入特征
        """
        N, Lx, F = X.shape
        self.n_features_ = F
        
        # 将 (N, Lx, F) reshape 为 (N*Lx, F) 进行统计
        X_flat = X.reshape(-1, F)
        
        self.mean_ = X_flat.mean(axis=0)  # (F,)
        self.std_ = X_flat.std(axis=0)    # (F,)
        
        # 防止除零
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        标准化
        
        Args:
            X: (N, Lx, F) 或 (Lx, F) 输入特征
        
        Returns:
            标准化后的 X
        """
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并标准化"""
        self.fit(X)
        return self.transform(X)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'n_features': self.n_features_,
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist()
        }
