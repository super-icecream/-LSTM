"""
训练诊断模块
功能：收集和保存详细的训练诊断信息，用于调试和问题定位
作者：DLFE-LSTM-WSI Team
日期：2025-11-30
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TrainingDiagnostics:
    """
    训练诊断收集器
    
    收集以下类别的诊断信息：
    1. 数据质量：特征和目标值统计
    2. 模型输出：预测值分布和范围
    3. 梯度信息：各层梯度统计
    4. 权重统计：模型权重变化
    5. 训练动态：损失、学习率、耗时等
    """
    
    def __init__(self, save_dir: Optional[Path] = None, collect_interval: int = 5):
        """
        参数：
            save_dir: 诊断日志保存目录
            collect_interval: 收集间隔（每N个epoch收集一次详细信息）
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.collect_interval = collect_interval
        
        # 诊断数据存储
        self.diagnostics: Dict[str, List[Dict[str, Any]]] = {}
        
        # 初始时间
        self.start_time = datetime.now()
        
    def init_weather(self, weather: str):
        """初始化天气类型的诊断存储"""
        if weather not in self.diagnostics:
            self.diagnostics[weather] = []
    
    def collect_data_stats(self, 
                          features: torch.Tensor, 
                          targets: torch.Tensor,
                          weather: str) -> Dict[str, float]:
        """
        收集数据统计信息
        
        监控环节：数据预处理是否正确
        """
        with torch.no_grad():
            feat_np = features.cpu().numpy() if features.is_cuda else features.numpy()
            targ_np = targets.cpu().numpy() if targets.is_cuda else targets.numpy()
            
            stats = {
                # 特征统计
                "feature_min": float(np.nanmin(feat_np)),
                "feature_max": float(np.nanmax(feat_np)),
                "feature_mean": float(np.nanmean(feat_np)),
                "feature_std": float(np.nanstd(feat_np)),
                "feature_nan_count": int(np.isnan(feat_np).sum()),
                "feature_inf_count": int(np.isinf(feat_np).sum()),
                
                # 目标值统计
                "target_min": float(np.nanmin(targ_np)),
                "target_max": float(np.nanmax(targ_np)),
                "target_mean": float(np.nanmean(targ_np)),
                "target_std": float(np.nanstd(targ_np)),
                "target_nan_count": int(np.isnan(targ_np).sum()),
                "zero_target_ratio": float((targ_np == 0).sum() / max(1, targ_np.size)),
            }
            
        return stats
    
    def collect_prediction_stats(self, 
                                 predictions: torch.Tensor, 
                                 targets: torch.Tensor) -> Dict[str, float]:
        """
        收集模型输出统计信息
        
        监控环节：模型是否正常学习（输出常数=问题）
        """
        with torch.no_grad():
            pred_np = predictions.cpu().numpy() if predictions.is_cuda else predictions.numpy()
            targ_np = targets.cpu().numpy() if targets.is_cuda else targets.numpy()
            
            pred_flat = pred_np.flatten()
            
            stats = {
                # 预测值统计
                "pred_min": float(np.min(pred_flat)),
                "pred_max": float(np.max(pred_flat)),
                "pred_mean": float(np.mean(pred_flat)),
                "pred_std": float(np.std(pred_flat)),
                "pred_variance": float(np.var(pred_flat)),
                
                # 预测值范围检查
                "pred_negative_ratio": float((pred_flat < 0).sum() / max(1, len(pred_flat))),
                "pred_over_one_ratio": float((pred_flat > 1).sum() / max(1, len(pred_flat))),
                "pred_out_of_range_ratio": float(((pred_flat < 0) | (pred_flat > 1)).sum() / max(1, len(pred_flat))),
                
                # 预测与目标的差异
                "residual_mean": float(np.mean(pred_flat - targ_np.flatten())),
                "residual_std": float(np.std(pred_flat - targ_np.flatten())),
            }
            
            # 判断是否输出常数（方差接近0）
            stats["is_constant_output"] = stats["pred_variance"] < 1e-6
            
        return stats
    
    def collect_gradient_stats(self, model: nn.Module) -> Dict[str, Any]:
        """
        收集各层梯度统计信息
        
        监控环节：梯度消失/爆炸，定位问题层
        """
        grad_stats = {
            "per_layer": {},
            "total_grad_norm": 0.0,
            "zero_grad_param_count": 0,
            "total_param_count": 0,
        }
        
        total_norm = 0.0
        zero_count = 0
        total_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                grad_norm = grad.norm().item()
                total_norm += grad_norm ** 2
                
                # 简化层名
                layer_name = name.replace(".weight", "_w").replace(".bias", "_b")
                
                grad_stats["per_layer"][layer_name] = {
                    "norm": float(grad_norm),
                    "mean": float(grad.mean().item()),
                    "std": float(grad.std().item()) if grad.numel() > 1 else 0.0,
                    "max": float(grad.abs().max().item()),
                    "zero_ratio": float((grad.abs() < 1e-10).sum().item() / max(1, grad.numel())),
                }
                
                # 统计梯度为0的参数
                if grad_norm < 1e-10:
                    zero_count += param.numel()
                total_count += param.numel()
        
        grad_stats["total_grad_norm"] = float(np.sqrt(total_norm))
        grad_stats["zero_grad_param_count"] = zero_count
        grad_stats["total_param_count"] = total_count
        grad_stats["zero_grad_ratio"] = float(zero_count / max(1, total_count))
        
        return grad_stats
    
    def collect_weight_stats(self, model: nn.Module) -> Dict[str, Any]:
        """
        收集模型权重统计信息
        
        监控环节：权重是否更新，是否稳定
        """
        weight_stats = {"per_layer": {}}
        
        for name, param in model.named_parameters():
            layer_name = name.replace(".weight", "_w").replace(".bias", "_b")
            
            with torch.no_grad():
                data = param.detach()
                weight_stats["per_layer"][layer_name] = {
                    "mean": float(data.mean().item()),
                    "std": float(data.std().item()) if data.numel() > 1 else 0.0,
                    "min": float(data.min().item()),
                    "max": float(data.max().item()),
                    "norm": float(data.norm().item()),
                }
        
        return weight_stats
    
    def collect_epoch_diagnostics(self,
                                  epoch: int,
                                  weather: str,
                                  model: nn.Module,
                                  train_loader,
                                  loss: float,
                                  lr: float,
                                  epoch_time: float,
                                  collect_full: bool = False) -> Dict[str, Any]:
        """
        收集单个epoch的诊断信息
        
        参数：
            epoch: 当前epoch
            weather: 天气类型
            model: 模型
            train_loader: 训练数据加载器
            loss: 当前损失
            lr: 当前学习率
            epoch_time: epoch耗时（秒）
            collect_full: 是否收集完整信息（否则只收集基本信息）
        """
        self.init_weather(weather)
        
        diag = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "training": {
                "loss": float(loss),
                "lr": float(lr),
                "epoch_time_sec": float(epoch_time),
            }
        }
        
        # 收集梯度信息（始终收集，这是关键诊断）
        diag["gradients"] = self.collect_gradient_stats(model)
        
        # 完整收集（每隔N个epoch）
        if collect_full:
            # 收集权重统计
            diag["weights"] = self.collect_weight_stats(model)
            
            # 从数据加载器获取一批数据进行统计
            try:
                batch = next(iter(train_loader))
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    features, targets = batch[0], batch[1]
                else:
                    features, targets = batch
                
                # 数据统计
                diag["data"] = self.collect_data_stats(features, targets, weather)
                
                # 模型输出统计
                model.eval()
                with torch.no_grad():
                    device = next(model.parameters()).device
                    features = features.to(device)
                    targets = targets.to(device)
                    predictions, _ = model(features)
                    diag["model_output"] = self.collect_prediction_stats(predictions, targets)
                model.train()
                
            except Exception as e:
                logger.warning(f"收集数据/输出统计时出错: {e}")
        
        # 存储到历史
        self.diagnostics[weather].append(diag)
        
        return diag
    
    def get_summary(self, weather: str) -> Dict[str, Any]:
        """获取指定天气的诊断摘要"""
        if weather not in self.diagnostics or not self.diagnostics[weather]:
            return {}
        
        records = self.diagnostics[weather]
        
        # 提取关键指标序列
        grad_norms = [r["gradients"]["total_grad_norm"] for r in records if "gradients" in r]
        zero_grad_ratios = [r["gradients"]["zero_grad_ratio"] for r in records if "gradients" in r]
        losses = [r["training"]["loss"] for r in records if "training" in r]
        
        # 检测问题
        issues = []
        
        # 检查梯度是否持续为0
        if grad_norms and np.mean(grad_norms[-5:]) < 1e-6:
            issues.append("CRITICAL: 梯度持续为0，模型无法学习")
        
        # 检查损失是否停滞
        if len(losses) >= 10:
            recent_std = np.std(losses[-10:])
            if recent_std < 1e-6:
                issues.append("WARNING: 损失停滞，可能陷入局部最优")
        
        # 检查是否输出常数
        for r in records[-3:]:
            if "model_output" in r and r["model_output"].get("is_constant_output", False):
                issues.append("CRITICAL: 模型输出常数值，可能是死神经元")
                break
        
        return {
            "weather": weather,
            "total_epochs": len(records),
            "final_loss": losses[-1] if losses else None,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            "grad_norm_final": grad_norms[-1] if grad_norms else None,
            "zero_grad_ratio_mean": float(np.mean(zero_grad_ratios)) if zero_grad_ratios else None,
            "issues_detected": issues,
        }
    
    def save(self, save_dir: Optional[Path] = None) -> Path:
        """保存诊断信息到JSON文件"""
        save_dir = Path(save_dir) if save_dir else self.save_dir
        if save_dir is None:
            raise ValueError("未指定保存目录")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "training_diagnostics.json"
        
        # 构建完整的诊断报告
        report = {
            "meta": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "collect_interval": self.collect_interval,
            },
            "summary": {
                weather: self.get_summary(weather) 
                for weather in self.diagnostics.keys()
            },
            "detailed_history": self.diagnostics,
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[*] 训练诊断已保存至: {save_path}")
        
        return save_path
    
    def print_summary(self):
        """打印诊断摘要（仅关键问题）"""
        for weather in self.diagnostics.keys():
            summary = self.get_summary(weather)
            if summary.get("issues_detected"):
                logger.warning(f"[诊断] {weather} 检测到问题:")
                for issue in summary["issues_detected"]:
                    logger.warning(f"  - {issue}")
