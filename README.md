# DLFE-LSTM-WSI: 动态局部特征嵌入的光伏功率预测系统

## 项目简介

DLFE-LSTM-WSI是一个基于动态局部特征嵌入和长短期记忆网络的光伏功率预测系统。该系统通过替代昂贵的全天空成像设备，仅使用低成本的气象传感器（气压、湿度等）实现高精度的光伏功率预测。

激活conda环境：conda activate "C:\Users\Administrator\桌面\专利\DLFE-LSTM-WSI\.conda"

## 核心技术

### 1. 多尺度特征提取架构
- **宏观层面**: 基于CI和WSI的双路径天气状态识别
- **中观层面**: 基于NCA优化的动态相空间重构(DPSR)
- **微观层面**: 基于ADMM算法的动态局部特征嵌入(DLFE)

### 2. 关键算法
- **VMD分解**: 变分模态分解（ADMM优化），将功率信号分解为5个本征模态函数
  - 修正：使用绝对频率处理Nyquist分量，tau=0.1适应噪声数据
  - 质量诊断：重构误差<1e-3，模态正交性，能量分布分析
- **CI计算**: 清晰度指数，基于天文参数计算太阳辐照度透过率
- **WSI计算**: 天气状态指数，基于气象参数的综合评估
- **NCA优化**: 邻域成分分析，学习动态特征权重
- **DLFE-ADMM**: 交替方向乘子法，实现流形学习降维（特征工程核心）

### 3. 数据处理流程
```
原始数据(P,I,T,Pre,Hum)
  ↓
VMD-ADMM分解(5个IMF) [tau=0.1, 绝对频率]
  ↓
双路径天气识别(CI+WSI)
  ↓
DPSR重构(30维) [NCA优化]
  ↓
DLFE降维(30维) [流形学习]
  ↓
LSTM预测 + Walk-Forward验证
  ↓
多天气子模型融合
```

## 项目结构

```
DLFE-LSTM-WSI/
├── config/                 # 配置文件
│   ├── config.yaml         # 主配置
│   ├── model_config.yaml   # 模型配置
│   └── data_config.yaml    # 数据配置
├── environment/            # 环境配置
│   ├── environment.yml     # Conda环境
│   ├── requirements.txt    # Python依赖
│   └── setup_env.sh       # 环境安装脚本
├── src/                    # 源代码
│   ├── data_processing/    # 数据处理模块
│   ├── feature_engineering/# 特征工程模块（算法核心）
│   ├── models/            # 模型定义模块
│   ├── training/          # 训练模块
│   └── evaluation/        # 评估模块
├── tests/                 # 测试代码
├── data/                  # 处理数据目录
│   ├── processed/        # 处理后数据
│   ├── features/         # 特征数据
│   ├── splits/           # 数据划分
│   └── cache/            # 缓存文件
├── datas/                 # 原始数据集目录
│   ├── 1. 河北光伏功率预测数据/
│   ├── 2. 甘肃光伏功率预测数据/
│   ├── 3. 澳大利亚气象和光伏数据集/
│   ├── 4. 光伏模组功率预测数据/
│   └── 5. 内蒙古光伏或风电数据/
├── experiments/           # 实验记录
├── docs/                 # 文档
└── scripts/              # 脚本工具
```

## 技术特点

### 1. 参数隔离原则
严格遵循时序数据的参数隔离原则，所有预处理参数仅从训练集学习，防止数据泄露。

### 2. 时序完整性保护与Walk-Forward验证
- **数据划分**：采用4-Fold Walk-Forward策略，训练集从12月逐步扩展到21月
- **时序严格递增**：测试集/验证集互不重叠，保持因果关系
- **Online Learning**：在测试阶段进行模型微调，权重继承优化

### 3. 多天气自适应
针对晴天、多云、阴天三种天气状态构建专门的子模型，提高预测精度。

### 4. 算法创新性
- 首次将VMD分解与双路径天气识别相结合
- 创新性地使用NCA优化进行动态相空间重构
- 提出基于ADMM的动态局部特征嵌入方法

### 5. 白天/夜间处理策略（不改公式）
- CI/WSI 天气判别仅在“白天”执行：白天判定默认 GE≥20 W/m²（或太阳高度≥5°）。
- 夜间样本不参与天气分类与训练、验证、测试的序列构造（采用“序列末端为白天”的过滤）。
- 诊断与阈值搜索同样只在白天末端序列上进行，避免夜间噪声挤压 overcast。

## 运行环境

- Python >= 3.8
- PyTorch >= 1.12
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0
- Scikit-learn >= 1.0.0

## 快速开始

1. **环境安装**
```bash
cd environment
bash setup_env.sh            # 默认安装GPU版本
# 或者 CPU 环境
bash setup_env.sh --cpu
conda activate dlfe-lstm-wsi
```

2. **数据准备**

- 原始“甘肃光伏功率预测数据集”位于 `./datas/2. 甘肃光伏功率预测数据集/data_original/solar_stations`
- DataLoader 已支持 `.xlsx/.xls`，会自动识别 `Time(year-month-day h:m:s)` 为 `timestamp`
- 关键列自动重命名：
  - `Global horizontal irradiance (W/m2)` → `irradiance`
  - `Air temperature (°C)` → `temperature`
  - `Atmosphere (hpa)` → `pressure`
  - `Relative humidity (%)` → `humidity`
  - `Power (MW)` → `power`（自动转化为 kW）
- 其余列仍保留，可按需在特征工程阶段使用
- 原始数据集存放在 `datas/` 目录，处理后数据存放于 `data/` 目录
- 如果换用其它数据集，可在 `config/config.yaml` 或命令行参数中覆盖 `data.raw_dir`

3. **配置修改**
根据实际需求修改 `config/` 目录下的配置文件

4. **运行训练 / 评估**
```bash
# 标准模式（单次训练）
python main.py prepare --run-name demo
python main.py train --run-name demo
python main.py test --run-name demo

# Walk-Forward验证模式（4-Fold渐进验证）
python main.py walk-forward --run-name wf_demo

# 诊断/校准（仅终端输出，不落盘；默认只看白天样本）
python -m scripts.diagnose_weather --use-cache --sequence-length 24 --ge-min 20
# 如需阈值建议（粗搜索）：
python -m scripts.diagnose_weather --use-cache --calibrate --sequence-length 24 --ge-min 20

# 或使用脚本包装
python scripts/prepare_data.py --run-name demo
python scripts/train_model.py --run-name demo
python scripts/evaluate_model.py --run-name demo
```

## 数据格式

输入数据应包含以下列：
- `power` 或 `P`: 光伏功率 (kW)
- `irradiance` 或 `I`: 太阳辐照度 (W/m²)
- `temperature` 或 `T`: 温度 (°C)
- `pressure` 或 `Pre`: 气压 (hPa)
- `humidity` 或 `Hum`: 相对湿度 (%)
- `datetime`: 时间戳

## 模型性能

该系统在甘肃光伏数据集上的预测性能：
- 晴天条件：MAPE < 5%
- 多云条件：MAPE < 8%
- 阴天条件：MAPE < 12%

## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 作者

DLFE-LSTM-WSI Team

## 致谢

感谢甘肃省提供的光伏发电数据支持。
