# 聚类阈值初始化脚本使用说明（聚类助手）

本文档说明如何使用 `src/tools/init_thresholds.py` 辅助完成“晴/多云/阴”三类的无监督聚类，并反推 CI/WSI 阈值。内容为中文，UTF‑8 编码。

## 1. 脚本作用
- 读取训练集（与 `main.py` 一致的分割和配置）
- 进行日间筛选（默认 GHI 法，可调）
- 构建聚类特征、执行 KMeans(k=3) 聚类
- 可选：
  - 从“人工典型日”学习三类原型中心（强烈推荐）
  - 使用“锚点规则”做半监督初始化
  - 特征权重、PCA、诊断/图表输出
- 输出：`diagnostics/init_thresholds/init_YYYYmmdd_HHMMSS/` 下的 `thresholds.yaml` 与 `diagnostics.json`

## 2. 快速开始（建议）
在项目根目录运行（确保 `config/config.yaml` 正确）：

```bash
python -m src.tools.init_thresholds \
  --config config/config.yaml --split train \
  --gmm-enable --bootstrap 200 \
  --gray-width-ci 0.02 --gray-width-wsi 0.02 \
  --daytime-mode ghi --daytime-ghi-min 5 \
  --features-basic-only \
  --learn-prototypes \
  --proto-sunny src/tools/晴天.xlsx \
  --proto-cloudy src/tools/多云.xlsx \
  --proto-overcast src/tools/雨天.xlsx \
  --prototypes-out diagnostics/anchors/prototypes.yaml \
  --proto-tolerance-mins 8 --prototypes-min-count 30 \
  --feat-weight-tod 3 --feat-weight-diffuse 3 --feat-weight-z 1
```

含义：
- 基于 GHI≥5 W/m² 的日间掩码；
- “基础字段模式”仅用时间正余弦 + power/GHI/DNI(若有)/irradiance_total(若有)/温度/气压/湿度；
- 从三份 Excel 典型日时间戳学习原型中心（每类≥30），并保存 `diagnostics/anchors/prototypes.yaml`；
- 特征权重：时间×3、直射占比×3、z分数×1（基础字段模式下 z_* 默认不生效）。

运行完成后，在“聚类结果统计”里应看到较为均衡的三类样本数。

## 3. 直接使用已保存原型中心
前述命令成功后，后续可直接复用：

```bash
python -m src.tools.init_thresholds \
  --config config/config.yaml --split train \
  --gmm-enable --bootstrap 200 \
  --gray-width-ci 0.02 --gray-width-wsi 0.02 \
  --daytime-mode ghi --daytime-ghi-min 5 \
  --features-basic-only \
  --use-prototypes diagnostics/anchors/prototypes.yaml \
  --feat-weight-tod 3 --feat-weight-diffuse 3 --feat-weight-z 1
```

## 4. 可选：启用“锚点规则”
当仅依赖原型仍不理想时，可叠加“锚点”在全体白天样本内进行二次拉拽：

```bash
... --kmeans-anchors --anchor-midday-half-min 720 --anchor-q-high 0.6 --anchor-q-low 0.4
```

- 需要 DNI 列（无 DNI 会自动跳过）；
- 内部使用直射占比 `BF=(DNI·cosZ)/GHI`（cosZ 由 GE/(I0·E0) 计算），q 参数为高/低分位阈值；
- `--anchor-midday-half-min 720` 等价“白天全时段”；
- 锚点足够时会打印 `KMeans anchors: sunny=..., cloudy=..., overcast=...`；不足则跳过。

## 5. 可选：PCA 与缩放
- 缩放器：`--scaler robust|standard`（默认 robust）
- PCA：`--use-pca --pca-n 3`（可改善主方向的表达）

## 6. 日间掩码设置
- 默认使用 GHI 法：`--daytime-mode ghi --daytime-ghi-min 5`
- 也支持：`ge|or|and` 模式（与 GE 阈值组合），详见 `--daytime-mode` 帮助。

## 7. 典型日 Excel/CSV 要求
- 至少包含一个时间戳列（以下列名之一）：
  - `timestamp` / `time` / `datetime` / `Time(year-month-day h:m:s)` / `Time (year-month-day h:m:s)`
- 仅“时间戳子集”即可，不必全日；脚本会与训练集 `df_day.index` 最近邻对齐（默认容差 ±8 分钟，可用 `--proto-tolerance-mins` 调整）；
- 典型日样本越“典型”，原型越稳健。建议：
  - 晴：正午附近，高 GHI，高直射占比（BF 高）；
  - 多云：正午附近，高 GHI，低直射占比（BF 低）；
  - 阴：低 GHI，低直射占比。

## 8. 特征模式与权重
- `--features-basic-only`：仅用基础字段 + 时间正余弦（推荐做原型学习与最终聚类）
- 若不使用该开关，则默认还会构建滚动统计与分时 z-score，适合做算法对比；
- 权重：`--feat-weight-tod`（时间）、`--feat-weight-diffuse`（直射占比 BF）、`--feat-weight-z`（z_*）

## 9. 输出产物
- `diagnostics/init_thresholds/init_YYYYmmdd_HHMMSS/`
  - `thresholds.yaml`：CI/WSI 阈值（含灰区带宽）
  - `diagnostics.json`：聚类计数、一致性指标、GMM 诊断、灰区占比等
  - 若 `--prototypes-out`：在 `diagnostics/anchors/` 下保存 `prototypes.yaml`

## 10. 常见问题
- 日志出现 `Prototypes (...) matches: sunny=..., cloudy=..., overcast=...`：表示原型学习匹配成功；若三类其中之一小于 `--prototypes-min-count`，将跳过并回退到锚点/普通 KMeans；
- 锚点被跳过：`KMeans anchors skipped (insufficient anchor samples)` 或 `DNI not available`；对应增大时间窗口、放宽 q 阈值，或确认数据中是否有 DNI 列；
- 三类极不均衡：
  - 优先检查典型日是否“典型”；
  - 可提高 `--feat-weight-diffuse`（BF 的权重），或启用 `--kmeans-anchors`；
  - 若仍不佳，尝试 `--use-pca --pca-n 3`；
- Windows 路径含中文：建议在命令中使用 `/` 或用引号包裹路径。

## 11. 推荐流程
1) 准备三份典型日 Excel（晴/多云/阴，列含时间戳；尽量选择正午典型时刻）；
2) 运行“学习并使用原型”的命令（第 2 节），观察聚类统计是否均衡；
3) 固化原型，后续直接 `--use-prototypes`；
4) 需要时加 `--kmeans-anchors` 做小幅拉拽；
5) 查看 `thresholds.yaml` 与 `diagnostics.json`，将 CI/WSI 阈值写回配置或用于下游模型。

—— 以上为聚类阈值初始化的推荐用法。若需要扩展（季节化/多站点/更复杂的诊断），可在此文件基础上继续完善。 

