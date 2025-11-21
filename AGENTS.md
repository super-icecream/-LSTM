# Repository Guidelines

## 项目结构与模块组织
- `src/`：含 data_processing、feature_engineering、models、training、evaluation 五大子模块，所有新算法放在对应子目录并提供 `__init__.py` 导出。
- `config/`：集中存放 `config.yaml`、`model_config.yaml` 与数据路径，请通过 YAML 参数驱动实验，勿在代码中硬编码。
- `environment/`：`environment.yml` 和 `setup_env.sh` 定义可复现依赖，变更版本后同步更新此处。
- `datas/` 持有原始多站点样本，保持只读；`data/` 承载派生数据（processed/features/splits/cache），写脚本时注意区分。
- `experiments/` 用于保存 Walk-Forward 记录与指标，`docs/` 存放论文与架构材料，`scripts/` 聚合 CLI 工具（prepare/train/evaluate/diagnose），`tests/` 维护单元测试。

## 构建、测试与开发命令
 - 激活环境：`conda activate "C:\Users\Administrator\桌面\专利\DLFE-LSTM-WSI\.conda"`，或运行 `bash environment/setup_env.sh --cpu` 以安装依赖。
- 标准流程：
```bash
python main.py prepare --run-name demo   # 清洗并缓存
python main.py train --run-name demo     # 训练 LSTM 及天气子模型
python main.py test --run-name demo      # 输出指标与曲线
python main.py walk-forward --run-name wf_demo  # 渐进验证
```
- 快速诊断：`python -m scripts.diagnose_weather --use-cache --sequence-length 24 --ge-min 20`。

## 编码风格与命名规范
- Python 代码遵循 PEP 8，使用 4 空格缩进、类型注解与完整 docstring；配置键使用小写加下划线。
- 文件与模块命名采用 snake_case，类使用 PascalCase，常量全大写；测试夹具放入 `tests/fixtures/`。
- 提交前请运行 `python -m compileall src tests` 以捕获语法错误，并保持 `src/__init__.py` 导出同步。

## 测试规范
- 统一使用 `unittest`，以 `Test*` 类和 `test_*` 方法命名；新增模块需在 `tests/` 下创建镜像文件。
- 本仓要求新增功能单测覆盖率不低于 80%，并在 PR 描述中写明通过的命令。
- 常用命令：
```bash
python -m unittest discover tests -p "test_*.py"
python -m unittest tests.test_feature_engineering
```

## 提交与 Pull Request
- Git 历史沿用 Conventional Commits（例：`feat(v0.5.5): ...`、`docs: ...`），请按 `类型(作用域): 摘要` 格式书写并使用祈使句。
- 为每个 PR 附上：变更背景、配置差异、关键指标（MAPE 或 RMSE）以及 Walk-Forward 屏幕截图或日志片段；关联 issue 并勾选自测清单。
- 禁止在 PR 中提交大体量原始数据，必要样例放入 `data/processed` 或使用脚本生成。

## 环境与安全提示
- 复制 `.env.example` 到本地 `.env`，仅在其中写入最小化凭据；任何密钥不得进入 Git。
- 在 `config/data_config.yaml` 中声明绝对路径时优先使用相对路径或环境变量，避免在团队服务器泄露个人目录。
- 大型实验请使用 `scripts/run_experiment.sh`，其中已对 GPU 显存与缓存清理做保护；提交前清理 `__pycache__` 及临时文件。

## 当前研究问题

  - CI/WSI 天气识别高度依赖人为设定的阈值。首轮训练缺乏可靠先验时，容易将白天样本几乎全部划入同一类别（如全部判为阴天），导致三路 LSTM 子模型的训练样本极度不平衡，最小类样 
  本守卫频繁触发，训练流程被迫暂停。
  - 阈值网格搜索与多次 Walk-Forward 反复校准耗费大量计算资源与时间，影响工程推进效率。

  ## 正在探索的方向

  1. **基于聚类的阈值冷启动**
     - 对白天样本（GE≥20 W/m² 或太阳高度≥5°）构建聚类特征：归一化功率（P/GE 或 P/装机容量）、CI/GHI、温度、气压、湿度及其变化率等。
     - 采用稳健、可解释的 3 簇聚类算法（候选：KMeans/KMedoids、GMM、HDBSCAN、DTW 或 K-shape 等）将样本分为高功率/中功率/低功率三簇，并按簇内平均功率映射到晴/多云/阴。      
     - 统计各簇 CI/WSI 分布，以簇与簇的分位点或均值中点反推出首轮 CI/WSI 阈值，提供“自动化初始划分”。

  2. **工程化落地**
     - 在 `prepare` 阶段新增阈值初始化脚本（计划放在 `scripts/init_thresholds.py` 或复用 `scripts.diagnose_weather`），确保首次运行即可产出阈值与样本量日志。
     - 将聚类产生的阈值写入 `config/feature_engineering`，并记录“晴/多云/阴样本数 + 阈值来源”以保证可追溯性。
     - 维持既有误差反馈机制：聚类仅负责冷启动，训练期间仍按在线自适应规则微调 CI/WSI 权重与边界。

  3. **后续验证**
     - 对不同站点（甘肃、河北等）进行对比实验，评估聚类阈值的稳定性与对 Walk-Forward 结果的影响。
     - 若聚类结果对特定季节/站点仍存在类别稀缺，考虑引入白天覆盖率权重、季节性分层或自适应簇数（例如 X-means、GMM-BIC）。

  复制完毕即可，无需我继续操作。
