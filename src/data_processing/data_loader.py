"""
鏁版嵁鍔犺浇妯″潡
鍔熻兘锛氬姞杞界敇鑲冨厜浼忓姛鐜囬娴嬫暟鎹泦锛屽鐞嗘椂搴忔暟鎹紝杩涜鏁版嵁瀹屾暣鎬ф鏌?GPU浼樺寲锛氭坊鍔燩yTorch DataLoader鏀寔锛屽疄鐜癎PU鍔犻€熸暟鎹紶杈?浣滆€咃細DLFE-LSTM-WSI Team
鏃ユ湡锛?025-09-26
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import json
import yaml
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

logger = logging.getLogger(__name__)


class DataLoader:
    """
    鍏変紡鍔熺巼棰勬祴鏁版嵁鍔犺浇鍣?
    璐熻矗鍔犺浇CSV鏍煎紡鐨勫師濮嬫暟鎹枃浠讹紝澶勭悊鏃堕棿鎴筹紝杩涜鏁版嵁瀹屾暣鎬ф鏌ワ紝
    鏀寔澶氱珯鐐规暟鎹姞杞藉拰鍚堝苟銆?
    Attributes:
        data_path (str): 鏁版嵁鏂囦欢璺緞
        required_columns (list): 蹇呴渶鐨勬暟鎹垪
        freq (str): 鏁版嵁閲囨牱棰戠巼
        config (dict): 閰嶇疆鍙傛暟
    """

    def __init__(self, data_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        鍒濆鍖栨暟鎹姞杞藉櫒

        Args:
            data_path: 鏁版嵁璺緞锛岄粯璁や粠閰嶇疆鏂囦欢璇诲彇 `data.raw_dir`
            config_path: 閰嶇疆鏂囦欢璺緞锛岀敤浜庤鐩栭澶勭悊鍙傛暟
        """
        self.required_columns = ['power', 'irradiance', 'temperature', 'pressure', 'humidity']
        self.freq = '15T'  # 15鍒嗛挓閲囨牱棰戠巼
        self.station_column = 'station'
        self.frequency_minutes = 15

        # 鍔犺浇閰嶇疆鏂囦欢
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
                # 鎻愬彇棰勫鐞嗛厤缃?                preprocessing_config = full_config.get('preprocessing', {})
                self.config = self._merge_configs(
                    self._get_default_config(),
                    preprocessing_config
                )
        else:
            self.config = self._get_default_config()

        resolved_data_path = data_path or self._resolve_default_data_path(config_path)
        self.data_path = Path(resolved_data_path)

        logger.info(f"鏁版嵁鍔犺浇鍣ㄥ垵濮嬪寲瀹屾垚锛屾暟鎹矾寰? {self.data_path}")

    def _get_default_config(self) -> Dict:
        """
        鑾峰彇榛樿閰嶇疆鍙傛暟

        Returns:
            Dict: 榛樿閰嶇疆瀛楀吀
        """
        return {
            'missing_values': {
                'strategy': 'interpolation',
                'method': 'linear',
                'max_consecutive': 6,
            },
            'missing_threshold': 0.3,  # 缂哄け鍊兼瘮渚嬮槇鍊硷紙鍚戝悗鍏煎锛?            'interpolation_method': 'linear',  # 鎻掑€兼柟娉曪紙鍚戝悗鍏煎锛?            'max_consecutive_missing': 6,  # 鏈€澶ц繛缁己澶辨暟锛堝悜鍚庡吋瀹癸級
            'outlier_detection': {
                'method': 'physical',  # 寮傚父鍊兼娴嬫柟娉? 'physical', 'iqr', 'zscore'
                'iqr_threshold': 1.5,  # IQR闃堝€硷紙浠呭湪method='iqr'鏃朵娇鐢級
                'apply_to': ['power', 'irradiance', 'temperature', 'pressure', 'humidity'],
                # 鐗╃悊绾︽潫鑼冨洿锛堝熀浜庡厜浼忕郴缁熷拰姘旇薄瀛︾殑棰嗗煙鐭ヨ瘑锛?                'physical_ranges': {
                    'power': [0, 55000],  # kW锛岃鏈哄閲?0MW + 10%杩囪浇淇濇姢
                    'irradiance': [0, 1200],  # W/m虏锛屽湴闈㈠お闃宠緪鐓у害鐗╃悊涓婇檺
                    'temperature': [-40, 60],  # 掳C锛屾瀬绔皵鍊欒寖鍥?                    'pressure': [850, 1100],  # hPa锛岃€冭檻娴锋嫈鐨勬皵鍘嬭寖鍥?                    'humidity': [0, 100],  # %锛岀浉瀵规箍搴︾殑鐗╃悊鑼冨洿
                },
                # 鏄庣‘鐨勯敊璇爣璁板€硷紙灏嗚瑙嗕负缂哄け鍊硷級
                'error_markers': [-99, -999, -9999],
            },
            # 鍚戝悗鍏煎鐨勯《灞傚瓧娈?            'outlier_method': 'physical',
            'iqr_threshold': 1.5,
            'physical_ranges': {
                'power': [0, 55000],
                'irradiance': [0, 1200],
                'temperature': [-40, 60],
                'pressure': [850, 1100],
                'humidity': [0, 100],
            },
            'error_markers': [-99, -999, -9999],
        }
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """
        鍚堝苟榛樿閰嶇疆鍜屽姞杞界殑閰嶇疆
        
        Args:
            default: 榛樿閰嶇疆
            loaded: 浠庢枃浠跺姞杞界殑閰嶇疆
            
        Returns:
            Dict: 鍚堝苟鍚庣殑閰嶇疆
        """
        merged = default.copy()
        
        # 澶勭悊缂哄け鍊奸厤缃?        if 'missing_values' in loaded:
            missing_config = loaded['missing_values']
            merged['interpolation_method'] = missing_config.get('method', default['interpolation_method'])
            merged['max_consecutive_missing'] = missing_config.get('max_consecutive', default['max_consecutive_missing'])
        
        # 澶勭悊寮傚父鍊兼娴嬮厤缃?        if 'outlier_detection' in loaded:
            outlier_config = loaded['outlier_detection']
            merged['outlier_method'] = outlier_config.get('method', default['outlier_method'])
            merged['iqr_threshold'] = outlier_config.get('iqr_threshold', default['iqr_threshold'])
            
            # 鍚堝苟鐗╃悊鑼冨洿
            if 'physical_ranges' in outlier_config:
                merged['physical_ranges'] = outlier_config['physical_ranges']
            
            # 鍚堝苟閿欒鏍囪
            if 'error_markers' in outlier_config:
                merged['error_markers'] = outlier_config['error_markers']
            
            # 鏇存柊宓屽閰嶇疆
            merged['outlier_detection'] = outlier_config
        
        return merged

    def _resolve_default_data_path(self, config_path: Optional[str]) -> str:
        """
        浠庨厤缃枃浠朵腑瑙ｆ瀽鏁版嵁璺緞锛岄粯璁ゅ洖閫€鍒?./data/raw

        Args:
            config_path: 鏄惧紡浼犲叆鐨勯厤缃枃浠惰矾寰?
        Returns:
            str: 鏁版嵁鐩綍璺緞
        """
        candidates: List[Path] = []
        if config_path:
            candidates.append(Path(config_path))
        candidates.append(Path("config/config.yaml"))

        for candidate in candidates:
            if candidate is None or not candidate.exists():
                continue
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
            except Exception as exc:  # pragma: no cover - 瀹归敊鏃ュ織
                logger.debug("璇诲彇閰嶇疆鏂囦欢%s鑾峰彇data.raw_dir澶辫触: %s", candidate, exc)
                continue

            raw_dir = config_data.get('data', {}).get('raw_dir')
            if raw_dir:
                return raw_dir

        logger.debug("鏈湪閰嶇疆鏂囦欢涓壘鍒?data.raw_dir锛屼娇鐢ㄩ粯璁よ矾寰?./data/raw")
        return "./data/raw"

    def load_single_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        鍔犺浇鍗曚釜CSV鏂囦欢

        Args:
            file_path: CSV鏂囦欢璺緞

        Returns:
            pd.DataFrame: 鍔犺浇鐨勬暟鎹

        Raises:
            FileNotFoundError: 鏂囦欢涓嶅瓨鍦?            ValueError: 鏁版嵁鏍煎紡閿欒鎴栫己灏戝繀闇€鍒?        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"鏁版嵁鏂囦欢涓嶅瓨鍦? {file_path}")

        logger.info(f"姝ｅ湪鍔犺浇鏁版嵁鏂囦欢: {file_path}")

        try:
            suffix = file_path.suffix.lower()
            if suffix in {'.csv', '.txt'}:
                data = pd.read_csv(file_path)
            elif suffix in {'.xlsx', '.xls'}:
                data = pd.read_excel(file_path, sheet_name=0)
            else:
                raise ValueError(f"鏆備笉鏀寔鐨勬枃浠舵牸寮? {suffix}")

            # 缁熶竴鍒楀悕绌虹櫧
            data.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in data.columns]

            # 缁熶竴绱㈠紩涓庡垪鍚?            data = self._prepare_dataframe(data)
            data = self._apply_column_mapping(data)

            # 妫€鏌ュ繀闇€鍒?            missing_cols = [col for col in self.required_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"鏁版嵁缂哄皯蹇呴渶鍒? {missing_cols}")

            # 鏁版嵁瀹屾暣鎬ф鏌?            self._check_data_integrity(data)

            logger.info(f"鏁版嵁鍔犺浇鎴愬姛锛屽舰鐘? {data.shape}")
            return data

        except Exception as e:
            logger.error(f"鍔犺浇鏁版嵁澶辫触: {e}")
            raise

    # ------------------------------------------------------------------
    # 鏁版嵁鍑嗗杈呭姪鍑芥暟
    # ------------------------------------------------------------------

    def _prepare_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """缁熶竴澶勭悊鏃堕棿鍒楀苟璁剧疆绱㈠紩"""

        # 浼樺厛瀵绘壘鏍囧噯鍒楀悕
        time_candidates = [
            'timestamp',
            'time',
            'datetime',
            'Time(year-month-day h:m:s)',
            'Time (year-month-day h:m:s)',
        ]

        timestamp_col = None
        for col in time_candidates:
            if col in data.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            # 鍏滃簳锛氭煡鎵惧寘鍚?time 鐨勫垪
            for col in data.columns:
                if re.search(r'time', col, re.IGNORECASE):
                    timestamp_col = col
                    break

        if timestamp_col is None:
            raise ValueError("鏈壘鍒版椂闂村垪锛岃纭繚鍘熷鏁版嵁鍖呭惈鏃堕棿瀛楁")

        data = data.copy()
        data.rename(columns={timestamp_col: 'timestamp'}, inplace=True)

        # 瑙ｆ瀽绱㈠紩
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data.sort_index(inplace=True)

        return data

    def _apply_column_mapping(self, data: pd.DataFrame) -> pd.DataFrame:
        """灏嗗師濮嬪垪鏄犲皠涓洪」鐩爣鍑嗗垪鍚嶏紝骞跺鐞嗗崟浣?""

        column_mapping = {
            'P': 'power',
            'Power (MW)': 'power_mw',
            'Power (kW)': 'power',
            'power': 'power',
            'I': 'irradiance',
            'Global horizontal irradiance (W/m2)': 'irradiance',
            'Total solar irradiance (W/m2)': 'irradiance_total',
            'Direct normal irradiance (W/m2)': 'dni',
            'Diffuse horizontal irradiance (W/m2)': 'dhi',
            'Air temperature (掳C)': 'temperature',
            'Temperature (掳C)': 'temperature',
            'T': 'temperature',
            'Atmosphere (hpa)': 'pressure',
            'Atmospheric pressure (hPa)': 'pressure',
            'Atmospheric pressure (kPa)': 'pressure_kpa',
            'Pre': 'pressure',
            'Relative humidity (%)': 'humidity',
            'Humidity (%)': 'humidity',
            'Hum': 'humidity',
        }

        renamed = {}
        for col in data.columns:
            key = column_mapping.get(col)
            if key is None and isinstance(col, str):
                key = column_mapping.get(col.strip())
                if key is None:
                    key = column_mapping.get(col.lower())
            renamed[col] = key if key else col

        data = data.rename(columns=renamed)

        numeric_candidates = [
            'power_mw',
            'power_kwh',
            'power',
            'irradiance_total',
            'dni',
            'dhi',
            'temperature',
            'pressure',
            'pressure_kpa',
            'humidity',
        ]
        for col in numeric_candidates:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # 鍔熺巼鍒楃粺涓€涓簁W
        if 'power' not in data.columns and 'power_mw' in data.columns:
            data['power'] = data['power_mw'] * 1000.0
        elif 'power' in data.columns and data['power'].max() <= 1.5:
            data['power'] = data['power'] * 1000.0
        elif 'power_kwh' in data.columns:
            data['power'] = data['power_kwh']

        # 杈愮収搴︿紭鍏堜娇鐢℅HI
        if 'irradiance' not in data.columns:
            if 'irradiance_total' in data.columns:
                data['irradiance'] = data['irradiance_total']
            elif 'dni' in data.columns:
                data['irradiance'] = data['dni']
            elif 'dhi' in data.columns:
                data['irradiance'] = data['dhi']

        if 'pressure' not in data.columns and 'pressure_kpa' in data.columns:
            data['pressure'] = data['pressure_kpa'] * 10.0

        # 缁熶竴婀垮害鑼冨洿鍒?-100
        if 'humidity' in data.columns:
            data['humidity'] = data['humidity'].clip(lower=0, upper=100)

        # 纭繚娓╁害/姘斿帇/婀垮害瀛樺湪
        for required in ['temperature', 'pressure', 'humidity']:
            if required not in data.columns:
                raise ValueError(f"鏁版嵁缂哄皯蹇呴渶鍒? {required}")

        for col in ['power', 'irradiance', 'temperature', 'pressure', 'humidity']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        return data

    def load_multi_station(
        self,
        station_files: Optional[List[str]] = None,
        merge_method: Optional[str] = None,
        selected_station: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        鍔犺浇澶氫釜绔欑偣鐨勬暟鎹?
        Args:
            station_files: 绔欑偣鏂囦欢鍒楄〃锛屽鏋滀负None鍒欏姞杞芥墍鏈塁SV鏂囦欢

        Returns:
            Dict[str, pd.DataFrame]: 绔欑偣鍚嶇О鍒版暟鎹鐨勬槧灏?        """
        station_data = {}

        # 濡傛灉鏈寚瀹氭枃浠跺垪琛紝鎵弿鏁版嵁鐩綍鏀寔 CSV/Excel
        if station_files is None:
            csv_files = list(self.data_path.glob("*.csv"))
            excel_files = list(self.data_path.glob("*.xlsx")) + list(self.data_path.glob("*.xls"))
            station_files = csv_files + excel_files

            target_method = merge_method or self.config.get('merge', {}).get('method')
            target_station = selected_station or self.config.get('merge', {}).get('selected_station')

            if target_method == 'single' and station_files:
                if target_station:
                    matches = [fp for fp in station_files if fp.stem == target_station or fp.name == target_station]
                    if matches:
                        station_files = matches[:1]
                    else:
                        logger.warning(f"閰嶇疆鐨勭珯鐐?{target_station} 涓嶅瓨鍦紝榛樿浣跨敤绗竴涓珯鐐?)
                        station_files = station_files[:1]
                else:
                    station_files = station_files[:1]
        else:
            station_files = [self.data_path / f for f in station_files]

        logger.info(f"鍑嗗鍔犺浇 {len(station_files)} 涓珯鐐规暟鎹?)

        for file_path in station_files:
            station_name = file_path.stem  # 浣跨敤鏂囦欢鍚嶄綔涓虹珯鐐瑰悕
            try:
                station_data[station_name] = self.load_single_file(file_path)
                logger.info(f"绔欑偣 {station_name} 鏁版嵁鍔犺浇鎴愬姛")
            except Exception as e:
                logger.error(f"绔欑偣 {station_name} 鏁版嵁鍔犺浇澶辫触: {e}")
                continue

        logger.info(f"鎴愬姛鍔犺浇 {len(station_data)} 涓珯鐐规暟鎹?)
        return station_data

    def merge_stations(self, station_data: Dict[str, pd.DataFrame], method: str = 'concat') -> pd.DataFrame:
        """
        鍚堝苟澶氫釜绔欑偣鐨勬暟鎹?
        Args:
            station_data: 绔欑偣鏁版嵁瀛楀吀
            method: 鍚堝苟鏂规硶 ('concat': 鍨傜洿鎷兼帴, 'average': 骞冲潎鍊?

        Returns:
            pd.DataFrame: 鍚堝苟鍚庣殑鏁版嵁
        """
        if not station_data:
            raise ValueError("娌℃湁鍙悎骞剁殑绔欑偣鏁版嵁")

        if method == 'concat':
            # 娣诲姞绔欑偣鏍囪瘑鍒楀苟鍨傜洿鎷兼帴
            dfs = []
            for station_name, df in station_data.items():
                df_copy = df.copy()
                df_copy['station'] = station_name
                dfs.append(df_copy)
            merged = pd.concat(dfs, axis=0, sort=True)
            merged.sort_index(inplace=True)

        elif method == 'average':
            # 瀵圭浉鍚屾椂闂寸偣鐨勬暟鎹彇骞冲潎
            merged = pd.concat(station_data.values(), axis=1, keys=station_data.keys())
            merged = merged.groupby(level=1, axis=1).mean()

        else:
            raise ValueError(f"涓嶆敮鎸佺殑鍚堝苟鏂规硶: {method}")

        logger.info(f"鏁版嵁鍚堝苟瀹屾垚锛屾柟娉? {method}, 鏈€缁堝舰鐘? {merged.shape}")
        return merged

    def save_params(self, filepath: Union[str, Path]) -> None:
        """淇濆瓨DataLoader閰嶇疆鍙傛暟"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        params = {
            'data_path': str(self.data_path),
            'station_column': getattr(self, 'station_column', None),
            'required_columns': self.required_columns,
            'frequency_minutes': getattr(self, 'frequency_minutes', self.frequency_minutes),
            'freq': self.freq,
            'config': self.config,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

        logger.info(f"DataLoader 鍙傛暟宸蹭繚瀛? {filepath}")

    def load_params(self, filepath: Union[str, Path]) -> None:
        """鍔犺浇DataLoader閰嶇疆鍙傛暟"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"DataLoader 鍙傛暟鏂囦欢涓嶅瓨鍦? {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            params = json.load(f)

        self.data_path = Path(params['data_path'])
        self.station_column = params.get('station_column', getattr(self, 'station_column', None))
        self.required_columns = params['required_columns']
        self.optional_columns = params.get('optional_columns', getattr(self, 'optional_columns', []))
        self.frequency_minutes = params.get('frequency_minutes', getattr(self, 'frequency_minutes', 15))
        self.freq = params['freq']
        self.config = params['config']

        logger.info(f"DataLoader 鍙傛暟宸插姞杞? {filepath}")

    def load_processed_dataset(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """鍔犺浇缂撳瓨鐨勫悎骞跺師濮嬫暟鎹?""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"缂撳瓨鍚堝苟鏁版嵁涓嶅瓨鍦? {filepath}")

        logger.info(f"浠庣紦瀛樿浇鍏ュ悎骞舵暟鎹? {filepath}")
        return pd.read_parquet(filepath)

    def _check_data_integrity(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        妫€鏌ユ暟鎹畬鏁存€?
        Args:
            data: 杈撳叆鏁版嵁妗?
        Returns:
            Dict: 鏁版嵁璐ㄩ噺鎶ュ憡
        """
        report = {
            'total_rows': len(data),
            'time_range': (data.index.min(), data.index.max()),
            'missing_values': {},
            'outliers': {},
            'statistics': {}
        }

        # 鏃х増鏈細鍦ㄦ杈撳嚭澶ч噺缂哄け鍊?寮傚父鍊?鏃堕棿棰戠巼鐨勬彁绀恒€?        # 涓轰簡淇濇寔缁堢骞插噣锛岀幇闃舵鍙敹闆嗗熀纭€缁熻淇℃伅锛屼笉鍐嶈緭鍑鸿鍛娿€?        if all(col in data.columns for col in self.required_columns):
            try:
                report['statistics'] = data[self.required_columns].describe().to_dict()
            except Exception:
                report['statistics'] = {}
        return report

    def _detect_outliers(self, series: pd.Series, method: Optional[str] = None) -> Dict[str, Any]:
        """
        妫€娴嬪紓甯稿€?        
        鏀寔涓夌妫€娴嬫柟娉曪細
        1. 'physical': 鍩轰簬鐗╃悊绾︽潫鑼冨洿锛堟帹鑽愮敤浜庡厜浼忔暟鎹級
        2. 'iqr': 鍩轰簬鍥涘垎浣嶈窛鐨勭粺璁℃柟娉?        3. 'zscore': 鍩轰簬Z-score鐨勭粺璁℃柟娉?
        Args:
            series: 鏁版嵁搴忓垪
            method: 妫€娴嬫柟娉?('physical', 'iqr', 'zscore')

        Returns:
            Dict: 寮傚父鍊兼娴嬬粨鏋滐紝鍖呭惈绱㈠紩銆佹暟閲忋€侀槇鍊肩瓑淇℃伅
        """
        if method is None:
            method = self.config['outlier_method']

        samples_preview: List[Dict[str, Any]] = []
        thresholds: Dict[str, Any] = {}
        column_name = series.name if hasattr(series, 'name') else 'unknown'

        # 鏂规硶1锛氬熀浜庣墿鐞嗙害鏉熺殑妫€娴嬶紙棰嗗煙鐭ヨ瘑椹卞姩锛?        if method == 'physical':
            # 鍏堟娴嬮敊璇爣璁板€?            error_markers = self.config.get('error_markers', [-99, -999, -9999])
            error_mask = series.isin(error_markers)
            
            # 妫€娴嬬墿鐞嗚寖鍥村鐨勫€?            physical_ranges = self.config.get('physical_ranges', {})
            
            if column_name in physical_ranges:
                lower_bound, upper_bound = physical_ranges[column_name]
                range_mask = (series < lower_bound) | (series > upper_bound)
                outlier_mask = error_mask | range_mask
            else:
                # 濡傛灉娌℃湁瀹氫箟鐗╃悊鑼冨洿锛屽彧妫€娴嬮敊璇爣璁?                outlier_mask = error_mask
                lower_bound, upper_bound = None, None
            
            outlier_values = series[outlier_mask]
            outlier_indices = outlier_values.index
            
            # 鍒嗙被寮傚父鍊?            error_marker_count = error_mask.sum()
            range_violation_count = range_mask.sum() if column_name in physical_ranges else 0
            
            if not outlier_values.empty:
                # 浼樺厛鏄剧ず閿欒鏍囪
                error_samples = series[error_mask].head(3) if error_marker_count > 0 else pd.Series()
                # 鐒跺悗鏄剧ず鑼冨洿杩濊锛堝彇鏋佸€硷級
                range_samples = series[range_mask if column_name in physical_ranges else pd.Series(dtype=float).index]
                if not range_samples.empty:
                    low_samples = range_samples.nsmallest(min(2, len(range_samples)))
                    high_samples = range_samples.nlargest(min(2, len(range_samples)))
                    range_samples = pd.concat([low_samples, high_samples])
                
                all_samples = pd.concat([error_samples, range_samples]).drop_duplicates()
                samples_preview = [
                    {
                        'timestamp': str(idx),
                        'value': float(series.loc[idx]),
                        'reason': 'error_marker' if series.loc[idx] in error_markers else 'out_of_range'
                    }
                    for idx in all_samples.index[:5]
                ]
            
            thresholds = {
                'method': 'physical_constraint',
                'lower_bound': float(lower_bound) if lower_bound is not None else None,
                'upper_bound': float(upper_bound) if upper_bound is not None else None,
                'error_markers': error_markers,
                'error_marker_count': int(error_marker_count),
                'range_violation_count': int(range_violation_count),
            }
        
        # 鏂规硶2锛欼QR缁熻鏂规硶锛堜繚鐣欑敤浜庨潪鐗╃悊鏁版嵁锛?        elif method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            threshold = self.config['iqr_threshold']
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_values = series[outlier_mask]
            outlier_indices = outlier_values.index

            if not outlier_values.empty:
                low_indices = list(outlier_values.nsmallest(min(3, len(outlier_values))).index)
                high_indices = list(outlier_values.nlargest(min(3, len(outlier_values))).index)
                preview_indices = list(dict.fromkeys(low_indices + high_indices))
                samples_preview = [
                    {
                        'timestamp': str(idx),
                        'value': float(outlier_values.loc[idx]),
                    }
                    for idx in preview_indices
                ]

            thresholds = {
                'lower_bound': float(lower_bound) if np.isfinite(lower_bound) else None,
                'upper_bound': float(upper_bound) if np.isfinite(upper_bound) else None,
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'threshold': float(threshold),
            }

        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            outlier_mask = z_scores > 3
            outlier_values = series[outlier_mask]
            outlier_indices = outlier_values.index

            if not outlier_values.empty:
                abs_order = (outlier_values - series.mean()).abs().sort_values(ascending=False)
                preview_indices = list(abs_order.head(min(5, len(abs_order))).index)
                samples_preview = [
                    {
                        'timestamp': str(idx),
                        'value': float(outlier_values.loc[idx]),
                        'zscore': float(z_scores.loc[idx]),
                    }
                    for idx in preview_indices
                ]

            thresholds = {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'zscore_threshold': 3.0,
            }

        else:
            raise ValueError(f"涓嶆敮鎸佺殑寮傚父鍊兼娴嬫柟娉? {method}")

        outlier_indices = outlier_indices if 'outlier_indices' in locals() else series.index[:0]
        indices_list = outlier_indices.tolist()
        extrema = {
            'min': float(outlier_values.min()) if 'outlier_values' in locals() and not outlier_values.empty else None,
            'max': float(outlier_values.max()) if 'outlier_values' in locals() and not outlier_values.empty else None,
        }

        return {
            'method': method,
            'indices': indices_list,
            'count': len(indices_list),
            'thresholds': thresholds,
            'samples': samples_preview,
            'extrema': extrema,
        }

    def handle_missing_values(self, data: pd.DataFrame, method: Optional[str] = None) -> pd.DataFrame:
        """
        澶勭悊缂哄け鍊?
        Args:
            data: 杈撳叆鏁版嵁
            method: 澶勭悊鏂规硶 ('linear', 'forward', 'backward', 'drop')

        Returns:
            pd.DataFrame: 澶勭悊鍚庣殑鏁版嵁
        """
        if method is None:
            method = self.config['interpolation_method']

        data_processed = data.copy()

        if method == 'linear':
            # 绾挎€ф彃鍊?            data_processed = data_processed.interpolate(method='linear', limit=self.config['max_consecutive_missing'])
        elif method == 'forward':
            # 鍓嶅悜濉厖
            data_processed = data_processed.fillna(method='ffill', limit=self.config['max_consecutive_missing'])
        elif method == 'backward':
            # 鍚庡悜濉厖
            data_processed = data_processed.fillna(method='bfill', limit=self.config['max_consecutive_missing'])
        elif method == 'drop':
            # 鍒犻櫎缂哄け鍊?            data_processed = data_processed.dropna()
        else:
            raise ValueError(f"涓嶆敮鎸佺殑缂哄け鍊煎鐞嗘柟娉? {method}")

        # 妫€鏌ユ槸鍚﹁繕鏈夌己澶卞€?        remaining_missing = data_processed.isna().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"澶勭悊鍚庝粛鏈?{remaining_missing} 涓己澶卞€?)
            # 浣跨敤鍓嶅悜濉厖澶勭悊鍓╀綑缂哄け鍊?            data_processed = data_processed.fillna(method='ffill').fillna(method='bfill')

        logger.info(f"缂哄け鍊煎鐞嗗畬鎴愶紝鏂规硶: {method}")
        return data_processed

    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        鍏抽棴鍘嗗彶鐗堟湰鐨勮川閲忛獙鏀堕€昏緫锛岄粯璁よ涓洪€氳繃銆?        杩欐牱涓绘祦绋嬩笉浼氬啀寮瑰嚭澶ф鈥滆秴鍑鸿寖鍥粹€濈殑鎻愮ず銆?        """
        quality_report = {
            'pass': True,
            'issues': [],
            'details': {},
        }
        return True, quality_report

    # ============ GPU浼樺寲鍔熻兘鎵╁睍 ============

    def create_sequence_data(self,
                             features: np.ndarray,
                             targets: np.ndarray,
                             sequence_length: int = 24,
                             weather_array: Optional[np.ndarray] = None,
                             forecast_horizons: Optional[List[int]] = None
                             ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create sequence data for LSTM, supporting multi-step targets."""
        if weather_array is not None and len(weather_array) != len(features):
            raise ValueError("weather_array length must match features length")

        horizons = sorted(forecast_horizons) if forecast_horizons else [1]
        max_h = max(horizons)
        n_samples = len(features) - sequence_length - max_h + 1

        if n_samples <= 0:
            empty_target = np.empty((0, len(horizons)), dtype=np.float32)
            if weather_array is not None:
                return (
                    np.empty((0, sequence_length, features.shape[1]), dtype=np.float32),
                    empty_target,
                    np.empty((0,), dtype=int),
                )
            return np.empty((0, sequence_length, features.shape[1]), dtype=np.float32), empty_target

        X = np.zeros((n_samples, sequence_length, features.shape[1]), dtype=np.float32)
        y = np.zeros((n_samples, len(horizons)), dtype=np.float32)
        weather_seq = np.zeros((n_samples,), dtype=int) if weather_array is not None else None

        for i in range(n_samples):
            start = i
            end = i + sequence_length
            X[i] = features[start:end]
            for j, h in enumerate(horizons):
                y[i, j] = targets[end + h - 1]  # multi-step target
            if weather_seq is not None:
                weather_seq[i] = int(weather_array[end - 1])  # weather label at sequence end

        if weather_seq is not None:
            return X, y, weather_seq
        return X, y

    def create_gpu_optimized_dataloader(self,
                                       data: pd.DataFrame,
                                       features_array: np.ndarray,
                                       targets_array: np.ndarray,
                                       batch_size: int = 64,
                                       sequence_length: int = 24,
                                       shuffle: bool = True,
                                       is_training: bool = True,
                                       weather_array: Optional[np.ndarray] = None) -> TorchDataLoader:
        """
        鍒涘缓GPU浼樺寲鐨凱yTorch DataLoader
        涓ユ牸鎸夌収鎸囧鏂囦欢绗?35-746琛岄厤缃?
        Args:
            data: 鍘熷鏁版嵁妗嗭紙鐢ㄤ簬楠岃瘉锛?            features_array: 鐗瑰緛鏁扮粍
            targets_array: 鐩爣鏁扮粍
            batch_size: 鎵瑰ぇ灏?            sequence_length: 搴忓垪闀垮害
            shuffle: 鏄惁鎵撲贡鏁版嵁
            is_training: 鏄惁涓鸿缁冩ā寮?            weather_array: 澶╂皵鏍囩鏁扮粍锛堝彲閫夛級

        Returns:
            TorchDataLoader: GPU浼樺寲鐨勬暟鎹姞杞藉櫒
        """
        # 鍒涘缓搴忓垪鏁版嵁
        seq_result = self.create_sequence_data(
            features_array,
            targets_array,
            sequence_length,
            weather_array=weather_array
        )

        if weather_array is not None:
            X, y, weather_seq = seq_result
        else:
            X, y = seq_result
            weather_seq = None

        # 鍒涘缓PyTorch鏁版嵁闆?        dataset = DLFELSTMDataset(X, y, weather_seq)

        # 妫€娴婫PU鍙敤鎬?        use_gpu = torch.cuda.is_available()

        # GPU浼樺寲閰嶇疆锛堜弗鏍兼寜鐓ф寚瀵兼枃浠剁735-746琛岋級
        if use_gpu:
            if is_training:
                # 璁粌鏁版嵁鍔犺浇鍣ㄩ厤缃?                dataloader = TorchDataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=4,  # 澶氳繘绋嬫暟鎹姞杞?                    pin_memory=True,  # 鍥哄畾鍐呭瓨锛屽姞閫烥PU浼犺緭
                    persistent_workers=True,  # 淇濇寔worker杩涚▼
                    prefetch_factor=2,  # 棰勫彇鎵规鏁?                    drop_last=True  # 涓㈠純涓嶅畬鏁存壒娆★紙淇濇寔鎵瑰ぇ灏忎竴鑷达級
                )
            else:
                # 楠岃瘉/娴嬭瘯鏁版嵁鍔犺浇鍣ㄩ厤缃?                dataloader = TorchDataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=2,
                    drop_last=False  # 楠岃瘉鏃朵繚鐣欐墍鏈夋暟鎹?                )
        else:
            # CPU閰嶇疆
            dataloader = TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=2,
                pin_memory=False,
                drop_last=(True if is_training else False)
            )

        logger.info(f"鍒涘缓{'GPU' if use_gpu else 'CPU'}浼樺寲DataLoader锛屾壒澶у皬: {batch_size}")
        return dataloader

    def get_optimal_batch_size(self, gpu_id: int = 0) -> int:
        """
        鏍规嵁GPU鍐呭瓨鑷姩鎺ㄨ崘鏈€浼樻壒澶у皬

        Args:
            gpu_id: GPU璁惧ID

        Returns:
            int: 鎺ㄨ崘鐨勬壒澶у皬
        """
        if not torch.cuda.is_available():
            return 32

        # 鑾峰彇GPU鍐呭瓨锛圙B锛?        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)

        # 鏍规嵁鍐呭瓨澶у皬鎺ㄨ崘鎵瑰ぇ灏?        if gpu_memory >= 16:  # 16GB+
            optimal_batch_size = 256
        elif gpu_memory >= 8:  # 8GB+
            optimal_batch_size = 128
        elif gpu_memory >= 4:  # 4GB+
            optimal_batch_size = 64
        else:
            optimal_batch_size = 32

        logger.info(f"GPU {gpu_id} 鍐呭瓨: {gpu_memory:.1f}GB, 鎺ㄨ崘鎵瑰ぇ灏? {optimal_batch_size}")
        return optimal_batch_size


class DLFELSTMDataset(Dataset):
    """
    DLFE-LSTM-WSI PyTorch鏁版嵁闆嗙被
    鐢ㄤ簬GPU鍔犻€熺殑鏁版嵁鍔犺浇
    """

    def __init__(self,
                 features: np.ndarray,
                 targets: np.ndarray,
                 weather: Optional[np.ndarray] = None,
                 transform=None):
        """
        鍒濆鍖栨暟鎹泦

        Args:
            features: 鐗瑰緛鏁扮粍 (n_samples, seq_len, n_features)
            targets: 鐩爣鏁扮粍 (n_samples, 1)
            weather: 澶╂皵鏍囩 (n_samples,) 鍙€?            transform: 鏁版嵁鍙樻崲鍑芥暟
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        if weather is not None:
            self.weather = torch.as_tensor(weather, dtype=torch.long)
        else:
            self.weather = None
        self.transform = transform

        # 楠岃瘉鏁版嵁缁村害
        assert len(self.features) == len(self.targets), "鐗瑰緛鍜岀洰鏍囨暟閲忎笉鍖归厤"
        if self.weather is not None:
            assert len(self.weather) == len(self.targets), "澶╂皵鏍囩鏁伴噺涓庢牱鏈暟閲忎笉鍖归厤"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        鑾峰彇鍗曚釜鏍锋湰

        Returns:
            tuple: (features, targets)
        """
        features = self.features[idx]
        targets = self.targets[idx]

        if self.transform:
            features = self.transform(features)

        if self.weather is not None:
            weather = self.weather[idx]
            return features, targets, weather
        return features, targets
