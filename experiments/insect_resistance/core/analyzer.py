"""
Core Analyzer Module
抗虫性分析核心模块 - 多模态特征分析器
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score


class MultiModalInsectResistanceAnalyzer:
    """多模态抗虫分析器"""
    
    def __init__(self, feature_type='dinov3'):
        """
        初始化
        
        Args:
            feature_type: 'dinov3', 'vi' (vegetation indices), 'fusion'
        """
        self.feature_type = feature_type
        # 模块根目录: experiments/insect_resistance/
        self.module_dir = Path(__file__).parent.parent
        # 项目根目录
        self.project_root = self.module_dir.parent.parent
        
        # 数据和特征路径（使用项目根目录）
        self.data_dir = self.project_root / 'AnhumasPiracicaba' / 'dataset' / 'annotations'
        self.feature_dir = self.project_root / 'outputs' / 'features'
        
        # 输出目录（保存在模块内部）
        if feature_type == 'dinov3':
            self.output_dir = self.module_dir / 'outputs' / 'results' / 'dinov3'
        elif feature_type == 'vi':
            self.output_dir = self.module_dir / 'outputs' / 'results' / 'vi'
        elif feature_type == 'fusion':
            self.output_dir = self.module_dir / 'outputs' / 'results' / 'fusion'
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Initializing Analyzer - Feature Type: {feature_type.upper()}")
        print(f"{'='*80}")
        
        # 加载数据
        self._load_data()
        self._load_features()
    
    def _load_data(self):
        """加载元数据"""
        with open(self.data_dir / 'dataset_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 分离control和nocontrol
        self.control_df = []
        self.nocontrol_df = []
        
        for plot_id, info in metadata.items():
            labels = info.get('labels', {})
            data = {
                'plot_id': plot_id,
                'genotype': info['genotype'],
                'block': info['block'],
                'bug': labels.get('Bug'),
                'leaf_retention': labels.get('Leaf Retention (FR)'),
                'agronomic_value': labels.get('Agronomic Value (VA)'),
                'grain_yield': labels.get('Grain Yield - GY (kg/ha)'),
                'seed_weight': labels.get('Healthy Seed Weight (HSW)'),
                'filling_period': labels.get('Filling Period (PEG)'),
                'ndm': labels.get('Number Days To Maturity (NDM)', labels.get('NDM'))
            }
            
            if info['environment'] == 'control':
                self.control_df.append(data)
            else:
                self.nocontrol_df.append(data)
        
        self.control_df = pd.DataFrame(self.control_df)
        self.nocontrol_df = pd.DataFrame(self.nocontrol_df)
        
        print(f"✓ Data loaded: {len(self.control_df)} control, {len(self.nocontrol_df)} nocontrol samples")
    
    def _load_features(self):
        """加载特征"""
        if self.feature_type == 'dinov3':
            self._load_dinov3_features()
        elif self.feature_type == 'vi':
            self._load_vi_features()
        elif self.feature_type == 'fusion':
            self._load_dinov3_features()
            self._load_vi_features()
            self._fuse_features()
    
    def _load_dinov3_features(self):
        """加载DINOv3特征"""
        control_path = self.feature_dir / 'dinov3' / 'control_features.pkl'
        nocontrol_path = self.feature_dir / 'dinov3' / 'nocontrol_features.pkl'
        
        with open(control_path, 'rb') as f:
            self.dinov3_control = pickle.load(f)
        with open(nocontrol_path, 'rb') as f:
            self.dinov3_nocontrol = pickle.load(f)
        
        print(f"✓ DINOv3 features loaded: {len(self.dinov3_control)} control, {len(self.dinov3_nocontrol)} nocontrol")
    
    def _load_vi_features(self):
        """加载植被指数特征"""
        control_path = self.feature_dir / 'vegetation_indices' / 'control_features.pkl'
        nocontrol_path = self.feature_dir / 'vegetation_indices' / 'nocontrol_features.pkl'
        
        with open(control_path, 'rb') as f:
            self.vi_control = pickle.load(f)
        with open(nocontrol_path, 'rb') as f:
            self.vi_nocontrol = pickle.load(f)
        
        print(f"✓ Vegetation indices loaded: {len(self.vi_control)} control, {len(self.vi_nocontrol)} nocontrol")
    
    def _fuse_features(self):
        """融合DINOv3和植被指数特征"""
        self.fused_control = {}
        self.fused_nocontrol = {}
        
        # 融合control特征
        for plot_id in self.dinov3_control.keys():
            if plot_id in self.vi_control:
                dinov3_feat = self.dinov3_control[plot_id]['features']  # (8, 384)
                vi_feat = self.vi_control[plot_id]['features']  # (8, 24)
                
                # 时序拼接
                fused = np.concatenate([dinov3_feat, vi_feat], axis=1)  # (8, 408)
                
                self.fused_control[plot_id] = {
                    'genotype': self.dinov3_control[plot_id]['genotype'],
                    'labels': self.dinov3_control[plot_id]['labels'],
                    'features': fused
                }
        
        # 融合nocontrol特征
        for plot_id in self.dinov3_nocontrol.keys():
            if plot_id in self.vi_nocontrol:
                dinov3_feat = self.dinov3_nocontrol[plot_id]['features']
                vi_feat = self.vi_nocontrol[plot_id]['features']
                
                fused = np.concatenate([dinov3_feat, vi_feat], axis=1)
                
                self.fused_nocontrol[plot_id] = {
                    'genotype': self.dinov3_nocontrol[plot_id]['genotype'],
                    'labels': self.dinov3_nocontrol[plot_id]['labels'],
                    'features': fused
                }
        
        print(f"✓ Features fused: {len(self.fused_control)} control, {len(self.fused_nocontrol)} nocontrol")
        sample_feat = list(self.fused_nocontrol.values())[0]['features']
        print(f"  Fused feature shape: {sample_feat.shape} (DINOv3: 384 + VI: 24 = 408)")
    
    def get_features_for_training(self):
        """获取训练用的特征字典"""
        if self.feature_type == 'dinov3':
            return self.dinov3_nocontrol
        elif self.feature_type == 'vi':
            return self.vi_nocontrol
        elif self.feature_type == 'fusion':
            return self.fused_nocontrol
    
    def calculate_resistance_indices(self):
        """计算抗性指标"""
        print(f"\n{'='*80}")
        print("Calculating Resistance Indices")
        print(f"{'='*80}")
        
        genotypes = sorted(self.control_df['genotype'].unique())
        resistance_data = []
        
        for genotype in genotypes:
            control_subset = self.control_df[self.control_df['genotype'] == genotype]
            nocontrol_subset = self.nocontrol_df[self.nocontrol_df['genotype'] == genotype]
            
            data_point = {
                'genotype': genotype,
                'n_replicates': len(control_subset)
            }
            
            # Bug数量
            control_bugs = control_subset['bug'].dropna()
            nocontrol_bugs = nocontrol_subset['bug'].dropna()
            
            data_point['control_bug_mean'] = control_bugs.mean() if len(control_bugs) > 0 else np.nan
            data_point['nocontrol_bug_mean'] = nocontrol_bugs.mean() if len(nocontrol_bugs) > 0 else np.nan
            data_point['bug_increase'] = data_point['nocontrol_bug_mean'] - data_point['control_bug_mean']
            
            # 产量相关
            control_yield = control_subset['grain_yield'].mean()
            nocontrol_yield = nocontrol_subset['grain_yield'].mean()
            data_point['control_yield'] = control_yield
            data_point['nocontrol_yield'] = nocontrol_yield
            
            # 产量损失率
            data_point['yield_loss_rate'] = (control_yield - nocontrol_yield) / control_yield * 100
            
            # 耐受指数
            data_point['tolerance_index'] = 100 - abs(data_point['yield_loss_rate'])
            
            # 其他农艺性状
            for trait in ['leaf_retention', 'agronomic_value', 'seed_weight', 'filling_period']:
                data_point[f'control_{trait}'] = control_subset[trait].mean()
                data_point[f'nocontrol_{trait}'] = nocontrol_subset[trait].mean()
            
            resistance_data.append(data_point)
        
        self.resistance_df = pd.DataFrame(resistance_data)
        
        # 计算综合评分
        self._calculate_comprehensive_score()
        
        # 保存结果
        save_path = self.output_dir / 'resistance_indices.csv'
        self.resistance_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ Resistance indices saved: {save_path}")
        
        return self.resistance_df
    
    def _calculate_comprehensive_score(self):
        """计算两种综合抗性评分
        
        score_without_bug: 不含虫子指标的基础评分（所有30个品种都有）
        score_with_bug: 含虫子指标的完整评分（使用实测或预测的虫子数据）
        """
        df = self.resistance_df.copy()
        df['has_bug_data'] = df['nocontrol_bug_mean'].notna()
        
        # ========== 排名1: 不含虫子指标的基础排名 ==========
        # 只使用所有品种都有的指标：叶片保持、产量、种子重量、农艺价值
        
        scaler = StandardScaler()
        
        # 准备基础指标（不含虫子）
        base_scores = pd.DataFrame({
            'leaf_retention': df['nocontrol_leaf_retention'],
            'yield': df['nocontrol_yield'],  # 直接用产量，而非耐受指数
            'seed_weight': df['nocontrol_seed_weight'],
            'agronomic_value': df['nocontrol_agronomic_value']
        })
        
        # 归一化基础指标
        base_scores_normalized = pd.DataFrame()
        for col in base_scores.columns:
            if base_scores[col].notna().sum() > 0:
                valid_idx = base_scores[col].notna()
                normalized_col = np.zeros(len(base_scores))
                normalized_col[valid_idx] = scaler.fit_transform(
                    base_scores.loc[valid_idx, col].values.reshape(-1, 1)
                ).flatten()
                base_scores_normalized[col] = normalized_col
        
        # 基础权重（不含虫子）
        base_weights = {
            'leaf_retention': 0.35,  # 叶片保持率 - 35%
            'yield': 0.45,           # 产量 - 45%
            # 'seed_weight': 0.20,     # 种子重量 - 不使用
            'agronomic_value': 0.20  # 农艺价值 - 20%
        }
        
        df['score_without_bug'] = 0
        for col, weight in base_weights.items():
            if col in base_scores_normalized.columns:
                df['score_without_bug'] += base_scores_normalized[col] * weight
        
        # 归一化到0-100
        if df['score_without_bug'].std() > 0:
            df['score_without_bug'] = (
                (df['score_without_bug'] - df['score_without_bug'].min()) / 
                (df['score_without_bug'].max() - df['score_without_bug'].min()) * 100
            )
        
        # ========== 排名2: 含虫子指标的完整排名（先用实测，后面会用预测补充） ==========
        # 这里先计算有虫子数据的品种，预测数据会在后面补充
        
        full_scores = []
        for _, row in df.iterrows():
            score_components = {}
            
            # 虫害抗性（先用实测，如果有的话）
            if not np.isnan(row['nocontrol_bug_mean']):
                score_components['bug_resistance'] = -row['nocontrol_bug_mean']  # 负号：虫越少越好
            
            # 其他指标
            score_components['leaf_retention'] = row['nocontrol_leaf_retention']
            score_components['yield'] = row['nocontrol_yield']
            score_components['seed_weight'] = row['nocontrol_seed_weight']
            score_components['agronomic_value'] = row['nocontrol_agronomic_value']
            
            full_scores.append(score_components)
        
        full_scores_df = pd.DataFrame(full_scores)
        
        # 归一化完整指标
        full_scores_normalized = pd.DataFrame()
        for col in full_scores_df.columns:
            if full_scores_df[col].notna().sum() > 0:
                valid_idx = full_scores_df[col].notna()
                normalized_col = np.zeros(len(full_scores_df))
                normalized_col[valid_idx] = scaler.fit_transform(
                    full_scores_df.loc[valid_idx, col].values.reshape(-1, 1)
                ).flatten()
                full_scores_normalized[col] = normalized_col
                full_scores_normalized.loc[~valid_idx, col] = np.nan
        
        # 完整权重（含虫子） - 调整后的权重分配
        full_weights = {
            'bug_resistance': 0.25,      # 虫害密度 - 25%
            'leaf_retention': 0.30,      # 叶片保持率 - 30%
            'yield': 0.40,               # 产量 - 40%
            # 'seed_weight': 0.15,         # 种子重量 - 不使用
            'agronomic_value': 0.05      # 农艺价值 - 5%
        }
        
        # 暂时不计算score_with_bug，等预测数据补充后再一起计算
        # 这里只保存归一化后的指标，供后续使用
        df['score_with_bug'] = np.nan
        df['bug_source'] = 'none'  # 标记虫子数据来源：'actual', 'predicted', 'none'
        
        # 标记哪些有实测虫子数据
        for idx, row in df.iterrows():
            if row['has_bug_data']:
                df.at[idx, 'bug_source'] = 'actual'
        
        # 保存归一化后的指标供后续使用
        self._full_scores_normalized = full_scores_normalized
        self._full_weights = full_weights
        
        # 保留原来的resistance_score字段，设为score_without_bug（向后兼容）
        df['resistance_score'] = df['score_without_bug']
        
        self.resistance_df = df
    
    def complete_score_with_predictions(self, resistance_df, genotype_summary):
        """使用预测的bug数据补充完整score_with_bug
        
        重新计算所有30个品种的score_with_bug（实测+预测虫子数据一起归一化）
        """
        print(f"\n  ⭐ Completing score_with_bug using {self.feature_type.upper()} predicted bugs...")
        
        df = resistance_df.copy()
        
        # 将预测的bug数据合并到resistance_df
        pred_dict = dict(zip(genotype_summary['genotype'], genotype_summary['predicted_bug']))
        df['predicted_bug_mean'] = df['genotype'].map(pred_dict)
        
        # 标记虫子数据来源
        for idx, row in df.iterrows():
            if row['has_bug_data']:
                df.at[idx, 'bug_source'] = 'actual'
            else:
                df.at[idx, 'bug_source'] = 'predicted'
        
        # ⭐ 关键：重新收集所有品种的完整指标（包括实测和预测的bug）
        scaler = StandardScaler()
        
        all_scores = []
        for _, row in df.iterrows():
            score_components = {}
            
            # Bug数据：有实测用实测，无实测用预测
            if row['has_bug_data']:
                score_components['bug_resistance'] = -row['nocontrol_bug_mean']
            else:
                score_components['bug_resistance'] = -row['predicted_bug_mean']
            
            # 其他指标
            score_components['leaf_retention'] = row['nocontrol_leaf_retention']
            score_components['yield'] = row['nocontrol_yield']
            score_components['seed_weight'] = row['nocontrol_seed_weight']
            score_components['agronomic_value'] = row['nocontrol_agronomic_value']
            
            all_scores.append(score_components)
        
        all_scores_df = pd.DataFrame(all_scores)
        
        # 重新归一化所有指标（包括混合的bug数据）
        all_scores_normalized = pd.DataFrame()
        for col in all_scores_df.columns:
            if all_scores_df[col].notna().sum() > 0:
                valid_idx = all_scores_df[col].notna()
                normalized_col = np.zeros(len(all_scores_df))
                normalized_col[valid_idx] = scaler.fit_transform(
                    all_scores_df.loc[valid_idx, col].values.reshape(-1, 1)
                ).flatten()
                all_scores_normalized[col] = normalized_col
        
        # 完整权重（含虫子） - 调整后的权重分配
        full_weights = {
            'bug_resistance': 0.25,      # 虫害密度 - 25%
            'leaf_retention': 0.30,      # 叶片保持率 - 30%
            'yield': 0.40,               # 产量 - 40%
            # 'seed_weight': 0.15,         # 种子重量 - 不使用
            'agronomic_value': 0.05      # 农艺价值 - 5%
        }
        
        # 计算所有品种的score_with_bug
        df['score_with_bug'] = 0
        for col, weight in full_weights.items():
            if col in all_scores_normalized.columns:
                df['score_with_bug'] += all_scores_normalized[col] * weight
        
        # 归一化到0-100
        if df['score_with_bug'].std() > 0:
            df['score_with_bug'] = (
                (df['score_with_bug'] - df['score_with_bug'].min()) / 
                (df['score_with_bug'].max() - df['score_with_bug'].min()) * 100
            )
        
        n_actual = (df['bug_source'] == 'actual').sum()
        n_predicted = (df['bug_source'] == 'predicted').sum()
        print(f"  ✓ Score completed: {n_actual} with actual bugs, {n_predicted} with predicted bugs")
        
        return df
    
    def predict_bug_from_features(self):
        """基于特征预测虫害"""
        print(f"\n{'='*80}")
        print(f"Bug Prediction from {self.feature_type.upper()} Features")
        print(f"{'='*80}")
        
        features_dict = self.get_features_for_training()
        
        # 准备训练数据
        X_train = []
        y_train = []
        plot_ids_train = []
        
        for plot_id, features_data in features_dict.items():
            genotype = features_data['genotype']
            bug_count = features_data['labels']['Bug']
            
            if bug_count is not None:
                features = features_data['features'].mean(axis=0)
                X_train.append(features)
                y_train.append(bug_count)
                plot_ids_train.append(plot_id)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        
        # 训练模型
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 交叉验证
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=5, scoring='neg_mean_absolute_error'
        )
        cv_mae = -cv_scores
        
        print(f"\nCross-validation MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
        
        # 预测所有样本
        predictions = []
        genotype_predictions = {}
        
        for plot_id, features_data in features_dict.items():
            genotype = features_data['genotype']
            bug_count = features_data['labels']['Bug']
            features = features_data['features'].mean(axis=0).reshape(1, -1)
            
            pred_bug = model.predict(features)[0]
            
            predictions.append({
                'plot_id': plot_id,
                'genotype': genotype,
                'predicted_bug': pred_bug,
                'true_bug': bug_count if bug_count is not None else np.nan,
                'has_label': bug_count is not None
            })
            
            if genotype not in genotype_predictions:
                genotype_predictions[genotype] = {
                    'predicted_bugs': [],
                    'true_bugs': [],
                    'has_label': False
                }
            
            genotype_predictions[genotype]['predicted_bugs'].append(pred_bug)
            if bug_count is not None:
                genotype_predictions[genotype]['true_bugs'].append(bug_count)
                genotype_predictions[genotype]['has_label'] = True
        
        # 按品种汇总
        genotype_summary = []
        for genotype, data in genotype_predictions.items():
            summary = {
                'genotype': genotype,
                'predicted_bug': np.mean(data['predicted_bugs']),
                'true_bug': np.mean(data['true_bugs']) if data['true_bugs'] else np.nan,
                'has_label': data['has_label']
            }
            genotype_summary.append(summary)
        
        genotype_summary = pd.DataFrame(genotype_summary).sort_values('predicted_bug')
        
        # 保存结果
        prediction_df = pd.DataFrame(predictions)
        save_path = self.output_dir / 'bug_predictions.csv'
        prediction_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ Predictions saved: {save_path}")
        
        return prediction_df, genotype_summary, model, cv_mae
    
    def predict_multiple_indicators(self):
        """基于特征预测多个指标（叶片保持率、产量、种子重量、农艺价值）"""
        print(f"\n{'='*80}")
        print(f"Multi-Indicator Prediction from {self.feature_type.upper()} Features")
        print(f"{'='*80}")
        
        features_dict = self.get_features_for_training()
        
        # 定义要预测的指标（来自NoControl环境）
        indicators = {
            'leaf_retention': 'Leaf Retention Rate (%)',
            'grain_yield': 'Grain Yield (kg/ha)',
            'seed_weight': 'Seed Weight (g)',
            'agronomic_value': 'Agronomic Value (score)'
        }
        
        # 准备训练数据：使用NoControl环境的数据
        X_train = []
        y_train_dict = {ind: [] for ind in indicators}
        genotype_train = []
        
        for plot_id, features_data in features_dict.items():
            genotype = features_data['genotype']
            features = features_data['features'].mean(axis=0)
            
            # 从nocontrol_df获取对应的标签
            plot_data = self.nocontrol_df[self.nocontrol_df['plot_id'] == plot_id]
            if len(plot_data) > 0:
                X_train.append(features)
                genotype_train.append(genotype)
                
                # 收集各指标的真实值
                for ind in indicators:
                    y_train_dict[ind].append(plot_data[ind].values[0])
        
        X_train = np.array(X_train)
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        
        # 为每个指标训练模型并预测
        all_predictions = {}
        cv_results = {}
        models = {}
        
        for ind_key, ind_name in indicators.items():
            print(f"\n{'─'*60}")
            print(f"Predicting: {ind_name}")
            print(f"{'─'*60}")
            
            y_train = np.array(y_train_dict[ind_key])
            
            # 过滤NaN值
            valid_mask = ~np.isnan(y_train)
            X_train_valid = X_train[valid_mask]
            y_train_valid = y_train[valid_mask]
            
            if len(y_train_valid) < 5:
                print(f"  ⚠ 警告: 有效样本数太少({len(y_train_valid)})，跳过此指标")
                continue
            
            print(f"  Valid samples: {len(y_train_valid)}/{len(y_train)}")
            
            # 训练模型
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_valid, y_train_valid)
            models[ind_key] = model
            
            # 交叉验证
            cv_scores = cross_val_score(
                model, X_train_valid, y_train_valid, 
                cv=5, scoring='neg_mean_absolute_error'
            )
            cv_mae = -cv_scores
            cv_results[ind_key] = {
                'mae_mean': cv_mae.mean(),
                'mae_std': cv_mae.std()
            }
            
            print(f"  CV MAE: {cv_mae.mean():.3f} ± {cv_mae.std():.3f}")
            
            # 预测所有样本（NoControl环境）
            predictions = []
            genotype_predictions = {}
            
            for plot_id, features_data in features_dict.items():
                genotype = features_data['genotype']
                features = features_data['features'].mean(axis=0).reshape(1, -1)
                
                pred_value = model.predict(features)[0]
                
                # 获取真实值
                plot_data = self.nocontrol_df[self.nocontrol_df['plot_id'] == plot_id]
                true_value = plot_data[ind_key].values[0] if len(plot_data) > 0 else np.nan
                has_label = len(plot_data) > 0
                
                predictions.append({
                    'plot_id': plot_id,
                    'genotype': genotype,
                    f'predicted_{ind_key}': pred_value,
                    f'true_{ind_key}': true_value,
                    'has_label': has_label
                })
                
                if genotype not in genotype_predictions:
                    genotype_predictions[genotype] = {
                        'predicted': [],
                        'true': [],
                        'has_label': False
                    }
                
                genotype_predictions[genotype]['predicted'].append(pred_value)
                if has_label:
                    genotype_predictions[genotype]['true'].append(true_value)
                    genotype_predictions[genotype]['has_label'] = True
            
            # 按品种汇总
            genotype_summary = []
            for genotype, data in genotype_predictions.items():
                summary = {
                    'genotype': genotype,
                    f'predicted_{ind_key}': np.mean(data['predicted']),
                    f'true_{ind_key}': np.mean(data['true']) if data['true'] else np.nan,
                    'has_label': data['has_label']
                }
                genotype_summary.append(summary)
            
            all_predictions[ind_key] = {
                'predictions': pd.DataFrame(predictions),
                'genotype_summary': pd.DataFrame(genotype_summary)
            }
        
        # 合并所有预测结果
        combined_genotype_summary = all_predictions[list(indicators.keys())[0]]['genotype_summary'][['genotype', 'has_label']]
        
        for ind_key in indicators:
            summary = all_predictions[ind_key]['genotype_summary']
            combined_genotype_summary = combined_genotype_summary.merge(
                summary[['genotype', f'predicted_{ind_key}', f'true_{ind_key}']],
                on='genotype',
                how='outer'
            )
        
        # 保存结果
        save_path = self.output_dir / 'multi_indicator_predictions.csv'
        combined_genotype_summary.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ Multi-indicator predictions saved: {save_path}")
        
        return combined_genotype_summary, models, cv_results


