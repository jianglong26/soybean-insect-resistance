"""
1.从原始的大豆抗虫tif图像以及对应的gpkgshape文件，数据集中裁剪出来图像
2.根据excel表格特征，生成最终的标签文件dataset_metadata.json文件
3.json中包含labels与裁剪的小的plot图像路径等元数据，方便后续深度学习使用
"""
import numpy as np
import cv2
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box, Polygon
from shapely.affinity import rotate as shapely_rotate
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import json
import pandas as pd
from datetime import datetime, date
import re


def get_rotated_bounding_box(geometries):
    """获取所有几何图形的最小旋转外接矩形"""
    from shapely.ops import unary_union
    combined = unary_union(geometries)
    min_rect = combined.minimum_rotated_rectangle
    coords = np.array(list(min_rect.exterior.coords)[:-1])
    
    dx = coords[1][0] - coords[0][0]
    dy = coords[1][1] - coords[0][1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    edge1_len = np.sqrt(dx**2 + dy**2)
    dx2 = coords[2][0] - coords[1][0]
    dy2 = coords[2][1] - coords[1][1]
    edge2_len = np.sqrt(dx2**2 + dy2**2)
    
    if edge1_len > edge2_len:
        width = edge2_len
        height = edge1_len
    else:
        width = edge1_len
        height = edge2_len
        coords = np.roll(coords, -1, axis=0)
        angle += 90
    
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    
    return min_rect, coords, angle, width, height


def pixel_to_geo(transform, row, col):
    """像素坐标转地理坐标"""
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y


def geo_to_pixel(transform, x, y):
    """地理坐标转像素坐标"""
    row, col = rasterio.transform.rowcol(transform, x, y)
    return row, col


def extract_date_from_folder(folder_name):
    """
    从文件夹名提取日期
    例如: '250123_10m_VCU_CONTROL' -> '2025-01-23'
    """
    match = re.match(r'(\d{6})', folder_name)
    if match:
        date_str = match.group(1)
        # 假设格式是YYMMDD
        year = '20' + date_str[:2]
        month = date_str[2:4]
        day = date_str[4:6]
        return f"{year}-{month}-{day}"
    return None


def extract_growth_stage(folder_name):
    """
    从文件夹名提取生育期关键词
    例如: 'flowering', 'drought' 等
    """
    folder_lower = folder_name.lower()
    keywords = ['flowering', 'drought', 'maturity', 'vegetative']
    for kw in keywords:
        if kw in folder_lower:
            return kw
    return 'unknown'


def normalize_column_name(column_name):
    """标准化列名：压缩空白并去除首尾空格，兼容Excel中的换行/多空格。"""
    return re.sub(r'\s+', ' ', str(column_name)).strip()


def find_column(df_columns, aliases):
    """在DataFrame列中按别名查找，优先精确匹配，再做标准化匹配。"""
    columns = list(df_columns)
    alias_set = {alias for alias in aliases if alias}

    for col in columns:
        if col in alias_set:
            return col

    normalized_map = {normalize_column_name(col): col for col in columns}
    for alias in aliases:
        normalized_alias = normalize_column_name(alias)
        if normalized_alias in normalized_map:
            return normalized_map[normalized_alias]

    return None


def to_json_safe_value(value):
    """将常见的pandas/numpy类型转换为可JSON序列化的Python原生类型。"""
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.strftime('%Y-%m-%d')
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def load_phenotype_data(excel_path):
    """
    加载表型数据（Excel），返回最终标签字典
    结构: {plot_number: {label1: value1, label2: value2, ...}}
    """
    df = pd.read_csv(excel_path) if excel_path.endswith('.csv') else pd.read_excel(excel_path)
    
    # 关键列映射：key为输出到JSON的统一标签名，value为Excel中可能出现的列名别名
    label_aliases = {
        'Plot': ['Plot', 'PLOT'],
        'Genotype': ['Genotype', 'GENOTYPE'],
        'Block': ['Block', 'BLOCK'],
        'Grain Yield - GY (kg/ha)': ['Grain Yield - GY (kg/ha)'],
        'Bug': ['Bug'],
        'Nymph': ['Nymph'],
        'One Hundred Seed Weight (PCS)': ['One Hundred Seed Weight (PCS)'],
        'Healthy Seed Weight (HSW)': ['Healthy Seed Weight (HSW)'],
        'Leaf Retention (FR)': ['Leaf Retention (FR)'],
        'Agronomic Value (VA)': ['Agronomic Value (VA)'],
        'Number Days To R5 (NR5)': ['Number Days To R5 (NR5)'],
        'Number Days To R7 (NR7)': ['Number Days To R7 (NR7)'],
        'Filling Period (PEG)': ['Filling Period (PEG)'],
        'Date Maturity': ['Date Maturity'],
        'Number Days To Maturity (NDM)': ['Number Days To Maturity (NDM)']
    }
    
    # 处理可能的列名变体（Wor1 -> Worm）
    if 'Wor1' in df.columns and 'Worm' not in df.columns:
        df = df.rename(columns={'Wor1': 'Worm'})
    
    # 按别名解析可用列，并统一为规范标签名
    resolved_columns = {}
    for canonical_name, aliases in label_aliases.items():
        matched_col = find_column(df.columns, aliases)
        if matched_col is not None:
            resolved_columns[canonical_name] = matched_col

    available_cols = list(resolved_columns.keys())
    missing_cols = [name for name in label_aliases.keys() if name not in resolved_columns and name != 'Plot']
    
    if 'Plot' not in available_cols:
        print(f"⚠️ 警告: Excel中找不到'Plot'列，无法匹配标签")
        return {}
    
    if missing_cols:
        print(f"⚠️ 提示: Excel中缺少以下列: {', '.join(missing_cols)}")
    
    print(f"  成功识别 {len(available_cols)} 个列: {', '.join(available_cols)}")
    
    # 提取最终标签（每个plot对应Excel中的一行数据）
    plot_labels = {}
    plot_column = resolved_columns['Plot']
    for plot_num in df[plot_column].unique():
        plot_data = df[df[plot_column] == plot_num].iloc[-1]  # 取最后一条记录
        
        labels = {}
        for canonical_name in available_cols:
            if canonical_name == 'Plot':
                continue
            source_col = resolved_columns[canonical_name]
            value = plot_data[source_col]
            # 处理空值：转为None（便于JSON序列化）
            if pd.isna(value) or (isinstance(value, str) and value.strip() == ''):
                labels[canonical_name] = None
            else:
                # 标准化Genotype：移除 (T) 后缀，保持与JSON文件一致
                if canonical_name == 'Genotype' and isinstance(value, str):
                    value = value.replace(' (T)', '').strip()
                labels[canonical_name] = to_json_safe_value(value)
        
        # 合并 Bug 和 Nymph 列：将两列的数值相加
        if 'Bug' in resolved_columns and 'Nymph' in resolved_columns:
            bug_value = labels.get('Bug')
            nymph_value = labels.get('Nymph')
            
            # 将两列的值转换为数值并相加
            bug_count = 0
            if bug_value is not None:
                try:
                    bug_count += float(bug_value)
                except (ValueError, TypeError):
                    # 如果是字符串（虫子名称），计为1
                    bug_count += 1
            
            if nymph_value is not None:
                try:
                    bug_count += float(nymph_value)
                except (ValueError, TypeError):
                    # 如果是字符串（虫子名称），计为1
                    bug_count += 1
            
            # 更新Bug列为合并后的总数，如果总数为0则设为None
            labels['Bug'] = bug_count if bug_count > 0 else None
            # 移除Nymph列，只保留合并后的Bug列
            if 'Nymph' in labels:
                del labels['Nymph']
        
        plot_labels[int(plot_num)] = labels
    
    # 打印前3个plot的映射，用于验证准确性
    print(f"  验证Plot映射（前3个）：")
    for plot_num in sorted(plot_labels.keys())[:3]:
        genotype = plot_labels[plot_num].get('Genotype', 'Unknown')
        print(f"    Plot {plot_num} -> Genotype: {genotype}")
    
    return plot_labels


def process_field_plots_for_dataset(tif_path, gpkg_path, environment, date, 
                                     plot_labels, output_crops_dir, save_visualization=True):
    """
    处理单个时间点的图像，裁剪plots并返回元数据
    
    参数:
        save_visualization: 是否保存带标注的可视化图（默认True）
    
    返回:
        list of dict: [{plot_id, genotype, block, image_path, ...}, ...]
    """
    print(f"  读取GPKG: {os.path.basename(gpkg_path)}")
    gdf = gpd.read_file(gpkg_path)
    
    print(f"  读取TIF: {os.path.basename(tif_path)}")
    with rasterio.open(tif_path) as src:
        image = src.read()
        transform = src.transform
        
        if image.shape[0] >= 3:
            rgb_image = np.transpose(image[:3], (1, 2, 0))
        else:
            rgb_image = np.transpose(image, (1, 2, 0))
        
        rgb_image = ((rgb_image - rgb_image.min()) / 
                     (rgb_image.max() - rgb_image.min()) * 255).astype(np.uint8)
        
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        
        # 获取旋转矩形并透视变换（保持您的原始逻辑）
        min_rect_geo, rect_coords_geo, rotation_angle, rect_width, rect_height = \
            get_rotated_bounding_box(gdf.geometry)
        
        rect_coords_pixel = []
        for x, y in rect_coords_geo:
            r, c = geo_to_pixel(transform, x, y)
            rect_coords_pixel.append([c, r])
        rect_coords_pixel = np.array(rect_coords_pixel, dtype=np.float32)
        
        edge1 = np.linalg.norm(rect_coords_pixel[1] - rect_coords_pixel[0])
        edge2 = np.linalg.norm(rect_coords_pixel[2] - rect_coords_pixel[1])
        
        if edge1 > edge2:
            dst_width = int(edge2)
            dst_height = int(edge1)
            dst_rect = np.array([[0, 0], [0, dst_height], [dst_width, dst_height], [dst_width, 0]], dtype=np.float32)
        else:
            dst_width = int(edge1)
            dst_height = int(edge2)
            dst_rect = np.array([[0, 0], [dst_width, 0], [dst_width, dst_height], [0, dst_height]], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(rect_coords_pixel, dst_rect)
        warped_image = cv2.warpPerspective(rgb_image, M, (dst_width, dst_height),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(255, 255, 255))
        warped_image = cv2.rotate(warped_image, cv2.ROTATE_180)
        
        # 转换plots坐标（保持您的蛇形排序逻辑）
        warped_plots = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Polygon':
                coords = list(geom.exterior.coords)
                warped_coords = []
                for x, y in coords:
                    r, c = geo_to_pixel(transform, x, y)
                    point = np.array([[[c, r]]], dtype=np.float32)
                    transformed = cv2.perspectiveTransform(point, M)
                    tx, ty = transformed[0][0][0], transformed[0][0][1]
                    rotated_x = dst_width - tx
                    rotated_y = dst_height - ty
                    warped_coords.append((rotated_x, rotated_y))
                
                warped_plots.append({
                    'index': idx,
                    'coords': warped_coords
                })
        
        # 蛇形排序（保持您的原始逻辑）
        for plot in warped_plots:
            center_x = np.mean([p[0] for p in plot['coords']])
            center_y = np.mean([p[1] for p in plot['coords']])
            plot['center'] = (center_x, center_y)
        
        y_coords = [p['center'][1] for p in warped_plots]
        y_sorted = sorted(y_coords, reverse=True)
        
        rows = []
        current_row = [y_sorted[0]]
        threshold = 50
        
        for y in y_sorted[1:]:
            if abs(y - current_row[-1]) < threshold:
                current_row.append(y)
            else:
                rows.append(current_row)
                current_row = [y]
        rows.append(current_row)
        
        row_y_means = [np.mean(row) for row in rows]
        
        for plot in warped_plots:
            y = plot['center'][1]
            row_idx = min(range(len(row_y_means)), key=lambda i: abs(row_y_means[i] - y))
            plot['row'] = row_idx
        
        plots_by_row = {}
        for plot in warped_plots:
            row_idx = plot['row']
            if row_idx not in plots_by_row:
                plots_by_row[row_idx] = []
            plots_by_row[row_idx].append(plot)
        
        sorted_plots = []
        for row_idx in sorted(plots_by_row.keys()):
            row_plots = plots_by_row[row_idx]
            if row_idx % 2 == 0:
                row_plots.sort(key=lambda p: p['center'][0])
            else:
                row_plots.sort(key=lambda p: p['center'][0], reverse=True)
            sorted_plots.extend(row_plots)
        
        for new_idx, plot in enumerate(sorted_plots):
            plot['plot_number'] = new_idx + 1
        
        warped_plots = sorted_plots
        
        # 保存可视化图（带标注的plot编号和品种）
        if save_visualization:
            vis_image = warped_image.copy()
            for plot in warped_plots:
                plot_num = plot['plot_number']
                labels = plot_labels.get(plot_num, {})
                genotype = labels.get('Genotype', 'Unknown')
                if isinstance(genotype, str) and genotype != 'Unknown':
                    genotype = genotype.replace(' (T)', '').strip()
                
                # 绘制plot边界
                pts = np.array(plot['coords'], dtype=np.int32)
                cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
                
                # 计算中心点
                center_x = int(plot['center'][0])
                center_y = int(plot['center'][1])
                
                # 准备显示文本
                lines = [str(plot_num), genotype] if genotype != 'Unknown' else [str(plot_num)]
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                line_height = 18
                
                # 计算文本尺寸
                total_height = len(lines) * line_height
                max_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] 
                                for line in lines])
                
                # 绘制背景框
                padding = 5
                bg_x1 = center_x - max_width//2 - padding
                bg_y1 = center_y - total_height//2 - padding
                bg_x2 = center_x + max_width//2 + padding
                bg_y2 = center_y + total_height//2 + padding
                
                cv2.rectangle(vis_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
                cv2.rectangle(vis_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), 1)
                
                # 绘制文本
                y_offset = center_y - total_height//2 + line_height//2
                for i, line in enumerate(lines):
                    text_w, text_h = cv2.getTextSize(line, font, font_scale, thickness)[0]
                    text_x = center_x - text_w//2
                    text_y = y_offset + i * line_height + text_h//2
                    color = (255, 0, 0) if i == 0 else (0, 0, 255)
                    cv2.putText(vis_image, line, (text_x, text_y),
                               font, font_scale, color, thickness)
            
            # 保存可视化图
            vis_dir = os.path.dirname(tif_path)
            vis_filename = f"grid_{environment}_labeled_{date}.png"
            vis_path = os.path.join(vis_dir, vis_filename)
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"  ✓ 可视化图已保存: {vis_filename}")
        
        # 裁剪并保存plots
        print(f"  裁剪 {len(warped_plots)} 个plots...")
        plot_metadata_list = []
        
        for plot in warped_plots:
            plot_num = plot['plot_number']
            
            # 获取该plot的标签
            labels = plot_labels.get(plot_num, {})
            genotype = labels.get('Genotype', 'Unknown')
            # 再次确保Genotype标准化（移除可能的 (T) 后缀）
            if isinstance(genotype, str) and genotype != 'Unknown':
                genotype = genotype.replace(' (T)', '').strip()
            block = labels.get('Block', 0)
            
            # 裁剪区域
            coords = np.array(plot['coords'])
            x_min = max(0, int(coords[:, 0].min()))
            y_min = max(0, int(coords[:, 1].min()))
            x_max = min(warped_image.shape[1], int(coords[:, 0].max()))
            y_max = min(warped_image.shape[0], int(coords[:, 1].max()))
            
            if x_max > x_min and y_max > y_min:
                plot_crop = warped_image[y_min:y_max, x_min:x_max]
                
                # 新的命名规则: {Genotype}_{PlotID}_{Date}.png（品种名空格改为下划线，时间用下划线连接）
                genotype_safe = genotype.replace(' ', '_')  # AS 3730 -> AS_3730
                date_formatted = date.replace('-', '_')  # 2025-01-23 -> 2025_01_23
                filename = f"{genotype_safe}_plot{plot_num}_{date_formatted}.png"
                
                # 保存路径: output_crops_dir/environment/genotype/filename（文件夹名保持原样）
                genotype_dir = os.path.join(output_crops_dir, environment, str(genotype))
                os.makedirs(genotype_dir, exist_ok=True)
                
                save_path = os.path.join(genotype_dir, filename)
                cv2.imwrite(save_path, cv2.cvtColor(plot_crop, cv2.COLOR_RGB2BGR))
                
                # 记录元数据
                relative_path = os.path.join(environment, str(genotype), filename)
                plot_metadata_list.append({
                    'plot_number': int(plot_num),  # 确保是Python int
                    'genotype': genotype,
                    'block': int(block) if not pd.isna(block) else None,  # None代替np.nan
                    'environment': environment,
                    'date': date,
                    'image_path': relative_path
                })
        
        return plot_metadata_list


def create_dataset_metadata(base_dir):
    """
    遍历整个数据集，生成 dataset_metadata.json
    
    结构:
    {
      "V01_B1_sprayed": {
        "genotype": "BR001",
        "block": 1,
        "environment": "sprayed",
        "image_sequence": [
          {"date": "2025-01-23", "path": "sprayed/BR001/BR001_B1_2025-01-23.png"},
          ...
        ],
        "labels": {
          "bug_count": 12,
          "tolerance": 3.5,
          ...
        }
      },
      ...
    }
    """
    orthomosaic_dir = os.path.join(base_dir, "orthomosaic")
    output_crops_dir = os.path.join(base_dir, "dataset", "images")
    output_metadata_path = os.path.join(base_dir, "dataset", "annotations", "dataset_metadata.json")
    
    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
    
    environments = ['control', 'nocontrol']
    dataset_metadata = {}
    
    for env in environments:
        env_dir = os.path.join(orthomosaic_dir, env)
        if not os.path.exists(env_dir):
            print(f"⚠️ 跳过不存在的环境: {env_dir}")
            continue
        
        # 加载表型数据
        excel_path = os.path.join(env_dir, f"VCP_{env}_24_25_Anhumas_BC.xlsx")
        if not os.path.exists(excel_path):
            print(f"⚠️ 找不到Excel: {excel_path}")
            plot_labels = {}
        else:
            print(f"\n加载表型数据: {excel_path}")
            plot_labels = load_phenotype_data(excel_path)
            print(f"  加载了 {len(plot_labels)} 个plots的标签")
        
        # 查找所有时间点文件夹
        keyword = env.upper()
        time_folders = []
        for item in os.listdir(env_dir):
            item_path = os.path.join(env_dir, item)
            if os.path.isdir(item_path) and keyword in item.upper():
                time_folders.append(item)
        
        time_folders.sort()
        print(f"\n找到 {len(time_folders)} 个时间点: {time_folders}")
        
        # 处理每个时间点
        all_time_metadata = []
        for folder in time_folders:
            print(f"\n处理: {folder}")
            folder_path = os.path.join(env_dir, folder)
            
            tif_path = os.path.join(folder_path, "odm_orthophoto_modified.tif")

            # 优先使用新的标签文件命名，同时兼容旧命名。
            gpkg_candidates = [
                os.path.join(folder_path, f"{env}.gpkg"),
                # os.path.join(folder_path, f"grid_{env}.gpkg"),
            ]
            gpkg_path = next((p for p in gpkg_candidates if os.path.exists(p)), None)

            if not os.path.exists(tif_path) or gpkg_path is None:
                print(f"  ⚠️ 跳过（缺少文件）")
                continue
            
            # 提取日期
            date = extract_date_from_folder(folder)
            if not date:
                print(f"  ⚠️ 无法从文件夹名提取日期，跳过")
                continue
            
            # 处理并裁剪
            time_metadata = process_field_plots_for_dataset(
                tif_path, gpkg_path, env, date, plot_labels, output_crops_dir
            )
            all_time_metadata.extend(time_metadata)
            print(f"  ✓ 裁剪了 {len(time_metadata)} 个plots")
        
        # 按plot_number组织数据
        print(f"\n组织 {env} 环境的元数据...")
        for plot_num in range(1, 91):  # 假设有90个plots
            plot_key = f"P{plot_num:02d}_{env}"
            
            # 收集该plot的所有时序图像
            plot_images = [m for m in all_time_metadata if m['plot_number'] == plot_num]
            
            if not plot_images:
                continue
            
            # 按日期排序
            plot_images.sort(key=lambda x: x['date'])
            
            # 获取标签（来自plot_labels）
            labels = plot_labels.get(plot_num, {})
            
            # 转换labels中的numpy类型为Python原生类型
            clean_labels = {}
            for key, value in labels.items():
                clean_labels[key] = to_json_safe_value(value)
            
            dataset_metadata[plot_key] = {
                'genotype': plot_images[0]['genotype'],
                'block': plot_images[0]['block'],
                'environment': env,
                'image_sequence': [
                    {'date': img['date'], 'path': img['image_path']}
                    for img in plot_images
                ],
                'labels': clean_labels
            }
    
    # 保存元数据
    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 数据集元数据已保存到: {output_metadata_path}")
    print(f"✓ 总共 {len(dataset_metadata)} 个样本")
    print(f"✓ 图像保存位置: {output_crops_dir}")
    
    return dataset_metadata


if __name__ == "__main__":
    base_dir = "E:/projects/Agriculture/SoybeanBreeding/AnhumasPiracicaba"
    
    print("="*60)
    print("开始准备大豆抗虫数据集")
    print("="*60)
    
    dataset_metadata = create_dataset_metadata(base_dir)
    
    print("\n" + "="*60)
    print("数据集准备完成！")
    print("="*60)
    print("\n生成的文件:")
    print(f"  1. dataset/images/          # 裁剪的plot图像")
    print(f"     ├── control/genotype/")
    print(f"     └── nocontrol/genotype/")
    print(f"  2. dataset/annotations/dataset_metadata.json  # 元数据")
    print("\n下一步:")
    print("  - 使用此元数据进行数据集划分")
    print("  - 创建PyTorch DataLoader")
    print("  - 训练DINOv3模型")
    print("="*60)



   