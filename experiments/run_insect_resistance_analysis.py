"""
Insect Resistance Analysis - Main Entry Point
抗虫性分析主入口
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from insect_resistance.workflows import run_prediction_experiments
from insect_resistance.visualization import (
    analyze_feature_vs_yield_timeseries,
    create_comprehensive_ranking_visualization,
)
from insect_resistance.visualization.comparison_plots import visualize_genotype_images
from insect_resistance.visualization.ideal_zone_analysis import visualize_single_feature_ideal_zone, analyze_ideal_zone_genotypes


def export_cross_feature_ranking_summary(project_root):
    """Cross-feature export with 24-view appearance frequencies and strict intersections."""
    pd = __import__('pandas')
    np = __import__('numpy')
    plt = __import__('matplotlib.pyplot', fromlist=['pyplot'])

    base = Path(project_root) / 'experiments' / 'insect_resistance' / 'outputs' / 'results'
    cross_dir = base / 'cross_feature_analysis'
    cross_dir.mkdir(parents=True, exist_ok=True)

    feature_types = ['dinov3', 'vi']
    view_order = ['t1', 't2', 't3', 't4', 't5', 't6']

    def _save_png_pdf(path):
        plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.06, facecolor='white')
        try:
            plt.savefig(path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', pad_inches=0.06, facecolor='white')
        except PermissionError as ex:
            print(f"Warning: skip locked PDF file {path.with_suffix('.pdf').name}: {ex}")

    # Method 1 still exported.
    score_parts = []
    for ft in feature_types:
        fp = base / ft / f'ranking_by_score_{ft}.csv'
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        keep = [c for c in ['genotype', 'composite_score', 'rank_by_score'] if c in df.columns]
        tmp = df[keep].copy().rename(columns={
            'composite_score': f'composite_score_{ft}',
            'rank_by_score': f'rank_by_score_{ft}',
        })
        score_parts.append(tmp)
    if score_parts:
        score_merged = score_parts[0]
        for part in score_parts[1:]:
            score_merged = score_merged.merge(part, on='genotype', how='outer')
        rank_cols = [c for c in score_merged.columns if c.startswith('rank_by_score_')]
        if rank_cols:
            score_merged['avg_rank_by_score'] = score_merged[rank_cols].mean(axis=1)
            score_merged = score_merged.sort_values('avg_rank_by_score')
        score_merged.to_csv(cross_dir / 'ranking_by_score_all_features.csv', index=False, encoding='utf-8-sig')

    # Load per-view flags for both features.
    feature_data = {}
    for ft in feature_types:
        fp = base / ft / f'feature_vs_yield_timeseries_data_{ft}.csv'
        if not fp.exists():
            raise FileNotFoundError(f'Missing file: {fp}. Please run time-series analysis first.')
        df = pd.read_csv(fp)
        cols = ['genotype'] + [f'yield_r8_view_{v}' for v in view_order] + [f'gain_r8_view_{v}' for v in view_order]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f'Missing R8 view columns in {fp.name}: {missing}')
        sdf = df[cols].copy()
        for c in cols[1:]:
            sdf[c] = sdf[c].fillna(0).astype(int)
        feature_data[ft] = sdf

    merged = feature_data['dinov3'].merge(feature_data['vi'], on='genotype', suffixes=('_dinov3', '_vi'))
    out = pd.DataFrame({'genotype': merged['genotype']})

    # 12-view appearance per metric + strict 6-view inter-feature intersections.
    yield_cols_12 = []
    gain_cols_12 = []
    for ft in feature_types:
        for v in view_order:
            ycol = f'yield_r8_{ft}_{v}'
            gcol = f'gain_r8_{ft}_{v}'
            out[ycol] = merged[f'yield_r8_view_{v}_{ft}'].astype(int)
            out[gcol] = merged[f'gain_r8_view_{v}_{ft}'].astype(int)
            yield_cols_12.append(ycol)
            gain_cols_12.append(gcol)

    for v in view_order:
        out[f'cross_yield_intersection_{v}'] = ((out[f'yield_r8_dinov3_{v}'] == 1) & (out[f'yield_r8_vi_{v}'] == 1)).astype(int)
        out[f'cross_gain_intersection_{v}'] = ((out[f'gain_r8_dinov3_{v}'] == 1) & (out[f'gain_r8_vi_{v}'] == 1)).astype(int)
        out[f'cross_dual_intersection_{v}'] = ((out[f'cross_yield_intersection_{v}'] == 1) & (out[f'cross_gain_intersection_{v}'] == 1)).astype(int)

    out['count_yield_r8_12views'] = out[yield_cols_12].sum(axis=1)
    out['count_gain_r8_12views'] = out[gain_cols_12].sum(axis=1)
    out['count_r8_overall_24views'] = out['count_yield_r8_12views'] + out['count_gain_r8_12views']

    out['prob_yield_r8_12views'] = out['count_yield_r8_12views'] / 12.0
    out['prob_gain_r8_12views'] = out['count_gain_r8_12views'] / 12.0
    out['prob_r8_overall_24views'] = out['count_r8_overall_24views'] / 24.0

    inter_y_cols = [f'cross_yield_intersection_{v}' for v in view_order]
    inter_g_cols = [f'cross_gain_intersection_{v}' for v in view_order]
    inter_d_cols = [f'cross_dual_intersection_{v}' for v in view_order]
    out['count_cross_yield_intersection_6views'] = out[inter_y_cols].sum(axis=1)
    out['count_cross_gain_intersection_6views'] = out[inter_g_cols].sum(axis=1)
    out['count_cross_dual_intersection_6views'] = out[inter_d_cols].sum(axis=1)
    out['prob_cross_yield_intersection_6views'] = out['count_cross_yield_intersection_6views'] / 6.0
    out['prob_cross_gain_intersection_6views'] = out['count_cross_gain_intersection_6views'] / 6.0
    out['prob_cross_dual_intersection_6views'] = out['count_cross_dual_intersection_6views'] / 6.0

    out['rank_yield_r8_12views'] = out['prob_yield_r8_12views'].rank(ascending=False, method='min').astype(int)
    out['rank_gain_r8_12views'] = out['prob_gain_r8_12views'].rank(ascending=False, method='min').astype(int)
    out['rank_overall_r8_24views'] = out['prob_r8_overall_24views'].rank(ascending=False, method='min').astype(int)

    out = out.sort_values(['rank_overall_r8_24views', 'rank_yield_r8_12views', 'rank_gain_r8_12views'])
    out.to_csv(cross_dir / 'r8_cross_feature_comprehensive.csv', index=False, encoding='utf-8-sig')
    out.sort_values('rank_yield_r8_12views').to_csv(cross_dir / 'ranking_by_yield_r8_all_features.csv', index=False, encoding='utf-8-sig')
    out.sort_values('rank_gain_r8_12views').to_csv(cross_dir / 'ranking_by_gain_r8_all_features.csv', index=False, encoding='utf-8-sig')
    out.sort_values('rank_overall_r8_24views').to_csv(cross_dir / 'ranking_by_intersection_r8_all_features.csv', index=False, encoding='utf-8-sig')

    robust_threshold = 0.50
    robust_df = out[(out['prob_yield_r8_12views'] >= robust_threshold) & (out['prob_gain_r8_12views'] >= robust_threshold)].copy()
    robust_df = robust_df.sort_values('prob_r8_overall_24views', ascending=False)
    robust_df.to_csv(cross_dir / 'robust_candidates_intersection.csv', index=False, encoding='utf-8-sig')

    # Heatmaps (Yield + Gain), keep these for diagnostic transparency.
    plot_df_y = out.sort_values('prob_yield_r8_12views', ascending=False).reset_index(drop=True)
    mat_y = plot_df_y[yield_cols_12].to_numpy()
    fig_y, ax_y = plt.subplots(figsize=(15, 9))
    im = ax_y.imshow(mat_y, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    ax_y.set_yticks(np.arange(len(plot_df_y)))
    ax_y.set_yticklabels(plot_df_y['genotype'], fontsize=8)
    xlabels = [f'DINO-{v.upper()}' for v in view_order] + [f'VI-{v.upper()}' for v in view_order]
    ax_y.set_xticks(np.arange(len(xlabels)))
    ax_y.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=8)
    ax_y.set_title('Yield-R8 Presence Heatmap Across 12 Views (DINOV3 + VI)', fontsize=13, fontweight='bold')
    ax_y.set_xlabel('Feature-View', fontsize=11, fontweight='bold')
    ax_y.set_ylabel('Genotype (sorted by Yield-R8 frequency)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax_y, fraction=0.02, pad=0.01)
    plt.tight_layout()
    _save_png_pdf(cross_dir / 'yield_r8_cross_feature_heatmap.png')
    plt.close(fig_y)

    plot_df_g = out.sort_values('prob_gain_r8_12views', ascending=False).reset_index(drop=True)
    mat_g = plot_df_g[gain_cols_12].to_numpy()
    fig_g, ax_g = plt.subplots(figsize=(15, 9))
    im2 = ax_g.imshow(mat_g, cmap='YlOrBr', aspect='auto', vmin=0, vmax=1)
    ax_g.set_yticks(np.arange(len(plot_df_g)))
    ax_g.set_yticklabels(plot_df_g['genotype'], fontsize=8)
    ax_g.set_xticks(np.arange(len(xlabels)))
    ax_g.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=8)
    ax_g.set_title('Gain-R8 Presence Heatmap Across 12 Views (DINOV3 + VI)', fontsize=13, fontweight='bold')
    ax_g.set_xlabel('Feature-View', fontsize=11, fontweight='bold')
    ax_g.set_ylabel('Genotype (sorted by Gain-R8 frequency)', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax_g, fraction=0.02, pad=0.01)
    plt.tight_layout()
    _save_png_pdf(cross_dir / 'gain_r8_cross_feature_heatmap.png')
    plt.close(fig_g)

    # Frequency bars: Yield + Gain + overall(24 views).
    def _plot_prob_bar(df_sorted, prob_col, title, xlabel, out_name):
        fig, ax = plt.subplots(figsize=(14, 9))
        y_pos = np.arange(len(df_sorted))
        vals = df_sorted[prob_col].to_numpy() * 100
        colors = plt.cm.RdYlGn(vals / 100.0)
        ax.barh(y_pos, vals, color=colors, edgecolor='black', linewidth=0.6, alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted['genotype'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.25, linestyle=':')
        for i, v in enumerate(vals):
            ax.text(min(v + 0.8, 99.4), i, f'{v:.1f}%', va='center', fontsize=8, fontweight='bold')
        plt.tight_layout()
        _save_png_pdf(cross_dir / out_name)
        plt.close(fig)

    _plot_prob_bar(
        out.sort_values('prob_yield_r8_12views', ascending=False),
        'prob_yield_r8_12views',
        '(Yield) All 30 Genotypes by R8 Frequency Across 12 Views',
        'Yield-R8 Frequency (%)',
        'yield_r8_cross_feature_frequency_bar.png',
    )
    _plot_prob_bar(
        out.sort_values('prob_gain_r8_12views', ascending=False),
        'prob_gain_r8_12views',
        '(Gain) All 30 Genotypes by R8 Frequency Across 12 Views',
        'Gain-R8 Frequency (%)',
        'gain_r8_cross_feature_frequency_bar.png',
    )

    # Keep only panel-B style for ideal_zone_cross_feature_comparison.
    panel_b_df = out.sort_values('prob_r8_overall_24views', ascending=False)
    fig_b, ax_b = plt.subplots(figsize=(14, 9))
    y_pos = np.arange(len(panel_b_df))
    vals = panel_b_df['prob_r8_overall_24views'].to_numpy() * 100
    colors = plt.cm.YlGnBu(vals / 100.0)
    ax_b.barh(y_pos, vals, color=colors, edgecolor='black', linewidth=0.6)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(panel_b_df['genotype'], fontsize=9)
    ax_b.invert_yaxis()
    ax_b.set_xlabel('Overall R8 Frequency Across 24 Views (%)', fontsize=12, fontweight='bold')
    ax_b.set_title('(B) All 30 Genotypes by Overall R8 Frequency (24 views)', fontsize=14, fontweight='bold')
    ax_b.grid(True, axis='x', alpha=0.3, linestyle=':')
    for i, v in enumerate(vals):
        ax_b.text(min(v + 0.8, 99.3), i, f'{v:.1f}%', va='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    _save_png_pdf(cross_dir / 'ideal_zone_cross_feature_comparison.png')
    plt.close(fig_b)

    # cross_feature_r8_summary: explicit labels with offset + connector lines.
    summary_xlabel_fs = 16
    summary_ylabel_fs = 16
    summary_tick_fs = 14
    summary_ytick_fs = 13
    summary_panel_title_fs = 16
    summary_bar_value_fs = 11
    summary_scatter_label_fs = 10
    summary_cbar_label_fs = 14
    summary_cbar_tick_fs = 13
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.6), gridspec_kw={'width_ratios': [1.0, 1.0]})
    top_n = min(20, len(out))
    top_df = out.sort_values('prob_r8_overall_24views', ascending=False).head(top_n)
    y = np.arange(top_n)
    ax1.barh(y, top_df['prob_r8_overall_24views'] * 100, color='#2f7d4a', alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(top_df['genotype'], fontsize=summary_ytick_fs, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xlabel('Overall Z8 Frequency (24 views, %)', fontsize=summary_xlabel_fs)
    ax1.grid(True, axis='x', alpha=0.25, linestyle=':')
    ax1.tick_params(axis='x', labelsize=summary_tick_fs)
    ax1.tick_params(axis='y', labelsize=summary_ytick_fs)

    # Keep percentage labels inside bars and inside axes boundary.
    for i, v in enumerate((top_df['prob_r8_overall_24views'] * 100).to_numpy()):
        x_pos = max(0.8, v - 0.9)
        ax1.text(
            x_pos,
            i,
            f'{v:.1f}%',
            va='center',
            ha='right',
            fontsize=summary_bar_value_fs,
            fontweight='bold',
            color='white',
            clip_on=True,
        )

    ax1.text(
        0.5,
        -0.10,
        'a',
        transform=ax1.transAxes,
        ha='center',
        va='top',
        fontsize=summary_panel_title_fs,
        fontweight='bold',
    )

    sc = ax2.scatter(
        out['prob_yield_r8_12views'] * 100,
        out['prob_gain_r8_12views'] * 100,
        s=145,
        c=out['prob_r8_overall_24views'] * 100,
        cmap='YlGnBu',
        alpha=0.88,
        edgecolors='black',
        linewidths=0.9,
    )

    # Dynamic label placement in display-space to avoid overlap.
    # No fixed 8-direction constraint: use multi-angle + multi-radius candidates.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax2.get_window_extent(renderer=renderer)
    placed_bboxes = []

    # Put higher-frequency points first for better placement quality.
    label_df = out.sort_values('prob_r8_overall_24views', ascending=False).reset_index(drop=True)
    angle_list = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    radius_list = [52, 68, 86, 106, 128, 152]

    # 1) Collapse (0,0) dense points: do not label one-by-one.
    zero_zero_mask = (
        (label_df['prob_yield_r8_12views'] * 100 == 0.0)
        & (label_df['prob_gain_r8_12views'] * 100 == 0.0)
    )
    label_df_nonzero = label_df[~zero_zero_mask].copy()
    label_bbox_style = dict(
        boxstyle='round,pad=0.12',
        facecolor='#fff3b0',
        edgecolor='#8a7a00',
        linewidth=0.55,
        alpha=0.88,
    )

    # 2) Right-bottom edge points (high yield, near-zero gain): force a dedicated lane.
    x_rb = label_df_nonzero['prob_yield_r8_12views'] * 100
    y_rb = label_df_nonzero['prob_gain_r8_12views'] * 100
    right_bottom_mask = ((x_rb >= 66.0) & (y_rb <= 10.0)) | ((x_rb >= 54.0) & (y_rb <= 3.5))
    right_bottom_df = label_df_nonzero[right_bottom_mask].sort_values(
        ['prob_yield_r8_12views', 'prob_gain_r8_12views'],
        ascending=[False, False],
    )
    normal_df = label_df_nonzero[~right_bottom_mask]

    if len(right_bottom_df) > 0:
        rb_n = len(right_bottom_df)
        rb_top_y = min(18.0, 3.2 + 2.2 * rb_n)
        rb_bottom_y = 1.0
        rb_y_slots = np.linspace(rb_top_y, rb_bottom_y, rb_n)
        rb_x_lane = max(56.0, min(63.0, float((out['prob_yield_r8_12views'] * 100).max()) - 15.0))

        for slot_y, (_, row) in zip(rb_y_slots, right_bottom_df.iterrows()):
            x_val = float(row['prob_yield_r8_12views'] * 100)
            y_val = float(row['prob_gain_r8_12views'] * 100)
            label = str(row['genotype'])

            # Keep right-edge labels inside axes by pinning them to a stable lane.
            lx = rb_x_lane
            ly = slot_y

            temp_text = ax2.text(0, 0, label, fontsize=summary_scatter_label_fs, fontweight='bold', alpha=0.0)
            text_box = temp_text.get_window_extent(renderer=renderer).expanded(1.22, 1.40)
            temp_text.remove()
            text_w, text_h = text_box.width, text_box.height
            cx, cy = ax2.transData.transform((lx, ly))
            bbox = [
                cx - text_w,
                cy - text_h / 2.0,
                cx,
                cy + text_h / 2.0,
            ]
            placed_bboxes.append(bbox)

            ax2.plot([x_val, lx], [y_val, ly], color='#404040', linewidth=0.95, alpha=0.75, zorder=2)
            ax2.text(
                lx,
                ly,
                label,
                fontsize=summary_scatter_label_fs,
                fontweight='bold',
                color='#1b1b1b',
                ha='right',
                va='center',
                zorder=3,
                clip_on=False,
                bbox=label_bbox_style,
            )

    for _, row in normal_df.iterrows():
        x_val = float(row['prob_yield_r8_12views'] * 100)
        y_val = float(row['prob_gain_r8_12views'] * 100)
        label = str(row['genotype'])

        # Measure label size in pixel space for robust overlap checks.
        temp_text = ax2.text(0, 0, label, fontsize=summary_scatter_label_fs, fontweight='bold', alpha=0.0)
        text_box = temp_text.get_window_extent(renderer=renderer).expanded(1.22, 1.40)
        temp_text.remove()
        text_w, text_h = text_box.width, text_box.height

        point_disp = ax2.transData.transform((x_val, y_val))
        best = None
        best_score = float('inf')

        for r_px in radius_list:
            for ang in angle_list:
                cx = point_disp[0] + r_px * np.cos(ang)
                cy = point_disp[1] + r_px * np.sin(ang)

                dx = cx - point_disp[0]
                dy = cy - point_disp[1]
                if abs(dx) >= abs(dy):
                    ha = 'left' if dx >= 0 else 'right'
                    va = 'center'
                else:
                    ha = 'center'
                    va = 'bottom' if dy >= 0 else 'top'

                if ha == 'left':
                    bx0 = cx
                    bx1 = cx + text_w
                elif ha == 'right':
                    bx0 = cx - text_w
                    bx1 = cx
                else:
                    bx0 = cx - text_w / 2.0
                    bx1 = cx + text_w / 2.0

                if va == 'bottom':
                    by0 = cy
                    by1 = cy + text_h
                elif va == 'top':
                    by0 = cy - text_h
                    by1 = cy
                else:
                    by0 = cy - text_h / 2.0
                    by1 = cy + text_h / 2.0

                bbox = [bx0, by0, bx1, by1]

                inside_axes = (
                    bbox[0] >= axes_bbox.x0 + 4
                    and bbox[2] <= axes_bbox.x1 - 4
                    and bbox[1] >= axes_bbox.y0 + 4
                    and bbox[3] <= axes_bbox.y1 - 4
                )
                if not inside_axes:
                    continue

                overlap_area = 0.0
                for pb in placed_bboxes:
                    inter_w = max(0.0, min(bbox[2], pb[2]) - max(bbox[0], pb[0]))
                    inter_h = max(0.0, min(bbox[3], pb[3]) - max(bbox[1], pb[1]))
                    overlap_area += inter_w * inter_h

                # Prefer no-overlap candidates; secondarily prefer shorter connectors.
                score = overlap_area * 10.0 + (r_px / 140.0) ** 2
                if score < best_score:
                    best_score = score
                    best = (cx, cy, bbox, r_px, ha, va)

        # Fallback: clamp to visible area if all candidates are blocked.
        if best is None:
            cx = min(max(point_disp[0] + 120.0, axes_bbox.x0 + text_w / 2.0 + 4), axes_bbox.x1 - text_w / 2.0 - 4)
            cy = min(max(point_disp[1] + 40.0, axes_bbox.y0 + text_h / 2.0 + 4), axes_bbox.y1 - text_h / 2.0 - 4)
            bbox = [
                cx,
                cy - text_h / 2.0,
                cx + text_w,
                cy + text_h / 2.0,
            ]
            best = (cx, cy, bbox, 126, 'left', 'center')

        cx, cy, bbox, _, ha_text, va_text = best
        placed_bboxes.append(bbox)
        lx, ly = ax2.transData.inverted().transform((cx, cy))

        # Connector line from point to label center (slightly longer by design).
        ax2.plot([x_val, lx], [y_val, ly], color='#404040', linewidth=0.8, alpha=0.7, zorder=2)
        ax2.text(
            lx,
            ly,
            label,
            fontsize=summary_scatter_label_fs,
            fontweight='bold',
            color='#1b1b1b',
            ha=ha_text,
            va=va_text,
            zorder=3,
            clip_on=False,
            bbox=label_bbox_style,
        )

    ax2.set_xlabel('Yield-Z8 Frequency (12 views, %)', fontsize=summary_xlabel_fs)
    ax2.set_ylabel('Gain-Z8 Frequency (12 views, %)', fontsize=summary_ylabel_fs)
    ax2.set_aspect('equal', adjustable='box')
    ax2.margins(x=0.06, y=0.06)
    ax2.grid(True, alpha=0.25, linestyle=':')
    ax2.tick_params(axis='x', labelsize=summary_tick_fs)
    ax2.tick_params(axis='y', labelsize=summary_tick_fs)

    ax2.text(
        0.5,
        -0.10,
        'b',
        transform=ax2.transAxes,
        ha='center',
        va='top',
        fontsize=summary_panel_title_fs,
        fontweight='bold',
    )

    cbar = plt.colorbar(sc, ax=ax2, fraction=0.045, pad=0.02, label='Overall Z8 Frequency (24 views, %)')
    cbar.ax.tick_params(labelsize=summary_cbar_tick_fs)
    cbar.set_label('Overall Z8 Frequency (24 views, %)', fontsize=summary_cbar_label_fs)

    fig.subplots_adjust(bottom=0.13, wspace=0.10)
    _save_png_pdf(cross_dir / 'cross_feature_r8_summary.png')
    plt.close(fig)

    # Text summary
    summary_txt = cross_dir / 'ranking_cross_feature_summary.txt'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write('=' * 90 + '\n')
        f.write('CROSS-FEATURE R8 SUMMARY (DINOV3 + VI)\n')
        f.write('=' * 90 + '\n\n')
        f.write('Core frequency definitions:\n')
        f.write('  - Yield-R8 frequency: count in 12 yield views (2 features x 6 views) / 12\n')
        f.write('  - Gain-R8 frequency: count in 12 gain views (2 features x 6 views) / 12\n')
        f.write('  - Overall R8 frequency: total count in 24 views / 24\n\n')
        f.write('Strict intersection definitions (more conservative):\n')
        f.write('  - cross_yield_intersection_6views: both DINOV3 and VI are R8 in the same view\n')
        f.write('  - cross_gain_intersection_6views: both DINOV3 and VI are R8 in the same view\n')
        f.write('  - cross_dual_intersection_6views: both of the above are true in the same view\n\n')
        f.write('Files generated:\n')
        f.write('  - r8_cross_feature_comprehensive.csv\n')
        f.write('  - ranking_by_yield_r8_all_features.csv\n')
        f.write('  - ranking_by_gain_r8_all_features.csv\n')
        f.write('  - ranking_by_intersection_r8_all_features.csv\n')
        f.write('  - robust_candidates_intersection.csv\n')
        f.write('  - yield_r8_cross_feature_heatmap.png/.pdf\n')
        f.write('  - gain_r8_cross_feature_heatmap.png/.pdf\n')
        f.write('  - yield_r8_cross_feature_frequency_bar.png/.pdf\n')
        f.write('  - gain_r8_cross_feature_frequency_bar.png/.pdf\n')
        f.write('  - ideal_zone_cross_feature_comparison.png/.pdf\n')
        f.write('  - cross_feature_r8_summary.png/.pdf\n\n')

        top_overall = out.sort_values('prob_r8_overall_24views', ascending=False).head(20)
        f.write('Top 20 by Overall R8 Frequency (24 views):\n')
        for i, (_, row) in enumerate(top_overall.iterrows(), start=1):
            f.write(
                f"#{i:02d} {row['genotype']:<15} | "
                f"Yield={row['prob_yield_r8_12views']*100:5.1f}% | "
                f"Gain={row['prob_gain_r8_12views']*100:5.1f}% | "
                f"Overall={row['prob_r8_overall_24views']*100:5.1f}%\n"
            )

    print(f"✓ Cross-feature summary saved: {summary_txt}")


if __name__ == '__main__':
    # ============================================================================
    # 注释说明：Part 1 是原有的预测模型分析（含虫害预测、多指标预测等）
    # 如果需要运行，取消下面的注释即可
    # ============================================================================
    
    print("\n" + "="*80)
    print("🌾 PART 1: ORIGINAL ANALYSIS (Prediction Models & Visualizations)")
    print("="*80)
    # run_prediction_experiments()  # 运行三组预测实验 (DINOv3, VI, Fusion)
    print("\n")
    
    print("="*80)
    print("🌾 FEATURE DIFFERENCE & RANKING ANALYSIS")
    print("="*80)
    
    # 分析1：特征差异 vs NoControl产量（时序，6个子图）
    print("\n[1/2] Feature Diff vs NoControl Yield (Time-series)")
    print("-" * 80)
    for feature_type in ['dinov3', 'vi']:
        try:
            analyze_feature_vs_yield_timeseries(feature_type)
        except Exception as e:
            print(f"Error in yield time-series for {feature_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 分析2：综合排名可视化（带评分）
    print("\n[2/2] Comprehensive Ranking Visualization")
    print("-" * 80)
    for feature_type in ['dinov3', 'vi']:
        try:
            create_comprehensive_ranking_visualization(feature_type)
        except Exception as e:
            print(f"Error in ranking visualization for {feature_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # 图像裁剪/对比展示按当前需求跳过
    # ========================================================================
    print("\n[3/6] Genotype Image Comparison (Control vs NoControl)")
    print("-" * 80)
    print("Skipped by configuration.")
    
    # ========================================================================
    # 新增分析：产量对比可视化（Control vs NoControl）
    # ========================================================================
    print("\n[4/6] Yield Comparison Visualization (Control vs NoControl)")
    print("-" * 80)
    try:
        from insect_resistance.visualization import visualize_yield_comparison
        visualize_yield_comparison('dinov3')  # 只需生成一次，数据与特征类型无关
    except Exception as e:
        print(f"Error in yield comparison: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # IDEAL Zone分析：单特征分析（每种特征类型独立分析）
    # ========================================================================
    print("\n[5/6] IDEAL Zone Analysis (Single Feature Types)")
    print("-" * 80)
    print("Generating standalone scatter plots with all 30 genotypes labeled...")
    print("Saving statistical summaries as separate text files...")
    print()
    
    for feature_type in ['dinov3', 'vi']:
        print(f"\n{'='*80}")
        print(f"Processing {feature_type.upper()} features...")
        print(f"{'='*80}\n")
        try:
            visualize_single_feature_ideal_zone(feature_type)
        except Exception as e:
            print(f"Error in IDEAL zone analysis for {feature_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # IDEAL Zone分析：跨特征对比（三种特征类型综合对比）
    # ========================================================================
    print(f"\n[6/6] IDEAL Zone Cross-Feature Comparison")
    print("-" * 80)
    try:
        analyze_ideal_zone_genotypes(['dinov3', 'vi'])
    except Exception as e:
        print(f"Error in cross-feature IDEAL zone analysis: {str(e)}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # 统一导出：跨特征双排名汇总（便于横向对比）
    # ========================================================================
    try:
        export_cross_feature_ranking_summary(Path(__file__).parent.parent)
    except Exception as e:
        print(f"Error exporting cross-feature ranking summary: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✓ ALL ANALYSES COMPLETE!")
    print("="*80)
    print("\nResults saved in:")
    print("\n📈 Analysis 1: Feature vs Yield/Gain 3D Time-series")
    print("  • experiments/insect_resistance/outputs/results/dinov3/")
    print("    - feature_vs_yield_timeseries_dinov3.png (6 timepoints)")
    print("    - feature_vs_gain_timeseries_dinov3.png (6 timepoints)")
    print("    - feature_vs_yield_timeseries_data_dinov3.csv")
    print("  • experiments/insect_resistance/outputs/results/vi/")
    print("    - feature_vs_yield_timeseries_vi.png")
    print("    - feature_vs_gain_timeseries_vi.png")
    print("\n🏆 Analysis 2: Comprehensive Ranking (综合排名 - 不再输出单独条形图)")
    print("  • experiments/insect_resistance/outputs/results/dinov3/")
    print("    - comprehensive_ranking_scatter_dinov3.png (两个散点图)")
    print("    - comprehensive_ranking_gain_scatter_dinov3.png (增益率散点图)")
    print("    - comprehensive_ranking_table_dinov3.png (数据表)")
    print("    - comprehensive_ranking_data_dinov3.csv")
    print("  • experiments/insect_resistance/outputs/results/vi/")
    print("    - comprehensive_ranking_scatter_vi.png")
    print("    - comprehensive_ranking_gain_scatter_vi.png")
    print("    - comprehensive_ranking_table_vi.png")
    print("\n📷 Analysis 3: Genotype Image Comparison")
    print("  • Skipped by configuration")
    print("\n🎯 Analysis 4: IDEAL Zone Analysis (Single Features)")
    print("  • experiments/insect_resistance/outputs/results/dinov3/")
    print("    - ideal_zone_comprehensive_dinov3.png (3-panel visualization)")
    print("    - yield_vs_ideal_zone_scatter_dinov3.png (Standalone scatter with all 30 labeled)")
    print("    - ideal_zone_statistical_summary_dinov3.txt (Detailed text summary)")
    print("    - ideal_zone_detailed_dinov3.csv (Detailed data table)")
    print("  • experiments/insect_resistance/outputs/results/vi/")
    print("    - ideal_zone_comprehensive_vi.png")
    print("    - yield_vs_ideal_zone_scatter_vi.png")
    print("\n🎯 Analysis 5: IDEAL Zone Cross-Feature Comparison")
    print("  • experiments/insect_resistance/outputs/results/cross_feature_analysis/")
    print("    - ideal_zone_cross_feature_comparison.png (Main comparison visualization)")
    print("    - cross_feature_r8_summary.png (Cross-feature R8 summary)")
    print("    - ideal_zone_genotypes_comparison.csv (Comparison data)")
    print("    - ideal_zone_analysis_report.txt (Cross-feature report)")
    print()
