#!/usr/bin/env python3
"""
GeoTIFF to STL Converter for 3D Printing
將 GeoTIFF 高程數據轉換為 3D 列印用的 STL 模型

使用方式:
    python convert_tif_to_stl.py input.tif output.stl --base-thickness 5.0 --z-scale 1.5 --target-width 120.0

參數:
    input.tif          輸入的 GeoTIFF 檔案路徑
    output.stl         輸出的 STL 檔案路徑
    --base-thickness   底座厚度 (mm), 預設: 3.0
    --z-scale          地形 Z 軸倍率, 預設: 1.0
    --target-width     目標模型寬度 (mm), 預設: 100.0
    --max-resolution   最大解析度, 預設: 800
    --sealevel         海平面高度 (原始高程單位), 低於此高度的部份將被裁剪, 預設: 自動偵測最低海拔
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from stl import mesh
import math
import os
import argparse
import sys


def convert_tif_to_stl(
    input_tif,
    output_stl,
    base_thickness=3.0,
    z_scale=1.0,
    target_width_mm=100.0,
    max_resolution=800,
    sealevel=None,
    max_height_mm=None,
    flip_x=True,
    flip_y=False
):
    """
    將 GeoTIFF 轉換為 3D 列印用的 STL 模型。
    
    參數:
        input_tif (str): 輸入的 GeoTIFF 檔案路徑。
        output_stl (str): 輸出的 STL 檔案路徑。
        base_thickness (float): 底座厚度 (mm)。
        z_scale (float): 地形 Z 軸誇飾倍率 (1.0 為真實比例，2.0 為兩倍高)。
        target_width_mm (float): 預期的 3D 列印模型 X 軸寬度 (mm)。
        max_resolution (int): 最大解析度 (避免產生過大檔案，預設 800 像素)。
        sealevel (float): 海平面高度，低於此高度的部份將被裁剪至此高度 (原始高程單位)，預設 None = 自動偵測最低海拔。
        max_height_mm (float): 模型最大高度限制 (mm)，若指定則會自動調整 z_scale 以符合此限制。
        flip_x (bool): 是否翻轉 X 軸 (左右方向)，預設 True。
        flip_y (bool): 是否翻轉 Y 軸 (上下方向)，預設 False。
    """
    if not os.path.exists(input_tif):
        raise FileNotFoundError(f"找不到檔案: {input_tif}")

    print(f"[{input_tif}] 讀取中...")
    
    with rasterio.open(input_tif) as src:
        # 計算是否需要降採樣以防檔案過大
        width, height = src.width, src.height
        scale = 1.0
        if max(width, height) > max_resolution:
            scale = max_resolution / float(max(width, height))
            width = int(width * scale)
            height = int(height * scale)
            print(f"影像過大，自動降採樣至 {width}x{height}...")

        # 讀取高程數據並應用重採樣
        elev = src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.bilinear
        )
        
        # 獲取像素的真實單位 (考慮經緯度坐標系的緯度校正)
        res_x = src.res[0] / scale  # X 方向解析度
        res_y = abs(src.res[1]) / scale  # Y 方向解析度 (取絕對值)
        nodata = src.nodata
        
        # 判斷坐標系類型並計算實際像素大小 (公尺)
        # 獲取 CRS 信息
        crs = src.crs
        crs_str = str(crs) if crs else ""
        
        # 更穩健的方法判斷是否為經緯度坐標系
        is_geographic = False
        if crs is not None:
            # 方法 1: 使用 is_geographic 屬性 (rasterio 1.2+)
            if hasattr(crs, 'is_geographic') and crs.is_geographic:
                is_geographic = True
            # 方法 2: 檢查 EPSG 代碼 (4326 = WGS84 經緯度)
            elif '4326' in crs_str or 'EPSG:4326' in crs_str:
                is_geographic = True
            # 方法 3: 檢查是否包含 geographic 字眼
            elif 'geographic' in crs_str.lower():
                is_geographic = True
        
        if is_geographic:
            # 經緯度坐標系 (度為單位), 需要校正經度方向
            # 使用影像中心的緯度做為校正依據
            center_lat = (src.bounds.top + src.bounds.bottom) / 2
            # 1 度緯度 = 111,319 公尺 (近似值)
            # 1 度經度 = 111,319 * cos(緯度) 公尺
            meters_per_degree_lat = 111319.0
            meters_per_degree_lon = 111319.0 * math.cos(math.radians(center_lat))
            
            pixel_size_x = res_x * meters_per_degree_lon
            pixel_size_y = res_y * meters_per_degree_lat
            print(f"坐標系: 經緯度 ({crs_str}), 校正緯度: {center_lat:.4f}°, X解析度: {pixel_size_x:.2f}m, Y解析度: {pixel_size_y:.2f}m")
        else:
            # 投影坐標系 (單位通常是公尺)
            # 假設解析度已經是公尺/像素
            pixel_size_x = abs(res_x)
            pixel_size_y = abs(res_y)
            print(f"坐標系: 投影坐標系 ({crs_str}), X解析度: {pixel_size_x:.2f}m, Y解析度: {pixel_size_y:.2f}m")
        
        # 使用平均像素大小
        pixel_size_avg = (pixel_size_x + pixel_size_y) / 2

    # 自動偵測最低海拔 (如果 sealevel 未指定)
    if sealevel is None:
        # 暫時處理 NoData 以便計算最低值
        if nodata is not None:
            valid_mask = elev != nodata
            if not np.any(valid_mask):
                raise ValueError("GeoTIFF 內全部都是 NoData。")
            # 使用有效數據計算最低海拔
            sealevel = float(np.min(elev[valid_mask]))
        else:
            sealevel = float(np.min(elev))
        print(f"自動偵測到最低海拔: {sealevel:.2f} (已設為海平面)")

    # 處理 NoData 值 (填補為海平面)
    if nodata is not None:
        valid_mask = elev != nodata
        if not np.any(valid_mask):
            raise ValueError("GeoTIFF 內全部都是 NoData。")
        elev[elev == nodata] = sealevel

    # 處理海面以下部分：將低於海平面的部分裁剪至海平面高度
    print(f"應用海平面裁剪: {sealevel}...")
    below_sealevel_mask = elev < sealevel
    if np.any(below_sealevel_mask):
        elev[below_sealevel_mask] = sealevel
        clipped_count = np.sum(below_sealevel_mask)
        print(f"裁剪了 {clipped_count} 個像素 (低於海平面的部分已被提升至海平面高度)")

    # 找到最低海拔 (裁剪後的最低點)
    min_val = np.min(elev)
    max_val = np.max(elev)
    print(f"裁剪後的高程範圍: {min_val:.2f} 到 {max_val:.2f}")
    
    # 如果指定了最大高度限制，自動調整 z_scale
    # max_height_mm 指的是地形部分的高度 (不含底座)
    if max_height_mm is not None:
        elevation_range = max_val - min_val
        if elevation_range > 0:
            # 計算所需的 z_scale 使得地形高度 = max_height_mm
            # 地形高度 = z_real_scale * elevation_range = max_height_mm
            # z_real_scale = (xy_scale / pixel_size_avg) * z_scale
            # 所以: (xy_scale / pixel_size_avg) * z_scale * elevation_range = max_height_mm
            # 解得: z_scale = max_height_mm * pixel_size_avg / (xy_scale * elevation_range)
            
            # 先計算 xy_scale (使用降采樣後的寬度)
            xy_scale_temp = target_width_mm / width
            
            auto_z_scale = (max_height_mm * pixel_size_avg) / (xy_scale_temp * elevation_range)
            print(f"自動調整 z_scale: {z_scale} -> {auto_z_scale:.8f} (地形高度限制在 {max_height_mm}mm)")
            z_scale = auto_z_scale
        else:
            print("警告: 高程數據範圍為0，無法自動調整高度")

    # 歸零最低海拔，讓最底部的地形高度為 0
    elev = elev - min_val
    rows, cols = elev.shape

    # 計算 X, Y 縮放比例以符合列印台尺寸
    xy_scale = target_width_mm / cols

    # 計算真實的 Z 軸高度：
    # (高程值) * (縮放後的單位 / 真實單位) * 用戶指定倍率
    z_real_scale = (xy_scale / pixel_size_avg) * z_scale
    z_data = elev * z_real_scale

    # --- 1. 建立頂部地形 (Top Surface) 頂點與面 ---
    print("計算 3D 網格中...")
    x_lin = np.arange(cols) * xy_scale
    y_lin = np.arange(rows) * xy_scale
    x, y = np.meshgrid(x_lin, y_lin)
    
    # 翻轉坐標軸以修正方向
    if flip_x:
        x = np.fliplr(x)
        print("已翻轉 X 軸 (左右方向)")
    if flip_y:
        y = np.flipud(y)
        print("已翻轉 Y 軸 (上下方向)")

    # 頂點陣列：[rows * cols, 3]
    vertices_top = np.column_stack((x.flatten(), y.flatten(), z_data.flatten()))

    idx = np.arange(rows * cols).reshape(rows, cols)
    v1 = idx[:-1, :-1].flatten()
    v2 = idx[:-1, 1:].flatten()
    v3 = idx[1:, :-1].flatten()
    v4 = idx[1:, 1:].flatten()

    # 頂部三角面 (分為兩個三角形)
    faces_top = np.zeros((len(v1) * 2, 3), dtype=int)
    faces_top[0::2] = np.column_stack((v1, v2, v3))
    faces_top[1::2] = np.column_stack((v2, v4, v3))

    # --- 2. 建立邊界與牆壁 (Walls) ---
    # 依順時針方向取得邊界索引
    top_edge = idx[0, :-1]
    right_edge = idx[:-1, -1]
    bottom_edge = idx[-1, 1:][::-1]
    left_edge = idx[1:, 0][::-1]
    perimeter = np.concatenate([top_edge, right_edge, bottom_edge, left_edge])
    P = len(perimeter)

    # 底座頂點 (Z = -base_thickness)
    base_vertices = vertices_top[perimeter].copy()
    base_vertices[:, 2] = -base_thickness

    # 合併所有頂點
    all_vertices = np.vstack([vertices_top, base_vertices])
    p_top = perimeter
    p_bot = np.arange(rows * cols, rows * cols + P)

    # 牆壁三角面 (每個邊界線段產生兩個三角形)
    faces_wall = np.zeros((P * 2, 3), dtype=int)
    for i in range(P):
        next_i = (i + 1) % P
        faces_wall[i*2] = [p_top[i], p_bot[next_i], p_bot[i]]
        faces_wall[i*2 + 1] = [p_top[i], p_top[next_i], p_bot[next_i]]

    # --- 3. 建立平整底面 (Bottom Surface) ---
    # 使用 Triangle Fan 算法封閉底面
    faces_bottom = np.zeros((P - 2, 3), dtype=int)
    for i in range(1, P - 1):
        faces_bottom[i-1] = [p_bot[0], p_bot[i+1], p_bot[i]]

    # 組合所有面
    all_faces = np.vstack([faces_top, faces_wall, faces_bottom])

    # --- 4. 寫入 STL 檔案 ---
    print(f"輸出 STL 中 (總頂點數: {len(all_vertices)}, 總面數: {len(all_faces)})...")
    terrain_mesh = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
    
    for i, f in enumerate(all_faces):
        for j in range(3):
            terrain_mesh.vectors[i][j] = all_vertices[f[j], :]

    terrain_mesh.save(output_stl)
    print(f"成功！已儲存為: {output_stl}")
    
    return output_stl


def main():
    parser = argparse.ArgumentParser(
        description='將 GeoTIFF 高程檔案轉換為 3D 列印用 STL 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例子:
  python convert_tif_to_stl.py input.tif output.stl  # 自動偵測最低海拔
  python convert_tif_to_stl.py input.tif output.stl --base-thickness 5.0 --z-scale 2.0
  python convert_tif_to_stl.py input.tif output.stl --target-width 150.0 --max-resolution 1000
  python convert_tif_to_stl.py input.tif output.stl --sealevel 0  # 裁剪至海平面
  python convert_tif_to_stl.py input.tif output.stl --sealevel -100  # 保留海面下100公尺
  python convert_tif_to_stl.py input.tif output.stl --max-height 50  # 模型高度限制在50mm
  python convert_tif_to_stl.py input.tif output.stl --sealevel -7000 --max-height 50
  python convert_tif_to_stl.py input.tif output.stl --no-flip-x  # 關閉左右翻轉
  python convert_tif_to_stl.py input.tif output.stl --flip-y  # 翻轉上下方向
        """
    )
    
    parser.add_argument('input_tif', help='輸入 GeoTIFF 檔案路徑')
    parser.add_argument('output_stl', help='輸出 STL 檔案路徑')
    parser.add_argument('--base-thickness', type=float, default=3.0, 
                        help='底座厚度 (mm), 預設: 3.0')
    parser.add_argument('--z-scale', type=float, default=1.0,
                        help='地形 Z 軸倍率, 預設: 1.0')
    parser.add_argument('--target-width', type=float, default=100.0,
                        help='目標模型寬度 (mm), 預設: 100.0')
    parser.add_argument('--max-resolution', type=int, default=800,
                        help='最大解析度, 預設: 800')
    parser.add_argument('--sealevel', type=float, default=None,
                        help='海平面高度 (原始高程單位), 低於此高度的部份將被裁剪, 預設: 自動偵測最低海拔')
    parser.add_argument('--max-height', type=float, default=None,
                        help='模型最大高度限制 (mm)，若指定則會自動調整 z_scale 以符合此限制')
    parser.add_argument('--flip-x', action='store_true', default=True,
                        help='翻轉 X 軸 (左右方向)，預設啟用')
    parser.add_argument('--no-flip-x', action='store_false', dest='flip_x',
                        help='不翻轉 X 軸')
    parser.add_argument('--flip-y', action='store_true', default=False,
                        help='翻轉 Y 軸 (上下方向)')
    
    args = parser.parse_args()
    
    try:
        convert_tif_to_stl(
            input_tif=args.input_tif,
            output_stl=args.output_stl,
            base_thickness=args.base_thickness,
            z_scale=args.z_scale,
            target_width_mm=args.target_width,
            max_resolution=args.max_resolution,
            sealevel=args.sealevel,
            max_height_mm=args.max_height,
            flip_x=args.flip_x,
            flip_y=args.flip_y
        )
    except Exception as e:
        print(f"錯誤: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()