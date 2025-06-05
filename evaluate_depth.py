import os
import argparse
import numpy as np
import imageio
from glob import glob
from tqdm import tqdm

def read_depth_png(path):
    """讀取 KITTI depth PNG 並轉換成 meter 單位"""
    depth = imageio.v2.imread(path).astype(np.float32)
    return depth / 256.0  # uint16 -> meter

def compute_metrics(gt, pred):
    mask = (gt > 0)
    gt = gt[mask]
    pred = pred[mask]

    # 限制最小深度，防止反深度爆炸
    gt = np.clip(gt, 1.0, 80.0)
    pred = np.clip(pred, 1.0, 80.0)

    rmse = np.sqrt(np.mean((pred - gt) ** 2)) * 1000  # mm
    mae = np.mean(np.abs(pred - gt)) * 1000           # mm
    irmse = np.sqrt(np.mean((1. / pred - 1. / gt) ** 2)) * 1000  # 1/km
    imae = np.mean(np.abs(1. / pred - 1. / gt)) * 1000           # 1/km

    return rmse, mae, irmse, imae

def evaluate(pred_dir, gt_dir):
    pred_files = sorted(glob(os.path.join(pred_dir, "*.png")))
    gt_files = sorted(glob(os.path.join(gt_dir, "*.png")))

    assert len(pred_files) == len(gt_files), "數量不一致"

    rmse_list, mae_list, irmse_list, imae_list = [], [], [], []

    for pred_path, gt_path in tqdm(zip(pred_files, gt_files), total=len(pred_files)):
        pred = read_depth_png(pred_path)
        gt = read_depth_png(gt_path)

        rmse, mae, irmse, imae = compute_metrics(gt, pred)
        rmse_list.append(rmse)
        mae_list.append(mae)
        irmse_list.append(irmse)
        imae_list.append(imae)

    print("\n=== 評估結果 ===")
    print(f"RMSE (mm):    {np.mean(rmse_list):.2f}")
    print(f"MAE (mm):     {np.mean(mae_list):.2f}")
    print(f"iRMSE (1/km): {np.mean(irmse_list):.2f}")
    print(f"iMAE (1/km):  {np.mean(imae_list):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted depth maps against ground truth.")
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to directory with predicted depth PNGs')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to directory with ground truth depth PNGs')
    args = parser.parse_args()

    evaluate(args.pred_dir, args.gt_dir)
