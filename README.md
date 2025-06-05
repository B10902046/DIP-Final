# DIP-Final

1. Install Python requirements:
```
pip3 install -r requirements.txt
```

2. Run the script:
```
python process_depth_image.py \
  --depth_image_path /path/to/input_depth.png \
  --output_image_path /path/to/output_depth.png \
  --max_depth 100.0 \
  --custom_kernel_name DIAMOND_KERNEL_5 \
  --extrapolate \
  --blur_type bilateral
```

3. Set options in `process_depth_image.py`

- `depth_image_path`:  
  - **Type:** `str`  
  - **Required:** Yes  
  - Path to the input 16-bit PNG depth image (e.g., from KITTI dataset).

- `output_image_path`:  
  - **Type:** `str`  
  - **Required:** Yes  
  - Path where the processed depth image will be saved.

- `max_depth`:  
  - **Type:** `float`  
  - **Default:** `100.0`  
  - Maximum depth value; depth values greater than this will be clipped.

- `custom_kernel_name`:  
  - **Type:** `str`  
  - **Default:** `'DIAMOND_KERNEL_5'`  
  - Name of the dilation kernel used for depth map filling. Options include:
    - `'FULL_KERNEL_3'`, `'FULL_KERNEL_5'`, `'FULL_KERNEL_7'`, `'FULL_KERNEL_9'`  
    - `'CROSS_KERNEL_3'`, `'CROSS_KERNEL_5'`, `'CROSS_KERNEL_7'`  
    - `'DIAMOND_KERNEL_5'`, `'DIAMOND_KERNEL_7'`

- `extrapolate`:  
  - **Type:** `bool`  
  - **Default:** `False`  
  - Whether to extend depth values to the top of the image and perform large kernel dilation.

- `blur_type`:  
  - **Type:** `str`  
  - **Default:** `'bilateral'`  
  - Type of blur applied after filling:
    - `'bilateral'` — Apply bilateral filter  
    - `'gaussian'` — Apply Gaussian blur  
    - `'none'` — No blur applied

---

4. Evaluate
```
python evaluate_depth.py \
  --pred_dir /path/to/predicted_depths \
  --gt_dir /path/to/ground_truth_depths 
```

5. Set options in `evaluate_depth.py`

- `pred_dir`:  
  - **Type:** `str`  
  - **Required:** Yes  
  - Path to the directory containing predicted depth PNG files.

- `gt_dir`:  
  - **Type:** `str`  
  - **Required:** Yes  
  - Path to the directory containing ground-truth depth PNG files.

---
