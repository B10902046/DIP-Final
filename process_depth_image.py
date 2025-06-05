import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

KERNELS = {
    'FULL_KERNEL_3': np.ones((3, 3), np.uint8),
    'FULL_KERNEL_5': np.ones((5, 5), np.uint8),
    'FULL_KERNEL_7': np.ones((7, 7), np.uint8),
    'FULL_KERNEL_9': np.ones((9, 9), np.uint8),
    'FULL_KERNEL_31': np.ones((31, 31), np.uint8),
    'CROSS_KERNEL_3': np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8),
    'CROSS_KERNEL_5': np.array([
        [0,0,1,0,0],
        [0,0,1,0,0],
        [1,1,1,1,1],
        [0,0,1,0,0],
        [0,0,1,0,0]
    ], dtype=np.uint8),
    'DIAMOND_KERNEL_5': np.array([
        [0,0,1,0,0],
        [0,1,1,1,0],
        [1,1,1,1,1],
        [0,1,1,1,0],
        [0,0,1,0,0]
    ], dtype=np.uint8),
    'CROSS_KERNEL_7': np.array([
        [0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0],
        [1,1,1,1,1,1,1],
        [0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0]
    ], dtype=np.uint8),
    'DIAMOND_KERNEL_7': np.array([
        [0,0,0,1,0,0,0],
        [0,0,1,1,1,0,0],
        [0,1,1,1,1,1,0],
        [1,1,1,1,1,1,1],
        [0,1,1,1,1,1,0],
        [0,0,1,1,1,0,0],
        [0,0,0,1,0,0,0]
    ], dtype=np.uint8),
}

def process_depth_image(
    depth_image_path: str,
    output_image_path: str,
    max_depth: float,
    custom_kernel_name: str,
    extrapolate: bool,
    blur_type: str
):
    assert custom_kernel_name in KERNELS, f"Unknown kernel: {custom_kernel_name}"
    assert blur_type in ['bilateral', 'gaussian', 'none'], f"Unknown blur type: {blur_type}"

    custom_kernel = KERNELS[custom_kernel_name]
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    depth_map = np.float32(depth_image / 256.0)

    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    depth_map = cv2.dilate(depth_map, custom_kernel)
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, KERNELS['FULL_KERNEL_5'])

    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, KERNELS['FULL_KERNEL_7'])
    depth_map[empty_pixels] = dilated[empty_pixels]

    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]
        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = top_pixel_values[pixel_col_idx]
        empty_pixels = (depth_map < 0.1)
        dilated = cv2.dilate(depth_map, KERNELS['FULL_KERNEL_31'])
        depth_map[empty_pixels] = dilated[empty_pixels]

    depth_map = cv2.medianBlur(depth_map, 5)

    if blur_type == 'bilateral':
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    normalized_depth = np.clip(depth_map / max_depth * 255, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, normalized_depth)

    plt.imshow(depth_map, cmap='plasma', vmin=0, vmax=80)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and enhance depth image.")
    parser.add_argument('--depth_image_path', type=str, required=True, help='Path to the input depth image')
    parser.add_argument('--output_image_path', type=str, required=True, help='Path to save the output image')
    parser.add_argument('--max_depth', type=float, default=100.0, help='Maximum depth value')
    parser.add_argument('--custom_kernel_name', type=str, default='DIAMOND_KERNEL_5', help='Kernel name to use')
    parser.add_argument('--extrapolate', action='store_true', help='Enable extrapolation to top of image')
    parser.add_argument('--blur_type', type=str, default='bilateral', choices=['bilateral', 'gaussian', 'none'], help='Type of blur to apply')

    args = parser.parse_args()

    process_depth_image(
        depth_image_path=args.depth_image_path,
        output_image_path=args.output_image_path,
        max_depth=args.max_depth,
        custom_kernel_name=args.custom_kernel_name,
        extrapolate=args.extrapolate,
        blur_type=args.blur_type
    )
