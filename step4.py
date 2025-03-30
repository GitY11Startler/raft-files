import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from argparse import Namespace

# Link the current file to the core (needed to access the RAFT class in raft.py)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Set up RAFT
def load_raft(model_path):
    args = Namespace(
        small=False,
        mixed_precision=False,
        alternate_corr=False
    )

    from raft import RAFT
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))
    return model

def preprocess_image(image):
    """Convert image to tensor and normalize"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    return image[None].to('cuda')

def compute_optical_flow(model, img1, img2):
    """Compute flow between two images"""
    with torch.no_grad():
        flow_low, flow_up = model(img1, img2, iters=20, test_mode=True)
    return flow_up[0].permute(1, 2, 0).cpu().numpy()

def visualize_flow(flow):
    """Convert flow to RGB visualization"""
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def get_valid_disparity_mask(flow, max_vertical_threshold=1.0, min_horizontal_flow=0.5):
    horizontal_flow = flow[..., 0]
    vertical_flow = flow[..., 1]

    valid_mask = (
        (np.abs(vertical_flow) < max_vertical_threshold) & 
        (np.abs(horizontal_flow) > min_horizontal_flow) & 
        (horizontal_flow > 0))

    return valid_mask

def disparity_to_depth(disparity, focal_length, baseline, valid_mask=None):
    """Convert disparity map to depth map"""
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (focal_length * baseline) / disparity
    
    if valid_mask is not None:
        depth[~valid_mask] = np.nan
    
    return depth

# Main execution
if __name__ == '__main__':
    # Load your stereo images
    img1 = cv2.imread('/content/DL Medical Assignment/frame_left.png')
    img2 = cv2.imread('/content/DL Medical Assignment/frame_right.png')

    # Check if images loaded properly
    if img1 is None or img2 is None:
        raise ValueError("Could not load images! Check file paths")

    # Resize to same dimensions if needed
    if img1.shape != img2.shape:
        h, w = img1.shape[:2]
        img2 = cv2.resize(img2, (w, h))

    # Load RAFT model
    model = load_raft('/content/models/raft-sintel.pth').to('cuda')
    model.eval()

    # Preprocess images
    img1_tensor = preprocess_image(img1)
    img2_tensor = preprocess_image(img2)

    # Compute optical flow
    flow = compute_optical_flow(model, img1_tensor, img2_tensor)

    # Step 3: Valid disparity mask
    valid_mask = get_valid_disparity_mask(flow)
    disparity = flow[..., 0]  # Horizontal flow = disparity
    masked_disparity = np.ma.masked_where(~valid_mask, disparity)

    # Step 4: Disparity to depth conversion
    focal_length = 1.03530811e+03  
    baseline = 4.14339018e+00 
    
    depth_map = disparity_to_depth(disparity, focal_length, baseline, valid_mask)

    # Visualization
    plt.figure(figsize=(15, 6))

    # Original Disparity
    plt.subplot(1, 4, 1)
    plt.imshow(disparity, cmap='jet')
    plt.colorbar(label='Disparity (pixels)')
    plt.title("Raw Disparity")

    # Valid Mask
    plt.subplot(1, 4, 2)
    plt.imshow(valid_mask, cmap='gray')
    plt.title("Valid Pixels Mask")

    # Filtered Disparity
    plt.subplot(1, 4, 3)
    plt.imshow(masked_disparity, cmap='jet')
    plt.colorbar(label='Valid Disparity (pixels)')
    plt.title("Filtered Disparity")

    # Depth Map
    plt.subplot(1, 4, 4)
    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar(label='Depth (meters)')
    plt.title("Depth Map")

    plt.tight_layout()
    plt.savefig('disparity_depth_results.png')
    print("Results saved to disparity_depth_results.png")
