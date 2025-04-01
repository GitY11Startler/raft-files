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

def get_valid_disparity_mask(flow, max_vertical_threshold=0.5, min_horizontal_threshold=1, max_horizontal_threshold=50):
    '''Creates a mask of pixels where optical flow can be used as disparity'''
    horizontal_flow = flow[..., 0]
    vertical_flow = flow[..., 1]
    
    valid_mask = (
        (np.abs(vertical_flow) < max_vertical_threshold) & 
        (horizontal_flow > min_horizontal_threshold) & 
        (horizontal_flow < max_horizontal_threshold))
    
    return valid_mask

def disparity_to_depth(fx, B, cx_left, cx_right, disparity):
    """Compute depth from disparity, applying principal point offset correction."""
    delta_cx = cx_left - cx_right  # Principal offset
    disparity_corrected = disparity - delta_cx  # Corrected disparity
    
    # Avoid division by zero (set small values to a minimum disparity)
    disparity_corrected = np.maximum(disparity_corrected, 1e-6)

    # Compute depth
    depth = (fx * B) / disparity_corrected
    return depth

# Main execution
if __name__ == '__main__':
    # Load your stereo images (replace with your actual images)
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
    flow1 = compute_optical_flow(model, img1_tensor, img2_tensor)
    flow2 = compute_optical_flow(model, img2_tensor, img1_tensor)
    
    #Initializing some parameters for applying the masks
    max_v_thr = 0.5
    min_h_thr = flow1[..., 0].min()
    max_h_thr = flow1[..., 0].max()
    eps1 = 3.0
    eps2 = 4.0
    
    # 1. Compute valid disparity mask
    valid_mask1 = get_valid_disparity_mask(flow1, max_v_thr, min_h_thr+eps1, max_h_thr-eps1)
    valid_mask2 = get_valid_disparity_mask(flow2, max_v_thr, -max_h_thr+eps2, -min_h_thr-eps2)
    final_mask = valid_mask1 & valid_mask2

    # 2. Create masked disparity visualization
    disparity1 = np.where(final_mask, flow1[..., 0], 0)  # Horizontal flow = disparity
    disparity2 = np.where(final_mask, -flow2[..., 0], 0) 
    final_disparity = np.ma.masked_where(~final_mask, (disparity1+disparity2)/2)

    # Define camera parameters
    fx_left = 1035.31
    fx_right = 1035.17
    baseline = 4.14339018e-2 #baseline should be in meters, but it's given in centimeters
    cx_left = 596.96
    cx_right = 688.36

    # Compute depth maps
    depth_left = disparity_to_depth(fx_left, baseline, cx_left, cx_right, disparity1)
    depth_right = disparity_to_depth(fx_right, baseline, cx_left, cx_right, disparity2)
    final_depth = disparity_to_depth((fx_right+fx_left)/2, baseline, cx_left, cx_right, final_disparity)

    # Save the final depths set in a separate file
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(depth_left, cmap='viridis')
    plt.colorbar(label="Depth (meters)")
    plt.title("Depth Map (Left cam)")

    plt.subplot(1, 3, 2)
    plt.imshow(depth_right, cmap='viridis')
    plt.colorbar(label="Depth (meters)")
    plt.title("Depth Map (Right cam)")
    
    plt.subplot(1, 3, 3)
    plt.imshow(final_depth, cmap='viridis')
    plt.colorbar(label="Depth (meters)")
    plt.title("Final Depth Map")

    # Save third plot
    plt.savefig("depth_maps.png")
    print("Output saved to depth_maps.png")
    plt.close()
