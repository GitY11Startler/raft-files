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
    flow = compute_optical_flow(model, img1_tensor, img2_tensor)

    flow_img = visualize_flow(flow)

# 1. Compute valid disparity mask
valid_mask = get_valid_disparity_mask(flow)

# 2. Create masked disparity visualization
disparity = flow[..., 0]  # Horizontal flow = disparity
masked_disparity = np.ma.masked_where(~valid_mask, disparity)  # Mask invalid pixels

# 3. Create composite visualization (50% image + 50% disparity)
overlay = cv2.addWeighted(
    cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), 0.5,
    np.uint8(valid_mask * 255), 0.5, 0
)

# Plot results
plt.figure(figsize=(12, 6))

# Original Disparity
plt.subplot(1, 3, 1)
plt.imshow(disparity, cmap='jet')
plt.colorbar(label='Disparity (pixels)')
plt.title("Raw Disparity Map")

# Valid Mask Overlay
plt.subplot(1, 3, 2)
plt.imshow(overlay, cmap='gray')
plt.title("Valid Pixels (White = Valid)")

# Masked Disparity
plt.subplot(1, 3, 3)
plt.imshow(masked_disparity, cmap='jet')
plt.colorbar(label='Valid Disparity (pixels)')
plt.title("Filtered Disparity")

plt.savefig('masked_output.png')
print("Output saved to masked_output.png")