import numpy as np
import torch
import cv2
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

    # Save flow visualization
    cv2.imwrite('optical_flow.png', flow_img)
    print("Saved optical flow to optical_flow.png")