import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

def load_midas_model():
    model_type = "DPT_Large"  # High-precision model
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()
    return model

def estimate_depth(image_path):
    """Estimate depth using MiDaS and return a 2D depth map resized to original image dimensions."""
    model = load_midas_model()

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # Store original dimensions
    original_height, original_width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),  # Resize to 512x512 for model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 512, 512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        depth_map = model(input_tensor)

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8)

    # Resize depth map to original image dimensions
    depth_map = cv2.resize(depth_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    return depth_map