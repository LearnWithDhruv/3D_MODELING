import numpy as np
import cv2

def enhance_face_depth(depth_map):
    """Refine the depth map to enhance facial features."""
    # Ensure the input is uint8 for bilateralFilter (since depth_map is read as grayscale)
    depth_map = cv2.bilateralFilter(depth_map, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    depth_map = clahe.apply(depth_map)
    return depth_map

def detect_face(image_path):
    """Detect face and return bounding box (x, y, w, h)"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return x, y, w, h
    return None

def generate_3d_model(depth_map_path, image_path):
    """Generate a refined 3D model focusing on the face area."""
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise ValueError("❌ Error loading depth map!")

    # Get original image dimensions
    original_image = cv2.imread(image_path)
    orig_h, orig_w = original_image.shape[:2]

    # Depth map is 512x512
    depth_h, depth_w = 512, 512

    # Detect face in the original image
    face_box = detect_face(image_path)

    if face_box:
        x, y, w, h = face_box

        # Calculate scaling factors
        scale_x = depth_w / orig_w
        scale_y = depth_h / orig_h

        # Scale coordinates to depth map dimensions
        x_depth = max(0, min(depth_w - 1, int(x * scale_x)))
        y_depth = max(0, min(depth_h - 1, int(y * scale_y)))
        w_depth = max(1, int(w * scale_x))
        h_depth = max(1, int(h * scale_y))

        # Ensure the slice doesn’t exceed depth map bounds
        w_depth = min(w_depth, depth_w - x_depth)
        h_depth = min(h_depth, depth_h - y_depth)

        # Extract and enhance the face region
        face_region = depth_map[y_depth:y_depth+h_depth, x_depth:x_depth+w_depth]
        enhanced_face = enhance_face_depth(face_region)
        depth_map[y_depth:y_depth+h_depth, x_depth:x_depth+w_depth] = enhanced_face
    
    # Normalize depth values for consistency
    depth_map = depth_map / 255.0
    
    # Create mesh grid for 3D plotting
    h, w = depth_map.shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    Z = cv2.GaussianBlur(depth_map, (5,5), 0)  # Additional smoothing

    return X, Y, Z