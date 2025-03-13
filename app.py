import streamlit as st
import cv2
import numpy as np
import os
import plotly.graph_objects as go
from depth_estimation import estimate_depth
from generate_3d_model import generate_3d_model

# Ensure necessary directories exist
os.makedirs("assets", exist_ok=True)
os.makedirs("output", exist_ok=True)

st.title("📸 Image to 3D Face Reconstruction")

# File Upload
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_path = os.path.join("assets", uploaded_file.name)
    
    try:
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, caption="📷 Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"❌ Error saving uploaded file: {e}")
        st.stop()  # Stop execution if the file couldn't be saved

    # Depth Map Generation
    if st.button("🔍 Generate Depth Map"):
        try:
            depth_map = estimate_depth(image_path)
            if depth_map is None or len(depth_map.shape) != 2:
                st.error("❌ Error: Depth map has incorrect dimensions.")
            else:
                depth_map_path = "output/depth_map.png"
                depth_map_8bit = (depth_map * 255).astype(np.uint8)
                cv2.imwrite(depth_map_path, depth_map_8bit)

                st.image(depth_map_path, caption="🗺️ Depth Map", use_column_width=True)
                st.session_state["depth_map_path"] = depth_map_path
                st.success("✅ Depth map generated!")
        except FileNotFoundError:
            st.error("❌ Error: Depth estimation model file missing.")
        except Exception as e:
            st.error(f"❌ Error generating depth map: {e}")

    # 3D Model Generation
    if st.button("🎨 Generate 3D Model"):
        if "depth_map_path" in st.session_state:
            depth_map_path = st.session_state["depth_map_path"]
            try:
                X, Y, Z = generate_3d_model(depth_map_path, image_path)

                fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="gray")])
                fig.update_layout(
                    title="🌀 3D Model",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Depth",
                    ),
                    margin=dict(l=0, r=0, t=30, b=0),
                )

                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ 3D model displayed!")

            except FileNotFoundError:
                st.error("❌ Error: Depth map file not found.")
            except Exception as e:
                st.error(f"❌ Error generating 3D model: {e}")
        else:
            st.warning("⚠️ Generate a depth map first.")
