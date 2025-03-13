import cv2
import streamlit as st
import numpy as np
import os
from datetime import datetime

def capture_image():
    st.title("Capture Image for 3D Modeling")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        st.error("Could not access the camera")
        return None

    st.write("Press 'c' to capture, 'q' to exit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to capture image")
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("Press 'c' to capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            os.makedirs("assets", exist_ok=True)  # Ensure folder exists
            filename = f"assets/captured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            st.success(f"Image saved: {filename}")
            break
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return filename

if __name__ == "__main__":
    capture_image()
