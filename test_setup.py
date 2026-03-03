import torch
import cv2
import numpy as np

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
try:
    import streamlit
    print("Streamlit:", streamlit.__version__)
except ImportError:
    print("Streamlit: not installed")
