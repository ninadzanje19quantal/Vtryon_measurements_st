import os
from typing import Tuple
import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tempfile import TemporaryDirectory


half_band = 5  # pixels

def remove_bg(input_path: str, output_folder: str) -> str:
    os.makedirs(output_folder, exist_ok=True)
    input_image = Image.open(input_path)
    output_image = remove(input_image)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}.png")
    output_image.save(output_path)
    return output_path

def extract_torso_component(mask: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    label_at_center = labels[center_y, center_x]
    torso_mask = (labels == label_at_center).astype(np.uint8) * 255
    return torso_mask

def get_torso_edges(mask: np.ndarray, y: int, center_x: float) -> Tuple[int, int]:
    cols = np.where(mask[y] > 0)[0]
    if len(cols) < 2:
        raise RuntimeError(f"No silhouette pixels found at row {y}")
    runs = []
    start = cols[0]
    for i in range(1, len(cols)):
        if cols[i] != cols[i-1] + 1:
            runs.append((start, cols[i-1]))
            start = cols[i]
    runs.append((start, cols[-1]))
    for x0, x1 in runs:
        if x0 <= center_x <= x1:
            return x0, x1
    return cols[0], cols[-1]

def elliptical_circumference(a: float, b: float) -> float:
    h = ((a - b)**2) / ((a + b)**2)
    return np.pi * (a + b) * (1 + (3*h) / (10 + np.sqrt(4 - 3*h)))

def process_image(path: str):
    import cv2
    import mediapipe as mp
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        rgb = img[:, :, :3]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        rgb = img

    h, w = mask.shape
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            raise RuntimeError("No pose landmarks detected")
        lm = res.pose_landmarks.landmark
        shoulder_y = int(((lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y +
                           lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y) * 0.5) * h)
        hip_y      = int(((lm[mp.solutions.pose.PoseLandmark.LEFT_HIP].y +
                           lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y) * 0.5) * h)
        shoulder_x = int(((lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x +
                           lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x) * 0.5) * w)
    return mask, shoulder_y, hip_y, shoulder_x, h

def mapper(chest_size: float, waist_size: float) -> dict:
    user_size = {}
    chest_size = int(chest_size)
    waist_size = int(waist_size)

    if 0 < chest_size <= 74:
        user_size["chest_size"] = (chest_size, "XXS")
    if 74 < chest_size <= 81:
        user_size["chest_size"] = (chest_size, "XS")
    if 81 < chest_size <= 89:
        user_size["chest_size"] = (chest_size, "S")
    if 89 < chest_size <= 97:
        user_size["chest_size"] = (chest_size, "M")
    if 97 < chest_size <= 107:
        user_size["chest_size"] = (chest_size, "L")
    if 107 < chest_size <= 119:
        user_size["chest_size"] = (chest_size, "XL")
    if 119 < chest_size <= 131:
        user_size["chest_size"] = (chest_size, "XXL")

    if waist_size <= 58:
        user_size["waist_size"] = [waist_size, "XXS"]
    if 58 < waist_size <= 64:
        user_size["waist_size"] = [waist_size, "XS"]
    if 64 < waist_size <= 72:
        user_size["waist_size"] = [waist_size, "S"]
    if 72 < waist_size <= 81:
        user_size["waist_size"] = [waist_size, "M"]
    if 81 < waist_size <= 90:
        user_size["waist_size"] = [waist_size, "L"]
    if 90 < waist_size <= 102:
        user_size["waist_size"] = [waist_size, "XL"]
    if 102 < waist_size <- 114:
        user_size["waist_size"] = (waist_size, "XXL")
    return user_size


from PIL import Image
import os
from tempfile import TemporaryDirectory
import numpy as np # Make sure numpy is imported

# Helper functions (remove_bg, process_image, etc.) are assumed to be defined elsewhere
# def remove_bg(input_path, output_dir): ...
# def process_image(image_path): ...
# def extract_torso_component(mask, cx, cy): ...
# def get_torso_edges(mask, y, cx): ...
# def elliptical_circumference(a, b): ...
# def mapper(chest_size, waist_size): ...

def calculate_body_measurements_from_uploads(
        front_image_upload,
        side_image_upload,
        height_cm: float,
        waist_search_band_px: int = 20
) -> dict:
    """
    Calculates approximate chest and waist circumference from Streamlit image uploads.

    Args:
        front_image_upload: The uploaded front-facing image file from Streamlit.
        side_image_upload: The uploaded side-facing image file from Streamlit.
        height_cm (float): The person's actual height in centimeters.
        waist_search_band_px (int): The pixel range to search for the narrowest
                                    part of the waist around the initial estimate.

    Returns:
        dict: A dictionary containing the approximate chest and waist circumferences.
    """
    with TemporaryDirectory() as tmpdir:
        # Create temporary file paths
        front_image_path = os.path.join(tmpdir, "front_image.png")
        side_image_path = os.path.join(tmpdir, "side_image.png")

        # Save the uploaded files to the temporary paths
        with Image.open(front_image_upload) as img:
            img.save(front_image_path)
        with Image.open(side_image_upload) as img:
            img.save(side_image_path)

        # --- From this point on, the rest of your original function logic can remain unchanged ---

        # 1. Remove background from both images
        front_clean_path = remove_bg(front_image_path, tmpdir)
        side_clean_path = remove_bg(side_image_path, tmpdir)

        # 2. Process images to get masks and key landmarks
        raw_f_mask, f_chest_y, f_waist_center, f_chest_cx, pixel_h = process_image(front_clean_path)
        s_mask, s_chest_y, s_waist_center, s_chest_cx, _ = process_image(side_clean_path)

        # 3. Isolate the torso from the rest of the body mask
        f_mask = extract_torso_component(raw_f_mask, f_chest_cx, f_chest_y)

        # 4. Get chest width measurements from both views
        f_chest_x0, f_chest_x1 = get_torso_edges(f_mask, f_chest_y, f_chest_cx)
        s_chest_x0, s_chest_x1 = get_torso_edges(s_mask, s_chest_y, s_chest_cx)
        f_chest_w = f_chest_x1 - f_chest_x0
        s_chest_w = s_chest_x1 - s_chest_x0

        # 5. Search for the narrowest part of the waist
        f_best_w, s_best_w = float('inf'), float('inf')
        for dy in range(-waist_search_band_px, waist_search_band_px + 1):
            yf = np.clip(f_waist_center + dy, 0, pixel_h - 1)
            ys = np.clip(s_waist_center + dy, 0, s_mask.shape[0] - 1)

            x0f, x1f = get_torso_edges(f_mask, yf, f_chest_cx)
            x0s, x1s = get_torso_edges(s_mask, ys, s_chest_cx)
            wf, ws = x1f - x0f, x1s - x0s

            if wf < f_best_w: f_best_w = wf
            if ws < s_best_w: s_best_w = ws

        # 6. Calculate circumferences and convert to real-world units
        chest_c_px = elliptical_circumference(f_chest_w / 2, s_chest_w / 2)
        waist_c_px = elliptical_circumference(f_best_w / 2, s_best_w / 2)
        scale = height_cm / pixel_h
        chest_cm = chest_c_px * scale
        waist_cm = waist_c_px * scale

        # 7. Return the final results
        return mapper(chest_size=chest_cm, waist_size=waist_cm)


import streamlit as st

# --- Place the function definition from above here ---

# Streamlit UI
st.title("Body Measurement Calculator")

# Inputs for images and height
front_upload = st.file_uploader("Upload Front View Image", type=["jpg", "jpeg", "png"], key="front")
side_upload = st.file_uploader("Upload Side View Image", type=["jpg", "jpeg", "png"], key="side")
height_input = st.number_input("Enter your height in cm", min_value=100.0, max_value=250.0, value=175.0)

# Button to trigger the calculation
if st.button("Calculate Measurements"):
    if front_upload is not None and side_upload is not None:
        with st.spinner('Analyzing images...'):
            # Display the uploaded images for confirmation
            st.image([front_upload, side_upload], caption=['Front View', 'Side View'], width=250)

            # Call the function with the uploaded file objects and height
            results = calculate_body_measurements_from_uploads(
                front_image_upload=front_upload,
                side_image_upload=side_upload,
                height_cm=float(height_input)
            )

            st.success("Calculation Complete!")
            st.write("Here are your approximate measurements:")
            st.json(results)  # Display results in a clean format
    else:
        st.error("Please upload both front and side images to continue.")

#print(calculate_body_measurements(front_image_path="3dImages/bg_removed/front9.png", side_image_path="3dImages/bg_removed/side9.png", height_cm=189.0))

