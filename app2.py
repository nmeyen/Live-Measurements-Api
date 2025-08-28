

import cv2
import numpy as np
import mediapipe as mp
import torch
from flask import Flask, request, jsonify
import torch.nn.functional as F

# Add near your other imports
try:
    import cv2.aruco as aruco
    ARUCO_AVAILABLE = True
except Exception:
    ARUCO_AVAILABLE = False

# --- Calibration constants ---
ARUCO_MARKER_SIDE_CM = 5.0     # print a 5 cm square ArUco marker (DICT_5X5_50 works well)
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
A4_ASPECT = A4_HEIGHT_CM / A4_WIDTH_CM  # ~1.414
MIN_A4_AREA_PCT = 0.03  # 3% of image area to avoid tiny noise rectangles


app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
pose = mp_pose.Pose(model_complexity=2)  # Improved accuracy
holistic = mp_holistic.Holistic()  # For refining pose

KNOWN_OBJECT_WIDTH_CM = 21.0  # A4 paper width in cm
FOCAL_LENGTH = 600  # Default focal length
DEFAULT_HEIGHT_CM = 152.0  # Default height if not provided

def detect_aruco_scale(frame: np.ndarray):
    """
    Detects a single ArUco marker and returns (scale_cm_per_px, "aruco").
    Requires opencv-contrib-python to be installed.
    """
    if not ARUCO_AVAILABLE:
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(corners) == 0:
        return None, None

    # Use the largest marker by perimeter
    perims = [cv2.arcLength(c[0].astype(np.float32), True) for c in corners]
    idx = int(np.argmax(perims))
    c = corners[idx][0]  # 4x2
    # Estimate side length in pixels as the mean of the four edges
    edges = [
        np.linalg.norm(c[0] - c[1]),
        np.linalg.norm(c[1] - c[2]),
        np.linalg.norm(c[2] - c[3]),
        np.linalg.norm(c[3] - c[0]),
    ]
    side_px = float(np.mean(edges))
    if side_px <= 1:
        return None, None

    scale_cm_per_px = ARUCO_MARKER_SIDE_CM / side_px
    return scale_cm_per_px, "aruco"


def detect_a4_scale(frame: np.ndarray):
    """
    Detects an A4-like rectangle by contour and aspect ratio.
    Returns (scale_cm_per_px, "a4") if found, else (None, None).
    """
    h, w = frame.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    best = None
    best_area = 0
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) != 4:
            continue

        x, y, w_box, h_box = cv2.boundingRect(approx)
        area = float(w_box * h_box)
        if area < MIN_A4_AREA_PCT * img_area:
            continue

        aspect = max(h_box, 1) / max(w_box, 1)
        if 0.9 * A4_ASPECT <= aspect <= 1.1 * A4_ASPECT or 0.9 * (1 / A4_ASPECT) <= aspect <= 1.1 * (1 / A4_ASPECT):
            if area > best_area:
                best_area = area
                best = (w_box, h_box)

    if best is None:
        return None, None

    w_box, h_box = best
    # Choose the side that matches the A4 orientation we detected
    # If the detected rectangle is taller than wide, map height→29.7cm; else width→21cm
    if h_box >= w_box:
        cm_per_px_h = A4_HEIGHT_CM / h_box
        return cm_per_px_h, "a4"
    else:
        cm_per_px_w = A4_WIDTH_CM / w_box
        return cm_per_px_w, "a4"


def compute_scale(frame: np.ndarray, results_front, image_height: int, user_height_cm: float):
    """
    Tries ArUco → A4 → height-based fallback.
    Returns (scale_cm_per_px, focal_length, method_string)
    """
    # 1) ArUco
    scale, method = detect_aruco_scale(frame)
    if scale:
        return float(scale), float(FOCAL_LENGTH), method

    # 2) A4 detection
    scale, method = detect_a4_scale(frame)
    if scale:
        return float(scale), float(FOCAL_LENGTH), method

    # 3) Height-based fallback using your existing logic
    if results_front and results_front.pose_landmarks:
        landmarks = results_front.pose_landmarks.landmark
        _, scale_factor = calculate_distance_using_height(landmarks, image_height, user_height_cm)
        return float(scale_factor), float(FOCAL_LENGTH), "height"

    # Last resort
    return 0.05, float(FOCAL_LENGTH), "default"


def circumference_at_scanline(frame, depth_map, y_px, center_x_rel, scale_factor, default_width_px, depth_map_size=(384,384)):
    """
    Finds width via threshold scan at y_px, adjusts with depth map if available,
    then returns circumference (ellipse approximation) in cm.
    """
    h, w = frame.shape[:2]
    y_px = int(np.clip(y_px, 0, h - 1))
    width_px = get_body_width_at_height(frame, y_px, center_x_rel)
    width_px = max(width_px, default_width_px)

    # Depth adjustment
    depth_ratio = 1.0
    if depth_map is not None:
        cx_px = int(center_x_rel * w)
        scale_y = depth_map_size[0] / h
        scale_x = depth_map_size[1] / w
        y_scaled = int(y_px * scale_y)
        x_scaled = int(cx_px * scale_x)
        if 0 <= y_scaled < depth_map_size[0] and 0 <= x_scaled < depth_map_size[1]:
            d_here = depth_map[y_scaled, x_scaled]
            max_d = np.max(depth_map)
            if max_d > 0:
                depth_ratio = 1.0 + 0.5 * (1.0 - d_here / max_d)

    # Elliptical approximation (same as your helper)
    width_cm = width_px * scale_factor
    estimated_depth_cm = width_cm * depth_ratio * 0.7
    a = width_cm / 2.0
    b = estimated_depth_cm / 2.0
    circumference_cm = round(2 * np.pi * np.sqrt((a*a + b*b)/2.0), 2)
    return circumference_cm, width_cm


# Load depth estimation model
def load_depth_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    return model

depth_model = load_depth_model()

def calibrate_focal_length(image, real_width_cm, detected_width_px):
    """Dynamically calibrates focal length using a known object."""
    return (detected_width_px * FOCAL_LENGTH) / real_width_cm if detected_width_px else FOCAL_LENGTH



def detect_reference_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        focal_length = calibrate_focal_length(image, KNOWN_OBJECT_WIDTH_CM, w)
        scale_factor = KNOWN_OBJECT_WIDTH_CM / w
        return scale_factor, focal_length
    return 0.05, FOCAL_LENGTH

def estimate_depth(image):
    """Uses AI-based depth estimation to improve circumference calculations."""
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    
    # Resize input to match MiDaS model input size
    input_tensor = F.interpolate(input_tensor, size=(384, 384), mode="bilinear", align_corners=False)

    with torch.no_grad():
        depth_map = depth_model(input_tensor)
    
    return depth_map.squeeze().numpy()

def calculate_distance_using_height(landmarks, image_height, user_height_cm):
    """Calculate distance using the user's known height."""
    top_head = landmarks[mp_pose.PoseLandmark.NOSE.value].y * image_height
    bottom_foot = max(
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    ) * image_height
    
    person_height_px = abs(bottom_foot - top_head)
    
    # Using the formula: distance = (actual_height_cm * focal_length) / height_in_pixels
    distance = (user_height_cm * FOCAL_LENGTH) / person_height_px
    
    # Calculate more accurate scale_factor based on known height
    scale_factor = user_height_cm / person_height_px
    
    return distance, scale_factor

def get_body_width_at_height(frame, height_px, center_x):
    """Scan horizontally at a specific height to find body edges."""
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    
    # Ensure height_px is within image bounds
    if height_px >= frame.shape[0]:
        height_px = frame.shape[0] - 1
    
    # Get horizontal line at the specified height
    horizontal_line = thresh[height_px, :]
    
    # Find left and right edges starting from center
    center_x = int(center_x * frame.shape[1])
    left_edge, right_edge = center_x, center_x
    
    # Scan from center to left
    for i in range(center_x, 0, -1):
        if horizontal_line[i] == 0:  # Found edge (black pixel)
            left_edge = i
            break
    
    # Scan from center to right
    for i in range(center_x, len(horizontal_line)):
        if horizontal_line[i] == 0:  # Found edge (black pixel)
            right_edge = i
            break
            
    width_px = right_edge - left_edge
    
    # If width is unreasonably small, apply a minimum width
    min_width = 0.1 * frame.shape[1]  # Minimum width as 10% of image width
    if width_px < min_width:
        width_px = min_width
        
    return width_px

# def calculate_measurements(results, scale_factor, image_width, image_height, depth_map, frame=None, user_height_cm=None):
#     landmarks = results.pose_landmarks.landmark

#     # If user's height is provided, use it to get a more accurate scale factor
#     if user_height_cm:
#         _, scale_factor = calculate_distance_using_height(landmarks, image_height, user_height_cm)

#     def pixel_to_cm(value):
#         return round(value * scale_factor, 2)
    
#     def calculate_circumference(width_px, depth_ratio=1.0):
#         """Estimate circumference using width and depth adjustment."""
#         # Using a simplified elliptical approximation: C ≈ 2π * sqrt((a² + b²)/2)
#         # where a is half the width and b is estimated depth
#         width_cm = width_px * scale_factor
#         estimated_depth_cm = width_cm * depth_ratio * 0.7  # Depth is typically ~70% of width for torso
#         half_width = width_cm / 2
#         half_depth = estimated_depth_cm / 2
#         return round(2 * np.pi * np.sqrt((half_width**2 + half_depth**2) / 2), 2)

#     measurements = {}

#     # Shoulder Width
#     left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
#     right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
#     shoulder_width_px = abs(left_shoulder.x * image_width - right_shoulder.x * image_width)
    
#     # Apply a slight correction factor for shoulders (they're usually detected well)
#     shoulder_correction = 1.1  # 10% wider
#     shoulder_width_px *= shoulder_correction
    
#     measurements["shoulder_width"] = pixel_to_cm(shoulder_width_px)

#     # Chest/Bust Measurement
#     chest_y_ratio = 0.15  # Approximately 15% down from shoulder to hip
#     chest_y = left_shoulder.y + (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y - left_shoulder.y) * chest_y_ratio
    
#     chest_correction = 1.15  # 15% wider than detected width
#     chest_width_px = abs((right_shoulder.x - left_shoulder.x) * image_width) * chest_correction
    
#     if frame is not None:
#         chest_y_px = int(chest_y * image_height)
#         center_x = (left_shoulder.x + right_shoulder.x) / 2
#         detected_width = get_body_width_at_height(frame, chest_y_px, center_x)
#         if detected_width > 0:
#             chest_width_px = max(chest_width_px, detected_width)
    
#     chest_depth_ratio = 1.0
#     if depth_map is not None:
#         chest_x = int(((left_shoulder.x + right_shoulder.x) / 2) * image_width)
#         chest_y_px = int(chest_y * image_height)
#         scale_y = 384 / image_height
#         scale_x = 384 / image_width
#         chest_y_scaled = int(chest_y_px * scale_y)
#         chest_x_scaled = int(chest_x * scale_x)
#         if 0 <= chest_y_scaled < 384 and 0 <= chest_x_scaled < 384:
#             chest_depth = depth_map[chest_y_scaled, chest_x_scaled]
#             max_depth = np.max(depth_map)
#             chest_depth_ratio = 1.0 + 0.5 * (1.0 - chest_depth / max_depth)
    
#     measurements["chest_width"] = pixel_to_cm(chest_width_px)
#     measurements["chest_circumference"] = calculate_circumference(chest_width_px, chest_depth_ratio)
    

#     # Waist Measurement
#     left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
#     right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

#     # Adjust waist_y_ratio to better reflect the natural waistline
#     waist_y_ratio = 0.35  # 35% down from shoulder to hip (higher than before)
#     waist_y = left_shoulder.y + (left_hip.y - left_shoulder.y) * waist_y_ratio

#     # Use contour detection to dynamically estimate waist width
#     if frame is not None:
#         waist_y_px = int(waist_y * image_height)
#         center_x = (left_hip.x + right_hip.x) / 2
#         detected_width = get_body_width_at_height(frame, waist_y_px, center_x)
#         if detected_width > 0:
#             waist_width_px = detected_width
#         else:
#             # Fallback to hip width if contour detection fails
#             waist_width_px = abs(right_hip.x - left_hip.x) * image_width * 0.9  # 90% of hip width
#     else:
#         # Fallback to hip width if no frame is provided
#         waist_width_px = abs(right_hip.x - left_hip.x) * image_width * 0.9  # 90% of hip width

#     # Apply 30% correction factor to waist width
#     waist_correction = 1.16  # 30% wider
#     waist_width_px *= waist_correction

#     # Get depth adjustment for waist if available
#     waist_depth_ratio = 1.0
#     if depth_map is not None:
#         waist_x = int(((left_hip.x + right_hip.x) / 2) * image_width)
#         waist_y_px = int(waist_y * image_height)
#         scale_y = 384 / image_height
#         scale_x = 384 / image_width
#         waist_y_scaled = int(waist_y_px * scale_y)
#         waist_x_scaled = int(waist_x * scale_x)
#         if 0 <= waist_y_scaled < 384 and 0 <= waist_x_scaled < 384:
#             waist_depth = depth_map[waist_y_scaled, waist_x_scaled]
#             max_depth = np.max(depth_map)
#             waist_depth_ratio = 1.0 + 0.5 * (1.0 - waist_depth / max_depth)

#     measurements["waist_width"] = pixel_to_cm(waist_width_px)
#     measurements["waist"] = calculate_circumference(waist_width_px, waist_depth_ratio)
#     # Hip Measurement
#     hip_correction = 1.35  # Hips are typically 35% wider than detected landmarks
#     hip_width_px = abs(left_hip.x * image_width - right_hip.x * image_width) * hip_correction
    
#     if frame is not None:
#         hip_y_offset = 0.1  # 10% down from hip landmarks
#         hip_y = left_hip.y + (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y - left_hip.y) * hip_y_offset
#         hip_y_px = int(hip_y * image_height)
#         center_x = (left_hip.x + right_hip.x) / 2
#         detected_width = get_body_width_at_height(frame, hip_y_px, center_x)
#         if detected_width > 0:
#             hip_width_px = max(hip_width_px, detected_width)
    
#     hip_depth_ratio = 1.0
#     if depth_map is not None:
#         hip_x = int(((left_hip.x + right_hip.x) / 2) * image_width)
#         hip_y_px = int(left_hip.y * image_height)
#         hip_y_scaled = int(hip_y_px * scale_y)
#         hip_x_scaled = int(hip_x * scale_x)
#         if 0 <= hip_y_scaled < 384 and 0 <= hip_x_scaled < 384:
#             hip_depth = depth_map[hip_y_scaled, hip_x_scaled]
#             max_depth = np.max(depth_map)
#             hip_depth_ratio = 1.0 + 0.5 * (1.0 - hip_depth / max_depth)
    
#     measurements["hip_width"] = pixel_to_cm(hip_width_px)
#     measurements["hip"] = calculate_circumference(hip_width_px, hip_depth_ratio)

#     # Other measurements (unchanged)
#     neck = landmarks[mp_pose.PoseLandmark.NOSE.value]
#     left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
#     neck_width_px = abs(neck.x * image_width - left_ear.x * image_width) * 2.0
#     measurements["neck"] = calculate_circumference(neck_width_px, 1.0)
#     measurements["neck_width"] = pixel_to_cm(neck_width_px)

#     left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
#     sleeve_length_px = abs(left_shoulder.y * image_height - left_wrist.y * image_height)
#     measurements["arm_length"] = pixel_to_cm(sleeve_length_px)

#     shirt_length_px = abs(left_shoulder.y * image_height - left_hip.y * image_height) * 1.2
#     measurements["shirt_length"] = pixel_to_cm(shirt_length_px)

#      # Thigh Circumference (improved with depth information)
#     thigh_y_ratio = 0.2  # 20% down from hip to knee
#     left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
#     thigh_y = left_hip.y + (left_knee.y - left_hip.y) * thigh_y_ratio
    
#     # Apply correction factor for thigh width
#     thigh_correction = 1.2  # Thighs are typically wider than what can be estimated from front view
#     thigh_width_px = hip_width_px * 0.5 * thigh_correction  # Base thigh width on hip width
    
#     # Use contour detection if frame is available
#     if frame is not None:
#         thigh_y_px = int(thigh_y * image_height)
#         thigh_x = left_hip.x * 0.9  # Move slightly inward from hip
#         detected_width = get_body_width_at_height(frame, thigh_y_px, thigh_x)
#         if detected_width > 0 and detected_width < hip_width_px:  # Sanity check
#             thigh_width_px = detected_width  # Use detected width
    
#     # If depth map is available, use it for thigh measurement
#     thigh_depth_ratio = 1.0
#     if depth_map is not None:
#         thigh_x = int(left_hip.x * image_width)
#         thigh_y_px = int(thigh_y * image_height)
        
#         # Scale coordinates to match depth map size
#         thigh_y_scaled = int(thigh_y_px * scale_y)
#         thigh_x_scaled = int(thigh_x * scale_x)
        
#         if 0 <= thigh_y_scaled < 384 and 0 <= thigh_x_scaled < 384:
#             thigh_depth = depth_map[thigh_y_scaled, thigh_x_scaled]
#             max_depth = np.max(depth_map)
#             thigh_depth_ratio = 1.0 + 0.5 * (1.0 - thigh_depth / max_depth)
    
#     measurements["thigh"] = pixel_to_cm(thigh_width_px)
#     measurements["thigh_circumference"] = calculate_circumference(thigh_width_px, thigh_depth_ratio)


#     left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
#     trouser_length_px = abs(left_hip.y * image_height - left_ankle.y * image_height)
#     measurements["trouser_length"] = pixel_to_cm(trouser_length_px)

#     return measurements


def calculate_measurements(results, scale_factor, image_width, image_height, depth_map, frame=None, user_height_cm=None):
    landmarks = results.pose_landmarks.landmark

    # If user's height is provided, use it to get a more accurate scale factor
    if user_height_cm:
        _, scale_factor = calculate_distance_using_height(landmarks, image_height, user_height_cm)

    def pixel_to_cm(value):
        return round(value * scale_factor, 2)

    def calculate_circumference(width_px, depth_ratio=1.0):
        width_cm = width_px * scale_factor
        estimated_depth_cm = width_cm * depth_ratio * 0.7
        half_width = width_cm / 2
        half_depth = estimated_depth_cm / 2
        return round(2 * np.pi * np.sqrt((half_width**2 + half_depth**2) / 2), 2)

    measurements = {}

    # Shoulder Width
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_width_px = abs(left_shoulder.x * image_width - right_shoulder.x * image_width)
    shoulder_width_px *= 1.1  # correction
    measurements["shoulder_width"] = pixel_to_cm(shoulder_width_px)

    # Chest/Bust
    chest_y_ratio = 0.15
    chest_y = left_shoulder.y + (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y - left_shoulder.y) * chest_y_ratio
    chest_correction = 1.15
    chest_width_px = abs((right_shoulder.x - left_shoulder.x) * image_width) * chest_correction

    if frame is not None:
        chest_y_px = int(chest_y * image_height)
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        detected_width = get_body_width_at_height(frame, chest_y_px, center_x)
        if detected_width > 0:
            chest_width_px = max(chest_width_px, detected_width)

    chest_depth_ratio = 1.0
    if depth_map is not None:
        chest_x = int(((left_shoulder.x + right_shoulder.x) / 2) * image_width)
        chest_y_px = int(chest_y * image_height)
        scale_y = 384 / image_height
        scale_x = 384 / image_width
        chest_y_scaled = int(chest_y_px * scale_y)
        chest_x_scaled = int(chest_x * scale_x)
        if 0 <= chest_y_scaled < 384 and 0 <= chest_x_scaled < 384:
            chest_depth = depth_map[chest_y_scaled, chest_x_scaled]
            max_depth = np.max(depth_map)
            if max_depth > 0:
                chest_depth_ratio = 1.0 + 0.5 * (1.0 - chest_depth / max_depth)

    measurements["chest_width"] = pixel_to_cm(chest_width_px)
    measurements["chest_circumference"] = calculate_circumference(chest_width_px, chest_depth_ratio)

    # Waist
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    waist_y_ratio = 0.35
    waist_y = left_shoulder.y + (left_hip.y - left_shoulder.y) * waist_y_ratio

    if frame is not None:
        waist_y_px = int(waist_y * image_height)
        center_x = (left_hip.x + right_hip.x) / 2
        detected_width = get_body_width_at_height(frame, waist_y_px, center_x)
        if detected_width > 0:
            waist_width_px = detected_width
        else:
            waist_width_px = abs(right_hip.x - left_hip.x) * image_width * 0.9
    else:
        waist_width_px = abs(right_hip.x - left_hip.x) * image_width * 0.9

    waist_width_px *= 1.16
    waist_depth_ratio = 1.0
    if depth_map is not None:
        waist_x = int(((left_hip.x + right_hip.x) / 2) * image_width)
        waist_y_px = int(waist_y * image_height)
        scale_y = 384 / image_height
        scale_x = 384 / image_width
        waist_y_scaled = int(waist_y_px * scale_y)
        waist_x_scaled = int(waist_x * scale_x)
        if 0 <= waist_y_scaled < 384 and 0 <= waist_x_scaled < 384:
            waist_depth = depth_map[waist_y_scaled, waist_x_scaled]
            max_depth = np.max(depth_map)
            if max_depth > 0:
                waist_depth_ratio = 1.0 + 0.5 * (1.0 - waist_depth / max_depth)

    measurements["waist_width"] = pixel_to_cm(waist_width_px)
    measurements["waist"] = calculate_circumference(waist_width_px, waist_depth_ratio)

    # Hip
    hip_correction = 1.35
    hip_width_px = abs(left_hip.x * image_width - right_hip.x * image_width) * hip_correction

    if frame is not None:
        hip_y_offset = 0.1
        hip_y = left_hip.y + (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y - left_hip.y) * hip_y_offset
        hip_y_px = int(hip_y * image_height)
        center_x = (left_hip.x + right_hip.x) / 2
        detected_width = get_body_width_at_height(frame, hip_y_px, center_x)
        if detected_width > 0:
            hip_width_px = max(hip_width_px, detected_width)

    hip_depth_ratio = 1.0
    if depth_map is not None:
        hip_x = int(((left_hip.x + right_hip.x) / 2) * image_width)
        hip_y_px = int(left_hip.y * image_height)
        scale_y = 384 / image_height
        scale_x = 384 / image_width
        hip_y_scaled = int(hip_y_px * scale_y)
        hip_x_scaled = int(hip_x * scale_x)
        if 0 <= hip_y_scaled < 384 and 0 <= hip_x_scaled < 384:
            hip_depth = depth_map[hip_y_scaled, hip_x_scaled]
            max_depth = np.max(depth_map)
            if max_depth > 0:
                hip_depth_ratio = 1.0 + 0.5 * (1.0 - hip_depth / max_depth)

    measurements["hip_width"] = pixel_to_cm(hip_width_px)
    measurements["hip"] = calculate_circumference(hip_width_px, hip_depth_ratio)

    # Neck
    neck = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    neck_width_px = abs(neck.x * image_width - left_ear.x * image_width) * 2.0
    measurements["neck"] = calculate_circumference(neck_width_px, 1.0)
    measurements["neck_width"] = pixel_to_cm(neck_width_px)

    # Arm & Shirt length
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    sleeve_length_px = abs(left_shoulder.y * image_height - left_wrist.y * image_height)
    measurements["arm_length"] = pixel_to_cm(sleeve_length_px)
    shirt_length_px = abs(left_shoulder.y * image_height - left_hip.y * image_height) * 1.2
    measurements["shirt_length"] = pixel_to_cm(shirt_length_px)

    # Thigh
    thigh_y_ratio = 0.2
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    thigh_y = left_hip.y + (left_knee.y - left_hip.y) * thigh_y_ratio
    thigh_correction = 1.2
    thigh_width_px = abs((right_hip.x - left_hip.x) * image_width) * 0.5 * thigh_correction

    if frame is not None:
        thigh_y_px = int(thigh_y * image_height)
        thigh_x_center = (left_hip.x + right_hip.x) / 2
        detected_width = get_body_width_at_height(frame, thigh_y_px, thigh_x_center)
        if 0 < detected_width < hip_width_px:
            thigh_width_px = detected_width

    thigh_depth_ratio = 1.0
    if depth_map is not None:
        thigh_x = int(((left_hip.x + right_hip.x) / 2) * image_width)
        thigh_y_px = int(thigh_y * image_height)
        scale_y = 384 / image_height
        scale_x = 384 / image_width
        thigh_y_scaled = int(thigh_y_px * scale_y)
        thigh_x_scaled = int(thigh_x * scale_x)
        if 0 <= thigh_y_scaled < 384 and 0 <= thigh_x_scaled < 384:
            thigh_depth = depth_map[thigh_y_scaled, thigh_x_scaled]
            max_depth = np.max(depth_map)
            if max_depth > 0:
                thigh_depth_ratio = 1.0 + 0.5 * (1.0 - thigh_depth / max_depth)

    measurements["thigh_width"] = pixel_to_cm(thigh_width_px)
    measurements["thigh_circumference"] = calculate_circumference(thigh_width_px, thigh_depth_ratio)

    # NEW: Knee circumference at the knee landmark level
    if frame is not None:
        knee_y_px = int(((left_knee.y + landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y) / 2) * image_height)
        knee_center_x_rel = (left_hip.x + right_hip.x) / 2
        knee_circ, knee_width_cm = circumference_at_scanline(
            frame, depth_map, knee_y_px, knee_center_x_rel, scale_factor,
            default_width_px=hip_width_px * 0.35
        )
        measurements["knee"] = knee_circ
        measurements["knee_width"] = round(knee_width_cm, 2)

    # NEW: Calf circumference (midpoint between knee and ankle)
    if frame is not None:
        calf_y_rel = (left_knee.y + left_ankle.y) / 2
        calf_y_px = int(calf_y_rel * image_height)
        calf_center_x_rel = (left_hip.x + right_hip.x) / 2
        calf_circ, calf_width_cm = circumference_at_scanline(
            frame, depth_map, calf_y_px, calf_center_x_rel, scale_factor,
            default_width_px=hip_width_px * 0.3
        )
        measurements["calf"] = calf_circ
        measurements["calf_width"] = round(calf_width_cm, 2)

    # NEW: Ankle circumference
    if frame is not None:
        ankle_y_px = int(((left_ankle.y + landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y) / 2) * image_height)
        ankle_center_x_rel = (left_hip.x + right_hip.x) / 2
        ankle_circ, ankle_width_cm = circumference_at_scanline(
            frame, depth_map, ankle_y_px, ankle_center_x_rel, scale_factor,
            default_width_px=hip_width_px * 0.25
        )
        measurements["ankle"] = ankle_circ
        measurements["ankle_width"] = round(ankle_width_cm, 2)

    # Trouser length
    trouser_length_px = abs(left_hip.y * image_height - left_ankle.y * image_height)
    measurements["trouser_length"] = pixel_to_cm(trouser_length_px)

    # NEW: Inseam (inside leg) approximated as mid-hip to ankle
    mid_hip_y = ((left_hip.y + right_hip.y) / 2) * image_height
    mid_hip_x = ((left_hip.x + right_hip.x) / 2) * image_width
    left_ankle_px = np.array([left_ankle.x * image_width, left_ankle.y * image_height])
    inseam_px = np.linalg.norm(np.array([mid_hip_x, mid_hip_y]) - left_ankle_px)
    measurements["inseam"] = pixel_to_cm(inseam_px)

    return measurements

def to_iso_8559(measurements: dict, user_height_cm: float):
    """
    Builds a minimal ISO 8559-1 style set (names commonly used in apparel).
    All values in centimeters.
    """
    iso = {
        "stature": round(float(user_height_cm), 2),                         # body height
        "neck_girth": measurements.get("neck"),
        "bust_girth": measurements.get("chest_circumference"),
        "waist_girth": measurements.get("waist"),
        "hip_girth": measurements.get("hip"),
        "across_shoulder_breadth": measurements.get("shoulder_width"),
        "arm_length": measurements.get("arm_length"),
        "inside_leg_length": measurements.get("inseam"),
        "thigh_girth": measurements.get("thigh_circumference"),
        "knee_girth": measurements.get("knee"),
        "calf_girth": measurements.get("calf"),
        "ankle_girth": measurements.get("ankle"),
        "shirt_length_back": measurements.get("shirt_length"),
        "trouser_outseam": measurements.get("trouser_length"),
    }
    # Remove Nones for cleanliness
    return {k: round(v, 2) for k, v in iso.items() if v is not None}


def validate_front_image(image_np):
    """
    Basic validation for front image to ensure:
    - There is a person in the image
    - Not just a face/selfie (upper body visible)
    - Key upper landmarks are detected
    """
    try:
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_height, image_width = image_np.shape[:2]
        
        # Process with MediaPipe Holistic
        with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False) as holistic:
            
            results = holistic.process(rgb_frame)
        
        if not hasattr(results, 'pose_landmarks') or not results.pose_landmarks:
            return False, "No person detected. Please make sure you're clearly visible in the frame."

        # Minimum required upper body landmarks
        MINIMUM_LANDMARKS = [
            mp_holistic.PoseLandmark.NOSE,
            mp_holistic.PoseLandmark.LEFT_SHOULDER,
            mp_holistic.PoseLandmark.RIGHT_SHOULDER,
            mp_holistic.PoseLandmark.LEFT_ELBOW,
            mp_holistic.PoseLandmark.RIGHT_ELBOW,
            mp_holistic.PoseLandmark.RIGHT_KNEE,
            mp_holistic.PoseLandmark.LEFT_KNEE

           
        ]
        
        # Verify minimum landmarks are detected
        missing_upper = []
        for landmark in MINIMUM_LANDMARKS:
            landmark_data = results.pose_landmarks.landmark[landmark]
            if (landmark_data.visibility < 0.5 or
                landmark_data.x < 0 or 
                landmark_data.x > 1 or
                landmark_data.y < 0 or 
                landmark_data.y > 1):
                missing_upper.append(landmark.name.replace('_', ' '))
        
        if missing_upper:
            return False, f"Couldn't detect full body. Please make sure your full body is visible."

        # Check if this might be just a face/selfie (no torso)
        nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate approximate upper body size
        shoulder_width = abs(left_shoulder.x - right_shoulder.x) * image_width
        head_to_shoulder = abs(left_shoulder.y - nose.y) * image_height
        
        # If the shoulder width is small compared to head size, likely a selfie
        if shoulder_width < head_to_shoulder * 1.2:
            return False, "Please step back to show more of your upper body, not just your face."

        return True, "Validation passed - proceeding with measurements"
        
    except Exception as e:
        print(f"Error validating body image: {e}")
        return False, "You arent providing images correctly. Please try again."
    
@app.route("/upload_images", methods=["POST"])
def upload_images():
    if "front" not in request.files:
        return jsonify({"error": "Missing front image for reference."}), 400
    
    front_image_file = request.files["front"]
    front_image_np = np.frombuffer(front_image_file.read(), np.uint8)
    front_image_file.seek(0)  # Reset file pointer
    
    is_valid, error_msg = validate_front_image(cv2.imdecode(front_image_np, cv2.IMREAD_COLOR))
    
    if not is_valid:
        return jsonify({
            "error": error_msg,
            "pose": "front",
            "code": "INVALID_POSE"
        }), 400
    
    # Get user height if provided, otherwise use default
    user_height_cm = request.form.get('height_cm')
    print(user_height_cm)
    if user_height_cm:
        try:
            user_height_cm = float(user_height_cm)
        except ValueError:
            user_height_cm = DEFAULT_HEIGHT_CM
    else:
        user_height_cm = DEFAULT_HEIGHT_CM
    
    # received_images = {pose_name: request.files[pose_name] for pose_name in ["front", "left_side"] if pose_name in request.files}
    # measurements, scale_factor, focal_length, results = {}, None, FOCAL_LENGTH, {}
    # frames = {}
    
    # for pose_name, image_file in received_images.items():
    #     image_np = np.frombuffer(image_file.read(), np.uint8)
    #     frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    #     frames[pose_name] = frame  # Store the frame for contour detection
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results[pose_name] = holistic.process(rgb_frame)
    #     image_height, image_width, _ = frame.shape
        
    #     if pose_name == "front":
    #         # Always use height for calibration (default or provided)
    #         if results[pose_name].pose_landmarks:
    #             _, scale_factor = calculate_distance_using_height(
    #                 results[pose_name].pose_landmarks.landmark,
    #                 image_height,
    #                 user_height_cm
    #             )
    #         else:
    #             # Fallback to object detection only if pose landmarks aren't detected
    #             scale_factor, focal_length = detect_reference_object(frame)
        
    #     depth_map = estimate_depth(frame) if pose_name in ["front", "left_side"] else None
        
    #     if results[pose_name].pose_landmarks:
    #         if pose_name == "front":
    #             measurements.update(calculate_measurements(
    #                 results[pose_name], 
    #                 scale_factor, 
    #                 image_width, 
    #                 image_height, 
    #                 depth_map,
    #                 frames[pose_name],  # Pass the frame for contour detection
    #                 user_height_cm
    #             ))
    
    # # Debug information to help troubleshoot measurements
    # debug_info = {
    #     "scale_factor": float(scale_factor) if scale_factor else None,
    #     "focal_length": float(focal_length),
    #     "user_height_cm": float(user_height_cm)
    # }

    # print(measurements)
    
    # return jsonify({ 
    #     "measurements": measurements,
    #     "debug_info": debug_info
    # })

    received_images = {pose_name: request.files[pose_name] for pose_name in ["front", "left_side"] if pose_name in request.files}
    measurements, scale_factor, focal_length, results = {}, None, FOCAL_LENGTH, {}
    frames = {}
    calibration_method = "default"   # NEW

    for pose_name, image_file in received_images.items():
        image_np = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        frames[pose_name] = frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results[pose_name] = holistic.process(rgb_frame)
        image_height, image_width, _ = frame.shape

        if pose_name == "front":
        # NEW: robust scale computation (ArUco → A4 → height)
            scale_factor, focal_length, calibration_method = compute_scale(
            frame, results[pose_name], image_height, user_height_cm
        )

        depth_map = estimate_depth(frame) if pose_name in ["front", "left_side"] else None

        if results[pose_name].pose_landmarks:
            if pose_name == "front":
                measurements.update(calculate_measurements(
                results[pose_name],
                scale_factor,
                image_width,
                image_height,
                depth_map,
                frames[pose_name],
                user_height_cm
                ))

# After the loop, build ISO block
    iso_block = to_iso_8559(measurements, user_height_cm)

    debug_info = {
    "scale_factor_cm_per_px": float(scale_factor) if scale_factor else None,
    "focal_length": float(focal_length),
    "user_height_cm": float(user_height_cm),
    "calibration_method": calibration_method  # NEW
    }

    return jsonify({
    "measurements": measurements,    # keeps your original keys
    "iso_8559": iso_block,           # NEW standardized block
    "units": "cm",
    "debug_info": debug_info
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)