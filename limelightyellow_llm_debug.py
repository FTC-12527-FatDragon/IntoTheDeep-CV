import cv2
import numpy as np
import math
import time
from collections import defaultdict

# Track OpenCV function calls and timing
# opencv_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})

# def track_opencv(func_name):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             start = time.time()
#             result = func(*args, **kwargs)
#             end = time.time()
#             opencv_stats[func_name]['count'] += 1
#             opencv_stats[func_name]['total_time'] += (end - start)
#             return result
#         return wrapper
#     return decorator

# Wrap commonly used OpenCV functions
# cv2.split = track_opencv('split')(cv2.split)
# cv2.cvtColor = track_opencv('cvtColor')(cv2.cvtColor)
# cv2.inRange = track_opencv('inRange')(cv2.inRange)
# cv2.bitwise_and = track_opencv('bitwise_and')(cv2.bitwise_and)
# cv2.bitwise_or = track_opencv('bitwise_or')(cv2.bitwise_or)
# cv2.bitwise_not = track_opencv('bitwise_not')(cv2.bitwise_not)
# cv2.morphologyEx = track_opencv('morphologyEx')(cv2.morphologyEx)
# cv2.GaussianBlur = track_opencv('GaussianBlur')(cv2.GaussianBlur)
# cv2.Sobel = track_opencv('Sobel')(cv2.Sobel)
# cv2.Canny = track_opencv('Canny')(cv2.Canny)
# cv2.findContours = track_opencv('findContours')(cv2.findContours)
# cv2.drawContours = track_opencv('drawContours')(cv2.drawContours)
# cv2.bilateralFilter = track_opencv('bilateralFilter')(cv2.bilateralFilter)
# cv2.normalize = track_opencv('normalize')(cv2.normalize)
# cv2.dilate = track_opencv('dilate')(cv2.dilate)
# cv2.contourArea = track_opencv('contourArea')(cv2.contourArea)

# Constants for filtering contours
SMALL_CONTOUR_AREA = 300

# Minimum average brightness threshold (0-255)
MIN_BRIGHTNESS_THRESHOLD = 50

# Color detection ranges for yellow in HSV
HSV_YELLOW_RANGE = ([15, 60, 100], [80, 255, 255])

# Edge detection parameters - initial values
BLUR_SIZE = 17
SOBEL_KERNEL = 3

# Aspect ratio range for contour filtering
MIN_ASPECT_RATIO = 1.5  # Minimum width/height ratio
MAX_ASPECT_RATIO = 6.0  # Maximum width/height ratio

def calculate_angle(contour):
    if len(contour) < 5:
        return 0
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    return angle

def draw_info(image, color, angle, center, index, area):
    cv2.putText(image, f"#{index}: {color}", (center[0] - 40, center[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, f"Angle: {angle:.2f}", (center[0] - 40, center[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, f"Area: {area:.2f}", (center[0] - 40, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(image, center, 5, (0, 255, 0), -1)
    cv2.line(image, center, (int(center[0] + 50 * math.cos(math.radians(90 - angle))), 
                             int(center[1] - 50 * math.sin(math.radians(90 - angle)))), (0, 255, 0), 2)

def separate_touching_contours(contour, min_area_ratio=0.15):
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = contour - [x, y]
    cv2.drawContours(mask, [shifted_contour], -1, 255, -1)

    original_area = cv2.contourArea(contour)
    max_contours = []
    max_count = 1

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    for threshold in np.linspace(0.1, 0.9, 9):
        _, thresh = cv2.threshold(dist_transform, threshold * dist_transform.max(), 255, 0)
        thresh = np.uint8(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > original_area * min_area_ratio]

        if len(valid_contours) > max_count:
            max_count = len(valid_contours)
            max_contours = valid_contours

    if max_contours:
        return [c + [x, y] for c in max_contours]
    return [contour]

def pipeline_debug_return(frame):
    return None, None, None, True, frame

def process_color(frame, mask):
    debug_info = None
    #return pipeline_debug_return(frame)
    kernel = np.ones((5, 5), np.uint8)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)  if 1 else frame
    #return pipeline_debug_return(masked_frame)
    gray_masked = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)  if 1 else masked_frame
    #return pipeline_debug_return(gray_masked)
    gray_boosted = cv2.addWeighted(gray_masked, 1.5, mask, 0.5, 0)  if 0 else gray_masked
    #return pipeline_debug_return(gray_boosted)
    blurred = cv2.GaussianBlur(gray_boosted, (3, 3), 0) if 1 else gray_boosted
    #return pipeline_debug_return(blurred)

    sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=1)
    sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=1)

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(magnitude * 255 / np.max(magnitude))
    #return pipeline_debug_return(magnitude)

    _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY) if 1 else magnitude
    #return pipeline_debug_return(edges)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) if 1 else edges
    #return pipeline_debug_return(edges)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) if 1 else edges
    #return pipeline_debug_return(edges)
    edges = cv2.bitwise_not(edges) if 1 else edges
    #return pipeline_debug_return(edges)
    edges = cv2.bitwise_and(edges, edges, mask=mask) if 1 else edges
    #return pipeline_debug_return(edges)
    edges = cv2.GaussianBlur(edges, (3, 3), 0) if 1 else edges
    #return pipeline_debug_return(edges)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy, gray_masked, False, debug_info


def debug_return(frame):
    return np.array([[]]), frame, [0, 0, 0, 0, 0, 0, 0, 0]


def runPipeline(frame, llrobot):
    try:
        # Initialize Limelight-style output
        llpython = [0, 0, 0, 0, 0, 0, 0, 0]
        largest_contour = np.array([[]])
        largest_area = 0
        
        # Convert to HSV and denoise
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #cv2.imshow('1. HSV', hsv)
        
        hsv_denoised = cv2.GaussianBlur(hsv, (5, 5), 0)
        #cv2.imshow('2. HSV Denoised', hsv_denoised)
        hsv_denoised = hsv
        # Create mask for yellow
        yellow_mask = cv2.inRange(hsv_denoised, np.array(HSV_YELLOW_RANGE[0]), np.array(HSV_YELLOW_RANGE[1]))
        yellow_mask = cv2.erode(yellow_mask, np.ones((3, 3), np.uint8))

        #cv2.imshow('3. Yellow Mask', yellow_mask)
        #return debug_return(yellow_mask)
        # Process yellow color
        yellow_contours, yellow_hierarchy, yellow_gray, isDebug, debug_info = process_color(frame, yellow_mask)
        if isDebug:
            return debug_return(debug_info)

        # Create a copy for contour visualization
        contour_frame = frame.copy()
        valid_contours = []
        #cv2.drawContours(frame, yellow_contours, -1, (0, 255, 0), 2)
        for i, contour in enumerate(yellow_contours):
            if cv2.contourArea(contour) < SMALL_CONTOUR_AREA:
                continue

            # Check aspect ratio using minAreaRect
            rect = cv2.minAreaRect(contour)
            width = max(rect[1])
            height = min(rect[1])
            if width == 0 or height == 0:
                continue
                
            aspect_ratio = width / height
            if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                continue

            for sep_contour in separate_touching_contours(contour):
                mask = np.zeros(yellow_gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [sep_contour], -1, 255, -1)
                #cv2.imshow('14. Contour Mask', mask)

                if cv2.mean(yellow_gray, mask=mask)[0] < MIN_BRIGHTNESS_THRESHOLD:
                    continue

                M = cv2.moments(sep_contour)
                if M["m00"] == 0:
                    continue

                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                angle = calculate_angle(sep_contour)
                area = cv2.contourArea(sep_contour)

                # Store valid contour info
                valid_contours.append({
                    'contour': sep_contour,
                    'center': center,
                    'angle': angle,
                    'area': area,
                    'index': i
                })

                # Update llpython and largest_contour if this is the largest valid contour
                if area > largest_area:
                    largest_area = area
                    largest_contour = sep_contour
                    llpython = [1, center[0], center[1], angle, len(yellow_contours), 0, 0, 0]

        # Draw all valid contours and their info
        for contour_info in valid_contours:
            cv2.drawContours(frame, [contour_info['contour']], -1, (0, 255, 0), 2)
            draw_info(frame, "Yellow", contour_info['angle'], contour_info['center'], 
                     contour_info['index'] + 1, contour_info['area'])

        #cv2.imshow('15. Contours', contour_frame)
        #cv2.imshow('16. Final Output', frame)
        return largest_contour, frame, llpython

    except Exception as e:
        print(f"Error: {str(e)}")
        return np.array([[]]), frame, [0, 0, 0, 0, 0, 0, 0, 0]