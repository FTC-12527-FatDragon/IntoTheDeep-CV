import cv2
import numpy as np
import math
import time
from collections import defaultdict

# Track OpenCV function calls and timing
opencv_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})

def track_opencv(func_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            opencv_stats[func_name]['count'] += 1
            opencv_stats[func_name]['total_time'] += (end - start)
            return result
        return wrapper
    return decorator

# Wrap commonly used OpenCV functions
cv2.split = track_opencv('split')(cv2.split)
cv2.cvtColor = track_opencv('cvtColor')(cv2.cvtColor)
cv2.inRange = track_opencv('inRange')(cv2.inRange)
cv2.bitwise_and = track_opencv('bitwise_and')(cv2.bitwise_and)
cv2.bitwise_or = track_opencv('bitwise_or')(cv2.bitwise_or)
cv2.bitwise_not = track_opencv('bitwise_not')(cv2.bitwise_not)
cv2.morphologyEx = track_opencv('morphologyEx')(cv2.morphologyEx)
cv2.GaussianBlur = track_opencv('GaussianBlur')(cv2.GaussianBlur)
cv2.Sobel = track_opencv('Sobel')(cv2.Sobel)
cv2.Canny = track_opencv('Canny')(cv2.Canny)
cv2.findContours = track_opencv('findContours')(cv2.findContours)
cv2.drawContours = track_opencv('drawContours')(cv2.drawContours)
cv2.bilateralFilter = track_opencv('bilateralFilter')(cv2.bilateralFilter)
cv2.normalize = track_opencv('normalize')(cv2.normalize)
cv2.dilate = track_opencv('dilate')(cv2.dilate)
cv2.contourArea = track_opencv('contourArea')(cv2.contourArea)

# Camera settings
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 120

# Camera exposure settings
AUTO_EXPOSURE = 0.25
EXPOSURE = -5

# Camera gain settings
GAIN = 0  

# Camera white balance settings
AUTO_WB = 0
WB_RED = 1.0
WB_GREEN = 1.0
WB_BLUE = 1.0

# Edge detection parameters - initial values
CANNY_LOW = 120
CANNY_HIGH = 200
BLUR_SIZE = 13
SOBEL_KERNEL = 7

# Color detection ranges for different color spaces
HSV_BLUE_RANGE = ([90, 120, 40], [140, 255, 255])
HSV_RED_RANGE_1 = ([0, 120, 40], [10, 255, 255])  # Red wraps around in HSV
HSV_RED_RANGE_2 = ([170, 120, 40], [180, 255, 255])
HSV_YELLOW_RANGE = ([10, 120, 40], [30, 255, 255])

# Constants for filtering contours
SMALL_CONTOUR_AREA = 200

# Minimum average brightness threshold (0-255)
MIN_BRIGHTNESS_THRESHOLD = 60

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

def nothing(x):
    pass

def processFrame(frame):
    try:
        llpython = [0, 0, 0, 0, 0, 0, 0, 0]
        largest_contour = np.array([[]])
        frame = cv2.resize(frame, (256, 144))
        cv2.imshow('1. Original Frame', frame)
        
        blur_size = BLUR_SIZE
        sobel_kernel = SOBEL_KERNEL

        # Convert to HSV and denoise
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('2. HSV Conversion', hsv)
        
        # Split and display HSV channels
        h, s, v = cv2.split(hsv)
        cv2.imshow('2a. Hue Channel', h)
        cv2.imshow('2b. Saturation Channel', s) 
        cv2.imshow('2c. Value Channel', v)
        
        hsv_denoised = cv2.GaussianBlur(hsv, (5, 5), 0)
        cv2.imshow('3. HSV Denoised', hsv_denoised)
        hsv_denoised = hsv

        # Create masks for each color
        blue_mask = cv2.inRange(hsv_denoised, np.array(HSV_BLUE_RANGE[0]), np.array(HSV_BLUE_RANGE[1]))
        cv2.imshow('4. Blue Mask', blue_mask)
        
        red_mask1 = cv2.inRange(hsv_denoised, np.array(HSV_RED_RANGE_1[0]), np.array(HSV_RED_RANGE_1[1]))
        red_mask2 = cv2.inRange(hsv_denoised, np.array(HSV_RED_RANGE_2[0]), np.array(HSV_RED_RANGE_2[1]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        cv2.imshow('5. Red Mask', red_mask)
        
        yellow_mask = cv2.inRange(hsv_denoised, np.array(HSV_YELLOW_RANGE[0]), np.array(HSV_YELLOW_RANGE[1]))
        cv2.imshow('6. Yellow Mask', yellow_mask)

        # Combine all color masks
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(blue_mask, red_mask), yellow_mask)
        cv2.imshow('7. Combined Color Mask', combined_mask)
        
        kernel = np.ones((5,5), np.uint8)
        masked_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)
        cv2.imshow('8. Masked Original Frame', masked_frame)
        
        gray_masked = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('9. Gray Masked Frame', gray_masked)
        
        # Edge detection pipeline
        blurred = cv2.GaussianBlur(gray_masked, (blur_size, blur_size), 0)
        cv2.imshow('10. Blurred Frame', blurred)
        
        sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=sobel_kernel)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(magnitude * 255 / np.max(magnitude))
        cv2.imshow('11. Sobel Magnitude', magnitude)
        
        # Threshold the magnitude image
        _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        cv2.imshow('12. Thresholded Magnitude', edges)
        
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('13. Morphological Closing', edges)
        
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=3)
        cv2.imshow('14. Dilated Edges', edges)
        
        edges = cv2.bitwise_not(edges)
        edges = cv2.bitwise_and(edges, edges, mask=combined_mask)
        cv2.imshow('15. Final Edge Mask', edges)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_frame = frame.copy()
        game_pieces = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < SMALL_CONTOUR_AREA:
                continue
                
            for sep_contour in separate_touching_contours(contour):
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [sep_contour], -1, 255, -1)
                cv2.imshow('17. Current Contour Mask', mask)
                
                if cv2.mean(gray, mask=mask)[0] < MIN_BRIGHTNESS_THRESHOLD:
                    continue
                
                M = cv2.moments(sep_contour)
                if M["m00"] == 0:
                    continue
                    
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                angle = calculate_angle(sep_contour)
                area = cv2.contourArea(sep_contour)

                # Determine color of the piece by checking overlap with color masks
                roi_mask = np.zeros_like(gray)
                cv2.drawContours(roi_mask, [sep_contour], -1, 255, -1)
                cv2.imshow('18. ROI Mask', roi_mask)
                
                # Calculate percentage of overlap with each color mask
                total_area = cv2.countNonZero(roi_mask)
                blue_overlap = cv2.countNonZero(cv2.bitwise_and(blue_mask, roi_mask)) / total_area
                red_overlap = cv2.countNonZero(cv2.bitwise_and(red_mask, roi_mask)) / total_area
                yellow_overlap = cv2.countNonZero(cv2.bitwise_and(yellow_mask, roi_mask)) / total_area
                
                # Assign color based on highest overlap percentage (minimum 70% overlap required)
                color = "Unknown"
                max_overlap = max(blue_overlap, red_overlap, yellow_overlap)
                if max_overlap > 0.7:  # At least 70% of the contour must be the detected color
                    if blue_overlap == max_overlap:
                        color = "Blue"
                    elif red_overlap == max_overlap:
                        color = "Red"
                    elif yellow_overlap == max_overlap:
                        color = "Yellow"
                
                color_map = {"Blue": (255, 0, 0), "Red": (0, 0, 255), "Yellow": (0, 255, 255)}
                if color in color_map:  # Only draw if a valid color was detected
                    cv2.drawContours(contour_frame, [sep_contour], 0, color_map[color], 2)
                    draw_info(contour_frame, color, angle, center, len(game_pieces) + 1, area)
                    cv2.imshow('19. Current Detection', contour_frame)
                
                    game_pieces.append({
                        'index': len(game_pieces) + 1,
                        'color': color,
                        'position': center,
                        'angle': angle,
                        'area': area,
                        'brightness': cv2.mean(gray, mask=mask)[0],
                        'hierarchy_level': 'external' if hierarchy[0][i][3] == -1 else 'internal'
                    })
        
        cv2.imshow('16. Final Output', contour_frame)
        return contour_frame

    except Exception as e:
        print(frame, f"Error: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

def main():
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
    print(f"Camera settings:")
    print(f"Requested: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    print(f"Actual: {actual_width}x{actual_height} @ {actual_fps}fps")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        processed_frame = processFrame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()