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

# Define color ranges for game pieces (adjust these based on your specific game pieces)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Constants
SMALL_CONTOUR_AREA = 1000
ABSOLUTE_SHADOW_THRESHOLD = 160
RELATIVE_SHADOW_THRESHOLD_FACTOR = 1
SHADOW_EDGE_DISTANCE = 10
ABSOLUTE_SHADOW_CONFIRMED_THRESHOLD = 30
DARK_THRESHOLD = 50 # Threshold for dark areas
CANNY_LOW = 50
CANNY_HIGH = 150

# Camera settings
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 120

def identify_color(hsv):
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    blue_count = cv2.countNonZero(mask_blue)
    red_count = cv2.countNonZero(mask_red)
    yellow_count = cv2.countNonZero(mask_yellow)
    
    if blue_count > red_count and blue_count > yellow_count:
        return "Blue", mask_blue
    elif red_count > yellow_count:
        return "Red", mask_red
    else:
        return "Yellow", mask_yellow

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
    # Fix the direction vector visualization
    cv2.line(image, center, (int(center[0] + 50 * math.cos(math.radians(90 - angle))), 
                             int(center[1] - 50 * math.sin(math.radians(90 - angle)))), (0, 255, 0), 2)

def separate_touching_contours(contour, min_area_ratio=0.15):
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = contour - [x, y]
    cv2.drawContours(mask, [shifted_contour], -1, 255, -1)
    
    # Visualize initial mask
    cv2.imshow('Initial Contour Mask', mask)
    
    original_area = cv2.contourArea(contour)
    max_contours = []
    max_count = 1
    
    # Use distance transform instead of erosion
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    
    # Visualize distance transform
    dist_normalized = np.zeros(dist_transform.shape, dtype=np.uint8)
    cv2.normalize(dist_transform, dist_normalized, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('Distance Transform', dist_normalized)
    
    # Create visualization for separation process
    separation_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    for threshold in np.linspace(0.1, 0.9, 9):
        _, thresh = cv2.threshold(dist_transform, threshold * dist_transform.max(), 255, 0)
        thresh = np.uint8(thresh)
        
        # Visualize threshold result
        cv2.imshow(f'Threshold {threshold:.1f}', thresh)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > original_area * min_area_ratio]
        
        # Draw contours on visualization
        separation_vis.fill(0)
        cv2.drawContours(separation_vis, valid_contours, -1, (0, 255, 0), 2)
        cv2.imshow('Separation Process', separation_vis)
        
        if len(valid_contours) > max_count:
            max_count = len(valid_contours)
            max_contours = valid_contours
    
    # Show final separated contours
    if max_contours:
        final_vis = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.drawContours(final_vis, max_contours, -1, (0, 255, 0), 2)
        cv2.imshow('Final Separated Contours', final_vis)
        return [c + [x, y] for c in max_contours]
    return [contour]

def remove_shadows(image, mask, threshold):
    # Convert to grayscale for edge and dark area detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create dark mask
    _, dark_mask = cv2.threshold(gray, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Dark Mask', dark_mask)
    
    # Detect edges
    edges = cv2.Canny(dark_mask, CANNY_LOW, CANNY_HIGH)
    cv2.imshow('Edge Detection', edges)
    
    # Dilate edges to create edge mask
    kernel = np.ones((3,3), np.uint8)
    edge_mask = cv2.dilate(edges, kernel, iterations=1)
    cv2.imshow('Dilated Edge Mask', edge_mask)
    
    # Combine dark and edge masks
    combined_mask = cv2.bitwise_or(dark_mask, edge_mask)
    cv2.imshow('Combined Mask', combined_mask)
    
    # Invert the combined mask to keep non-edge, non-dark areas
    clean_mask = cv2.bitwise_not(combined_mask)
    
    # Apply the clean mask to the original mask
    mask = cv2.bitwise_and(mask, clean_mask)
    
    # Now proceed with shadow removal on the cleaned mask
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('Threshold', thresh)
    
    _, absolute_thresh = cv2.threshold(gray, ABSOLUTE_SHADOW_CONFIRMED_THRESHOLD, 255, cv2.THRESH_BINARY)
    cv2.imshow('Absolute Threshold', absolute_thresh)
    
    initial_mask = cv2.bitwise_and(mask, absolute_thresh)
    cv2.imshow('Initial Mask', initial_mask)
    
    # Calculate distance transform
    dist_transform = cv2.distanceTransform(initial_mask, cv2.DIST_L2, 5)
    
    # Normalize distance transform for visualization
    dist_normalized = np.zeros(dist_transform.shape, dtype=np.uint8)
    cv2.normalize(dist_transform, dist_normalized, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('Distance Transform', dist_normalized)
    
    # Threshold the distance transform
    _, noshadow_mask = cv2.threshold(dist_transform, SHADOW_EDGE_DISTANCE, 255, cv2.THRESH_BINARY)
    noshadow_mask = np.uint8(noshadow_mask)
    cv2.imshow('No Shadow Mask', noshadow_mask)
    
    final_mask = cv2.bitwise_and(initial_mask, thresh)
    final_mask = cv2.bitwise_or(final_mask, noshadow_mask)
    cv2.imshow('Final Mask', final_mask)
    
    result = cv2.bitwise_and(image, image, mask=final_mask)
    cv2.imshow('Shadow Removal Result', result)
    return result

def process_frame(frame):
    try:
        loop_start = time.time()
        
        # Step 1: Convert to HSV and blur
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        cv2.imshow('Step 1: Blurred HSV', blurred)

        # Step 2: Create color masks
        mask_blue = cv2.inRange(blurred, lower_blue, upper_blue)
        mask_red1 = cv2.inRange(blurred, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(blurred, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(blurred, lower_yellow, upper_yellow)
        cv2.imshow('Step 2: Blue Mask', mask_blue)
        cv2.imshow('Step 2: Red Mask', mask_red)
        cv2.imshow('Step 2: Yellow Mask', mask_yellow)

        # Step 3: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        cv2.imshow('Step 3: Morphed Blue Mask', mask_blue)
        cv2.imshow('Step 3: Morphed Red Mask', mask_red)
        cv2.imshow('Step 3: Morphed Yellow Mask', mask_yellow)

        # Step 4: Remove shadows
        blue_no_shadow = remove_shadows(frame, mask_blue, ABSOLUTE_SHADOW_THRESHOLD)
        red_no_shadow = remove_shadows(frame, mask_red, ABSOLUTE_SHADOW_THRESHOLD)
        yellow_no_shadow = remove_shadows(frame, mask_yellow, ABSOLUTE_SHADOW_THRESHOLD)
        cv2.imshow('Step 4: Blue No Shadow', blue_no_shadow)
        cv2.imshow('Step 4: Red No Shadow', red_no_shadow)
        cv2.imshow('Step 4: Yellow No Shadow', yellow_no_shadow)

        # Step 5: Convert to grayscale
        blue_gray = cv2.cvtColor(blue_no_shadow, cv2.COLOR_BGR2GRAY)
        red_gray = cv2.cvtColor(red_no_shadow, cv2.COLOR_BGR2GRAY)
        yellow_gray = cv2.cvtColor(yellow_no_shadow, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Step 5: Blue Gray', blue_gray)
        cv2.imshow('Step 5: Red Gray', red_gray)
        cv2.imshow('Step 5: Yellow Gray', yellow_gray)

        # Step 6: Find contours
        contours_blue, _ = cv2.findContours(blue_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(red_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(yellow_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Visualize initial contours
        contours_vis = frame.copy()
        cv2.drawContours(contours_vis, contours_blue, -1, (255, 0, 0), 2)
        cv2.drawContours(contours_vis, contours_red, -1, (0, 0, 255), 2)
        cv2.drawContours(contours_vis, contours_yellow, -1, (0, 255, 255), 2)
        cv2.imshow('Step 6: Initial Contours', contours_vis)

        # Step 7: Separate touching contours
        separated_blue = [c for contour in contours_blue if cv2.contourArea(contour) > SMALL_CONTOUR_AREA for c in separate_touching_contours(contour)]
        separated_red = [c for contour in contours_red if cv2.contourArea(contour) > SMALL_CONTOUR_AREA for c in separate_touching_contours(contour)]
        separated_yellow = [c for contour in contours_yellow if cv2.contourArea(contour) > SMALL_CONTOUR_AREA for c in separate_touching_contours(contour)]
        
        # Visualize separated contours
        separated_vis = frame.copy()
        cv2.drawContours(separated_vis, separated_blue, -1, (255, 0, 0), 2)
        cv2.drawContours(separated_vis, separated_red, -1, (0, 0, 255), 2)
        cv2.drawContours(separated_vis, separated_yellow, -1, (0, 255, 255), 2)
        cv2.imshow('Step 7: Separated Contours', separated_vis)

        # Step 8: Process game pieces
        game_pieces = []
        result_frame = frame.copy()

        for color, contours in [("Blue", separated_blue), ("Red", separated_red), ("Yellow", separated_yellow)]:
            for contour in contours:
                angle = calculate_angle(contour)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                else:
                    continue
                area = cv2.contourArea(contour)
                cv2.drawContours(result_frame, [contour], 0, (0, 255, 0), 2)
                draw_info(result_frame, color, angle, center, len(game_pieces) + 1, area)
                game_pieces.append({
                    'index': len(game_pieces) + 1,
                    'color': color,
                    'position': center,
                    'angle': angle,
                    'area': area
                })

        cv2.imshow('Step 8: Result', result_frame)

        # Calculate and print loop time
        loop_time = time.time() - loop_start
        print(f"Loop time: {loop_time:.3f} seconds ({1/loop_time:.1f} FPS)")
        
        # Print OpenCV function statistics
        print("\nOpenCV Function Statistics:")
        for func_name, stats in opencv_stats.items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            total_time = stats['total_time']
            print(f"{func_name}: {stats['count']} calls, avg time: {avg_time*1000:.2f}ms, total time: {total_time*1000:.2f}ms")

    except Exception as e:
        cv2.putText(frame, f"Error: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return result_frame

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

        processed_frame = process_frame(frame)

        cv2.imshow('Processed Frame', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()