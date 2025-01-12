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

# Constants for filtering contours
SMALL_CONTOUR_AREA = 200

# Minimum average brightness threshold (0-255)
MIN_BRIGHTNESS_THRESHOLD = 60

# Color detection ranges for yellow in HSV
HSV_YELLOW_RANGE = ([10, 120, 50], [30, 255, 255])

# Edge detection parameters - initial values
BLUR_SIZE = 17
SOBEL_KERNEL = 3

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

def separate_touching_contours(contour, gray_masked, min_area_ratio=0.15):
    # 获取轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)
    
    # 创建掩码
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = contour - [x, y]
    cv2.drawContours(mask, [shifted_contour], -1, 255, -1)
    
    # 计算原始面积
    original_area = cv2.contourArea(contour)
    
    # 获取最小面积矩形以确定旋转角度
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    if rect[1][0] < rect[1][1]:
        angle = angle + 90
        
    # 创建旋转矩阵
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 旋转掩码和灰度图像
    rotated_mask = cv2.warpAffine(mask, M, (w, h))
    
    # 提取并旋转对应的灰度图像区域
    gray_roi = np.zeros((h, w), dtype=np.uint8)
    gray_roi[mask > 0] = gray_masked[y:y+h, x:x+w][mask > 0]
    rotated_gray = cv2.warpAffine(gray_roi, M, (w, h))
    
    # 计算垂直投影
    vertical_proj = np.sum(rotated_mask, axis=0)
    valid_region = vertical_proj > 0
    
    if not np.any(valid_region):
        return [contour]
    
    # 在有效区域内分析亮度
    start_x = np.where(valid_region)[0][0]
    end_x = np.where(valid_region)[0][-1]
    
    # 计算每列的平均亮度
    brightness_profile = []
    for i in range(start_x, end_x + 1):
        col_mask = rotated_mask[:, i] > 0
        if np.any(col_mask):
            avg_brightness = np.mean(rotated_gray[col_mask, i])
            brightness_profile.append(avg_brightness)
        else:
            brightness_profile.append(0)
    
    brightness_profile = np.array(brightness_profile)
    
    # 平滑亮度曲线
    smoothed = cv2.GaussianBlur(brightness_profile.astype(np.float32), (11, 1), 2)
    
    # 寻找显著的暗色区域（缝隙）
    valleys = []
    window = 15  # 搜索窗口大小
    
    for i in range(window, len(smoothed) - window):
        left_mean = np.mean(smoothed[i-window:i])
        right_mean = np.mean(smoothed[i:i+window])
        current = smoothed[i]
        
        # 如果当前点显著低于两侧，认为是缝隙
        if (current < left_mean * 0.75 and 
            current < right_mean * 0.75 and 
            current < np.mean(smoothed) * 0.8):
            valleys.append(i)
    
    # 如果找到缝隙，根据缝隙位置分割轮廓
    if valleys:
        separated_contours = []
        prev_x = start_x
        
        # 创建反向旋转矩阵
        M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
        
        # 根据缝隙位置分割
        for valley in valleys + [end_x - start_x]:
            # 创建当前部分的掩码
            part_mask = np.zeros_like(rotated_mask)
            part_mask[:, prev_x:valley + start_x] = rotated_mask[:, prev_x:valley + start_x]
            
            # 反向旋转
            part_mask = cv2.warpAffine(part_mask, M_inv, (w, h))
            
            # 找到轮廓
            part_contours, _ = cv2.findContours(part_mask, cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
            
            # 添加面积足够大的轮廓
            for pc in part_contours:
                if cv2.contourArea(pc) > original_area * min_area_ratio:
                    separated_contours.append(pc + [x, y])
            
            prev_x = valley + start_x
        
        if separated_contours:
            return separated_contours
    
    # 如果没有找到有效的分离点，返回原始轮廓
    return [contour]

def process_color(frame, mask):
    kernel = np.ones((5, 5), np.uint8)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    gray_masked = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

    gray_boosted = cv2.addWeighted(gray_masked, 1.5, mask, 0.5, 0)

    blurred = cv2.GaussianBlur(gray_boosted, (BLUR_SIZE, BLUR_SIZE), 0)

    sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL)
    sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL)

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(magnitude * 255 / np.max(magnitude))

    _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
    edges = cv2.bitwise_not(edges)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy, gray_masked

def runPipeline(frame, llrobot):
    try:
        # Initialize Limelight-style output
        llpython = [0, 0, 0, 0, 0, 0, 0, 0]
        largest_contour = np.array([[]])

        # Convert to HSV and denoise
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('1. HSV', hsv)
        
        hsv_denoised = cv2.GaussianBlur(hsv, (5, 5), 0)
        cv2.imshow('2. HSV Denoised', hsv_denoised)

        # Create mask for yellow
        yellow_mask = cv2.inRange(hsv_denoised, np.array(HSV_YELLOW_RANGE[0]), np.array(HSV_YELLOW_RANGE[1]))
        cv2.imshow('3. Yellow Mask', yellow_mask)

        # Process yellow color
        yellow_contours, yellow_hierarchy, yellow_gray = process_color(frame, yellow_mask)

        # Show intermediate processing steps
        kernel = np.ones((5, 5), np.uint8)
        masked_frame = cv2.bitwise_and(frame, frame, mask=yellow_mask)
        cv2.imshow('4. Masked Frame', masked_frame)
        
        gray_masked = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('5. Gray Masked', gray_masked)
        
        gray_boosted = cv2.addWeighted(gray_masked, 1.5, yellow_mask, 0.5, 0)
        cv2.imshow('6. Gray Boosted', gray_boosted)
        
        blurred = cv2.GaussianBlur(gray_boosted, (BLUR_SIZE, BLUR_SIZE), 0)
        cv2.imshow('7. Blurred', blurred)
        
        sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL)
        sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(magnitude * 255 / np.max(magnitude))
        cv2.imshow('8. Sobel Magnitude', magnitude)
        
        _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        cv2.imshow('9. Thresholded Edges', edges)
        
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('10. Morphology Close', edges)
        
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
        cv2.imshow('11. Dilated', edges)
        
        edges = cv2.bitwise_not(edges)
        cv2.imshow('12. Inverted', edges)
        
        edges = cv2.bitwise_and(edges, edges, mask=yellow_mask)
        cv2.imshow('13. Final Edges', edges)

        # Create a copy for contour visualization
        contour_frame = frame.copy()

        for i, contour in enumerate(yellow_contours):
            if cv2.contourArea(contour) < SMALL_CONTOUR_AREA:
                continue

            for sep_contour in separate_touching_contours(contour, yellow_gray):
                mask = np.zeros(yellow_gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [sep_contour], -1, 255, -1)
                cv2.imshow('14. Contour Mask', mask)

                if cv2.mean(yellow_gray, mask=mask)[0] < MIN_BRIGHTNESS_THRESHOLD:
                    continue

                M = cv2.moments(sep_contour)
                if M["m00"] == 0:
                    continue

                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                angle = calculate_angle(sep_contour)
                area = cv2.contourArea(sep_contour)

                # Draw contours on visualization frame
                cv2.drawContours(contour_frame, [sep_contour], -1, (0, 255, 0), 2)
                cv2.imshow('15. Contours', contour_frame)

                # Update llpython output with the first valid contour data
                llpython = [1, center[0], center[1], angle, len(yellow_contours), 0, 0, 0]

                # Update largest_contour with the first valid contour
                largest_contour = sep_contour

                # Draw information on the frame
                draw_info(frame, "Yellow", angle, center, i + 1, area)

                # Break after finding the first valid contour for simplicity
                break

            # If a valid contour was found, stop processing further contours
            if llpython[0] == 1:
                break

        cv2.imshow('16. Final Output', frame)
        return largest_contour, frame, llpython

    except Exception as e:
        print(f"Error: {str(e)}")
        return np.array([[]]), frame, [0, 0, 0, 0, 0, 0, 0, 0]
    


def main():
    cap = cv2.VideoCapture('http://limelight.local:5800')


    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        _, processed_frame, _ = runPipeline(frame, None)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()