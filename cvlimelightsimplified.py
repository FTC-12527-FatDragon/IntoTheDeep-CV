import cv2
import numpy as np

# Camera settings
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 120

# Color detection ranges for different color spaces
HSV_BLUE_RANGE = ([90, 120, 40], [140, 255, 255])
HSV_RED_RANGE_1 = ([0, 120, 40], [10, 255, 255])  # Red wraps around in HSV
HSV_RED_RANGE_2 = ([170, 120, 40], [180, 255, 255])
HSV_YELLOW_RANGE = ([10, 120, 40], [30, 255, 255])

# Area threshold for filtering small contours
MIN_CONTOUR_AREA = 100


def process_color_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        largest_contour = max(contours, key=cv2.contourArea) if contours else None
        return largest_contour
    else:
        return None


def runPipeline(frame, llrobot):
    try:
        # llrobot: 0.0 Red, 1.0 Blue, 2.0 Yellow
        llpython = [0, 0, 0, 0, 0, 0, 0, 0]

        # Apply Gaussian blur
        frame = cv2.GaussianBlur(frame, (5,5), 0)

        #cv2.imshow("blur", frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

        # Convert foreground to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for each color
        blue_mask = cv2.inRange(hsv, np.array(HSV_BLUE_RANGE[0]), np.array(HSV_BLUE_RANGE[1]))
        
        red_mask1 = cv2.inRange(hsv, np.array(HSV_RED_RANGE_1[0]), np.array(HSV_RED_RANGE_1[1]))
        red_mask2 = cv2.inRange(hsv, np.array(HSV_RED_RANGE_2[0]), np.array(HSV_RED_RANGE_2[1]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        yellow_mask = cv2.inRange(hsv, np.array(HSV_YELLOW_RANGE[0]), np.array(HSV_YELLOW_RANGE[1]))

        # Clean up color masks
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)


        if llrobot == 0.0:
            largest_contour = process_color_from_mask(red_mask)
        elif llrobot == 1.0:
            largest_contour = process_color_from_mask(blue_mask)
        elif llrobot == 2.0:
            largest_contour = process_color_from_mask(yellow_mask)
        else:
            largest_contour = None
        return largest_contour, frame, llpython

    except Exception as e:
        print(f"Error: {str(e)}")
        return np.array([[]]), frame, [0, 0, 0, 0, 0, 0, 0, 0]


# def main():
#     cap = cv2.VideoCapture(0)
    
#     # Set camera properties
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
#     cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

#     actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
#     print(f"Camera settings:")
#     print(f"Requested: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
#     print(f"Actual: {actual_width}x{actual_height} @ {actual_fps}fps")

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             print("Failed to grab frame")
#             break

#         _, processed_frame, _ = runPipeline(frame, [])
#         cv2.imshow('16. Final Output', processed_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()