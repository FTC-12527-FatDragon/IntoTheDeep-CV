import cv2
import time
import os

def capture_video(url, duration_seconds=10, output_dir="recordings"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Initialize video capture
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return
        
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"recording_{timestamp}.mp4")
    out = cv2.VideoWriter(output_path, 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, (frame_width, frame_height))
    
    start_time = time.time()
    frames_captured = 0
    
    print(f"Starting {duration_seconds} second recording...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Write frame
        out.write(frame)
        frames_captured += 1
        
        # Check if duration reached
        if time.time() - start_time >= duration_seconds:
            break
            
    # Release everything
    cap.release()
    out.release()
    
    print(f"Recording complete: {frames_captured} frames captured")
    print(f"Saved to: {output_path}")

def main():
    url = 'http://limelight.local:5800'
    capture_video(url, duration_seconds=10)

if __name__ == "__main__":
    main()
