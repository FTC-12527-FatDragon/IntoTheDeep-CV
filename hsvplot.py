import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Create figure with two subplots
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

ax1.set_xlabel('Hue')
ax1.set_ylabel('Saturation') 
ax1.set_zlabel('Value')
ax1.set_xlim([0, 180])
ax1.set_ylim([0, 255])
ax1.set_zlim([0, 255])

# Initialize camera
cap = cv2.VideoCapture('http://limelight.local:5800')

def update(frame):
    # Clear previous points
    ax1.cla()
    ax2.cla()
    
    # Reset labels and limits
    ax1.set_xlabel('Hue')
    ax1.set_ylabel('Saturation')
    ax1.set_zlabel('Value')
    ax1.set_xlim([0, 180])
    ax1.set_ylim([0, 255])
    ax1.set_zlim([0, 255])
    
    # Read frame from camera
    ret, frame = cap.read()
    if ret:
        # Get frame dimensions
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2
        
        # Show camera feed
        ax2.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax2.axis('off')
        
        # Draw crosshair at center
        ax2.axvline(x=center_x, color='r', alpha=0.5)
        ax2.axhline(y=center_y, color='r', alpha=0.5)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get HSV value at center
        center_hsv = hsv[center_y, center_x]
        ax2.text(10, 30, f'HSV at center: {center_hsv}', 
                color='white', fontsize=10, 
                bbox=dict(facecolor='black', alpha=0.5))
        
        # Downsample to reduce number of points
        hsv = hsv[::10, ::10]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[::10, ::10]
        
        # Get H,S,V values
        h, s, v = cv2.split(hsv)
        
        # Normalize RGB values to 0-1 range for matplotlib
        rgb = rgb / 255.0
        
        # Plot points with their RGB colors
        ax1.scatter(h.flatten(), s.flatten(), v.flatten(), 
                   c=rgb.reshape(-1, 3),
                   marker='o', alpha=0.1, s=1)
        
        # Plot center point in red
        ax1.scatter(center_hsv[0], center_hsv[1], center_hsv[2],
                   color='red', marker='o', s=100)

# Create animation
ani = animation.FuncAnimation(fig, update, interval=50)

plt.show()

# Cleanup
cap.release()
cv2.destroyAllWindows()
