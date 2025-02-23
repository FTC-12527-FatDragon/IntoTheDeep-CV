import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 10, 1000)

# Asymmetric motion profile parameters
accel_time = 1.0
cruise_time = 2.0
decel_time = 2.0
max_velocity = 1.0

# Calculate profile segments for forward motion
position = np.zeros_like(t)
velocity = np.zeros_like(t)
acceleration = np.zeros_like(t)

# Calculate end position of forward motion for later use
forward_end_pos = (max_velocity * accel_time / 2) + \
                 (max_velocity * cruise_time) + \
                 (max_velocity * decel_time - 0.5 * (max_velocity / decel_time) * decel_time**2)

for i, time in enumerate(t):
    if time < 5: # Forward motion in first half
        if time < accel_time:
            # Acceleration phase
            acceleration[i] = max_velocity / accel_time
            velocity[i] = (max_velocity / accel_time) * time
            position[i] = 0.5 * (max_velocity / accel_time) * time**2
            
        elif time < (accel_time + cruise_time):
            # Constant velocity phase
            acceleration[i] = 0
            velocity[i] = max_velocity
            position[i] = (max_velocity * accel_time / 2) + \
                         max_velocity * (time - accel_time)
            
        elif time < (accel_time + cruise_time + decel_time):
            # Deceleration phase
            decel = max_velocity / decel_time
            t_decel = time - (accel_time + cruise_time)
            acceleration[i] = -decel
            velocity[i] = max_velocity - decel * t_decel
            position[i] = (max_velocity * accel_time / 2) + \
                         (max_velocity * cruise_time) + \
                         (max_velocity * t_decel - 0.5 * decel * t_decel**2)
    else: # Reverse motion in second half
        rel_time = time - 5
        if rel_time < accel_time:
            # Acceleration phase
            acceleration[i] = -max_velocity / accel_time
            velocity[i] = -(max_velocity / accel_time) * rel_time
            position[i] = forward_end_pos - 0.5 * (max_velocity / accel_time) * rel_time**2
            
        elif rel_time < (accel_time + cruise_time):
            # Constant velocity phase
            acceleration[i] = 0
            velocity[i] = -max_velocity
            position[i] = forward_end_pos - ((max_velocity * accel_time / 2) + \
                         max_velocity * (rel_time - accel_time))
            
        elif rel_time < (accel_time + cruise_time + decel_time):
            # Deceleration phase
            decel = max_velocity / decel_time
            t_decel = rel_time - (accel_time + cruise_time)
            acceleration[i] = decel
            velocity[i] = -max_velocity + decel * t_decel
            position[i] = forward_end_pos - ((max_velocity * accel_time / 2) + \
                         (max_velocity * cruise_time) + \
                         (max_velocity * t_decel - 0.5 * decel * t_decel**2))

# Create single plot
plt.figure(figsize=(10, 6))
plt.plot(t, position, 'b-', label='Position')
plt.plot(t, velocity, 'g-', label='Velocity')
plt.plot(t, acceleration, 'r-', label='Acceleration')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Motion Profile - Forward and Reverse')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
