import pygame
import math
import numpy as np

class PoseCalculator:
    def __init__(self, field_image_path):
        pygame.init()
        
        # Set up display
        self.WIDTH = 800
        self.HEIGHT = 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Bot Pose Calculator")
        
        # Load and scale field image
        self.field_img = pygame.image.load(field_image_path)
        self.field_img = pygame.transform.scale(self.field_img, (self.WIDTH, self.HEIGHT))
        
        # Conversion factors between pixels and inches
        self.INCHES_TO_PIXELS = self.WIDTH / 144  # 144 inches = field width
        self.PIXELS_TO_INCHES = 144 / self.WIDTH
        
        # Colors
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        
        # Interactive elements
        self.dragging_target = False
        self.dragging_target_heading = False
        self.dragging_bot_heading = False
        self.dragging_distance = False
        self.active_input = None
        self.input_text = ""
        
        # Initial positions and values
        self.target_x = 0
        self.target_y = 0
        self.target_heading = 90
        self.bot_heading = 45
        self.distance = 24
        
        # Font for text
        self.font = pygame.font.Font(None, 24)
        
        # Input boxes
        self.input_boxes = {
            'target_x': pygame.Rect(200, 10, 60, 25),
            'target_y': pygame.Rect(200, 35, 60, 25),
            'target_heading': pygame.Rect(200, 60, 60, 25),
            'bot_heading': pygame.Rect(200, 85, 60, 25),
            'distance': pygame.Rect(200, 110, 60, 25)
        }
        
    def pixels_to_inches(self, px, py):
        """Convert pixel coordinates to inches coordinates"""
        x = (px / self.INCHES_TO_PIXELS) - 72
        y = 72 - (py / self.INCHES_TO_PIXELS)
        return (x, y)
        
    def inches_to_pixels(self, x, y):
        """Convert inches coordinates to pixel coordinates"""
        pixel_x = (x + 72) * self.INCHES_TO_PIXELS
        pixel_y = (72 - y) * self.INCHES_TO_PIXELS
        return (int(pixel_x), int(pixel_y))
        
    def calculate_bot_pose(self, target_x, target_y, target_heading, bot_heading, distance):
        """Calculate bot pose given target pose and bot measurements"""
        # Convert angles to radians
        target_rad = math.radians(target_heading)
        bot_rad = math.radians(bot_heading)
        
        # Calculate bot position relative to target
        dx = -distance * math.cos(bot_rad)
        dy = -distance * math.sin(bot_rad)
        
        # Transform to field coordinates
        bot_x = target_x + dx
        bot_y = target_y + dy
        
        return (bot_x, bot_y)
        
    def handle_mouse_input(self, event):
        """Handle mouse events for interactive controls"""
        target_pos = self.inches_to_pixels(self.target_x, self.target_y)
        bot_pos = self.inches_to_pixels(*self.calculate_bot_pose(
            self.target_x, self.target_y, self.target_heading, 
            self.bot_heading, self.distance))
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            
            # Check input boxes
            clicked_input = None
            for key, box in self.input_boxes.items():
                if box.collidepoint(event.pos):
                    clicked_input = key
                    break
                    
            if clicked_input:
                self.active_input = clicked_input
                self.input_text = str(getattr(self, clicked_input))
            else:
                self.active_input = None
                
                # Check other interactive elements
                if math.dist((x,y), target_pos) < 10:
                    self.dragging_target = True
                # Check if clicking near target heading line
                target_end = (
                    target_pos[0] + int(30 * math.cos(math.radians(self.target_heading))),
                    target_pos[1] - int(30 * math.sin(math.radians(self.target_heading)))
                )
                if math.dist((x,y), target_end) < 10:
                    self.dragging_target_heading = True
                # Check if clicking near bot heading line
                bot_end = (
                    bot_pos[0] + int(30 * math.cos(math.radians(self.bot_heading))),
                    bot_pos[1] - int(30 * math.sin(math.radians(self.bot_heading)))
                )
                if math.dist((x,y), bot_end) < 10:
                    self.dragging_bot_heading = True
                # Check if clicking near distance line
                mid_point = ((target_pos[0] + bot_pos[0])//2, (target_pos[1] + bot_pos[1])//2)
                if math.dist((x,y), mid_point) < 10:
                    self.dragging_distance = True
                    
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging_target = False
            self.dragging_target_heading = False
            self.dragging_bot_heading = False
            self.dragging_distance = False
            
        elif event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            if self.dragging_target:
                self.target_x, self.target_y = self.pixels_to_inches(x, y)
            elif self.dragging_target_heading:
                dx = x - target_pos[0]
                dy = target_pos[1] - y
                self.target_heading = math.degrees(math.atan2(dy, dx))
            elif self.dragging_bot_heading:
                dx = x - bot_pos[0]
                dy = bot_pos[1] - y
                self.bot_heading = math.degrees(math.atan2(dy, dx))
            elif self.dragging_distance:
                self.distance = math.dist(target_pos, (x,y)) * self.PIXELS_TO_INCHES
                
        elif event.type == pygame.KEYDOWN and self.active_input:
            if event.key == pygame.K_RETURN:
                try:
                    value = float(self.input_text)
                    setattr(self, self.active_input, value)
                except ValueError:
                    pass
                self.active_input = None
                self.input_text = ""
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                if event.unicode.isnumeric() or event.unicode in '.-':
                    self.input_text += event.unicode
        
    def visualize(self):
        """Visualize the target and bot poses"""
        self.screen.blit(self.field_img, (0,0))
        
        # Calculate positions
        target_pos = self.inches_to_pixels(self.target_x, self.target_y)
        bot_x, bot_y = self.calculate_bot_pose(
            self.target_x, self.target_y, self.target_heading,
            self.bot_heading, self.distance)
        bot_pos = self.inches_to_pixels(bot_x, bot_y)
        
        # Draw target
        target_rad = math.radians(self.target_heading)
        target_end = (
            target_pos[0] + int(30 * math.cos(target_rad)),
            target_pos[1] - int(30 * math.sin(target_rad))
        )
        pygame.draw.circle(self.screen, self.RED, target_pos, 5)
        pygame.draw.line(self.screen, self.RED, target_pos, target_end, 2)
        
        # Draw bot
        bot_rad = math.radians(self.bot_heading)
        bot_end = (
            bot_pos[0] + int(30 * math.cos(bot_rad)),
            bot_pos[1] - int(30 * math.sin(bot_rad))
        )
        pygame.draw.circle(self.screen, self.BLUE, bot_pos, 5)
        pygame.draw.line(self.screen, self.BLUE, bot_pos, bot_end, 2)
        
        # Draw line between target and bot
        pygame.draw.line(self.screen, self.GREEN, target_pos, bot_pos, 1)
        
        # Draw text info and input boxes
        info_text = [
            (f"Target: ({self.target_x:.1f}, {self.target_y:.1f})", ['target_x', 'target_y']),
            (f"Target Heading: {self.target_heading:.1f}°", ['target_heading']),
            (f"Bot: ({bot_x:.1f}, {bot_y:.1f})", []),
            (f"Bot Heading: {self.bot_heading:.1f}°", ['bot_heading']),
            (f"Distance: {self.distance:.1f} inches", ['distance'])
        ]
        
        for i, (text, input_keys) in enumerate(info_text):
            text_surface = self.font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (10, 10 + i*25))
            
            for key in input_keys:
                box = self.input_boxes[key]
                color = self.WHITE if self.active_input == key else self.GRAY
                pygame.draw.rect(self.screen, color, box, 2)
                
                if self.active_input == key:
                    input_surface = self.font.render(self.input_text, True, self.WHITE)
                    self.screen.blit(input_surface, (box.x + 5, box.y + 5))
            
        pygame.display.flip()
        
    def run_visualization(self):
        """Run interactive visualization loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.handle_mouse_input(event)
                    
            self.visualize()
            pygame.time.wait(16)  # Cap at ~60 FPS
            
        pygame.quit()

# Example usage:
if __name__ == "__main__":
    calculator = PoseCalculator("image.png")  # Replace with actual field image path
    calculator.run_visualization()
