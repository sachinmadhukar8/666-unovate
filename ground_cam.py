import cv2
import numpy as np
from collections import deque
import time

def coordinates_to_loc(latitude, longitude):
    lat = int((latitude + 90) * 10**7)
    lon = int((longitude + 180) * 10**7)
    combined = lat * 10**7 + lon
    return str(combined).zfill(15)

def loc_to_coordinates(loc):
    combined = int(loc)
    lon = combined % 10**7
    lat = combined // 10**7
    return lat / 10*7 - 90, lon / 10*7 - 180

class MovingAverageFilter:
    def _init_(self, window_size):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
     
    def update(self, value):
        self.values.append(value)
        return np.mean(self.values, axis=0)

class RoverTracker:
    def _init_(self, scale_factor, initial_loc, path_width=640, path_height=360, window_size=10, grid_size=8):
        self.old_frame = None
        self.path_img = np.zeros((path_height, path_width, 3), dtype=np.uint8)
        self.total_distance_grid_units = 0
        self.grid_size = grid_size
        self.grid_scale = min(path_width / grid_size, path_height / grid_size)
        self.old_position = np.array([(self.grid_scale * (self.grid_size // 2)) + (self.grid_scale / 2), (self.grid_scale * (self.grid_size // 2)) + (self.grid_scale / 2)])
        self.filter = MovingAverageFilter(window_size)
        self.scale_factor = scale_factor
        self.grid_img = self.draw_grid(initial_loc)
        self.mean_flow = np.zeros(2)
        self.prev_time = time.time()
        self.speed = 0.0

    def draw_grid(self, initial_loc):
        grid_img = np.zeros((self.path_img.shape[0], self.path_img.shape[1], 3), dtype=np.uint8)
        text_img = np.zeros((self.path_img.shape[0], self.path_img.shape[1], 3), dtype=np.uint8)
        initial_lat, initial_lon = loc_to_coordinates(initial_loc)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                loc = coordinates_to_loc(initial_lat + (j - self.grid_size // 2) / 111111, initial_lon + (i - self.grid_size // 2) / 111320)
                cv2.rectangle(grid_img, (i * int(self.grid_scale), j * int(self.grid_scale)), ((i+1) * int(self.grid_scale), (j+1) * int(self.grid_scale)), (255, 255, 255), 1)
                
                text_size, _ = cv2.getTextSize(loc, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                text_x = (i * int(self.grid_scale) + int(self.grid_scale)//2) - (text_size[0] // 2)
                text_y = (j * int(self.grid_scale) + int(self.grid_scale)//2) + (text_size[1] // 2)
                cv2.putText(text_img, loc, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        grid_img = cv2.addWeighted(grid_img, 1, text_img, 0.5, 0)
        return grid_img

    def process_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.resize(frame_gray, (frame_gray.shape[1]//4, frame_gray.shape[0]//4))

        if self.old_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(self.old_frame, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            self.mean_flow = np.mean(flow, axis=(0, 1))
            self.mean_flow = self.filter.update(self.mean_flow)

            self.mean_flow *= -1
            self.mean_flow[1] /= self.scale_factor

            position = self.old_position + self.mean_flow
            cv2.line(self.path_img, tuple(self.old_position.astype(int)), tuple(position.astype(int)), (0, 255, 0), 2)

            distance = np.linalg.norm(position - self.old_position)
            self.total_distance_grid_units += distance / self.grid_scale

            self.old_position = position

            # Calculate speed in m/s
            current_time = time.time()
            elapsed_time = current_time - self.prev_time
            if elapsed_time > 0:
                self.speed = distance / elapsed_time

            self.prev_time = current_time

        self.old_frame = frame_gray

        current_pos_img = np.zeros_like(self.path_img)
        cv2.circle(current_pos_img, tuple(self.old_position.astype(int)), 10, (0, 0, 255), -1)

        combined_img = cv2.addWeighted(self.path_img, 1, self.grid_img, 1, 0)
        combined_img = cv2.addWeighted(combined_img, 1, current_pos_img, 1, 0)

        cv2.putText(combined_img, f"Total distance: {self.total_distance_grid_units:.2f} m ({self.total_distance_grid_units*100:.2f} cm)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(combined_img, f"Y Movement: {self.mean_flow[1]:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(combined_img, f"X Movement: {self.mean_flow[0]:.2f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(combined_img, f"Speed: {self.speed:.2f} m/s", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        return combined_img, self.total_distance_grid_units, self.speed

def on_trackbar(val, rover_tracker):
    rover_tracker.scale_factor = val / 1000

if _name_ == "_main_":
    cap = cv2.VideoCapture(1)
    initial_loc = input("Please enter the initial 12-digit code for the rover: ")

    scale_factor = 1000  # initialize with default scale factor

    rover_tracker = RoverTracker(scale_factor, initial_loc)

    # Set camera frame rate to 30 FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow('Rover Path', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Scale', 'Rover Path', 1000, 30000, lambda val: on_trackbar(val, rover_tracker))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        combined_img, total_distance_grid_units, speed = rover_tracker.process_frame(frame)

        cv2.imshow('Rover Path', combined_img)
        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
