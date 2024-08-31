import numpy as np
import cv2 as cv

cap = cv.VideoCapture(1)

# Parameters for ShiTomasi corner detection
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Create some random colors for drawing tracks
color = np.random.randint(0, 255, (100, 3))

# Initialize variables for tracking
points_to_track = []
old_gray = None
mask = None
directions = []

# Initialize distance counters
distance_moved = {'Right': 0, 'Up': 0, 'Down': 0, 'Left': 0}

# Define minimum movement threshold
MIN_MOVEMENT_THRESHOLD = 5  # Minimum pixels for movement to be counted

def get_direction(dx, dy):
    angle = np.degrees(np.arctan2(dy, dx))
    if -45 <= angle <= 45:
        return 'Right'
    elif 45 < angle <= 135:
        return 'Up'
    elif -135 <= angle < -45:
        return 'Down'
    else:
        return 'Left'

def update_distance(direction, distance):
    if direction in distance_moved:
        distance_moved[direction] += distance

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    if old_gray is None:
        old_gray = frame_gray.copy()
        points_to_track = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)
    else:
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, points_to_track, None, **lk_params)
        good_new = p1[st == 1]
        good_old = points_to_track[st == 1]
        points_to_track = good_new.reshape(-1, 1, 2)
        
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            dx, dy = a - c, b - d
            
            # Skip small movements
            if np.sqrt(dx*2 + dy*2) < MIN_MOVEMENT_THRESHOLD:
                continue
            
            # Draw the movement
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            
            direction = get_direction(dx, dy)
            distance = np.sqrt(dx*2 + dy*2)
            update_distance(direction, distance)
            
            cv.putText(frame, direction, (int(a), int(b)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    
    img = cv.add(frame, mask)
    cv.imshow('Tracking Frame', img)
    
    # Create a blank image for distance display
    distance_img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv.putText(distance_img, f"Right: {distance_moved['Right']:.2f} px", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(distance_img, f"Up: {distance_moved['Up']:.2f} px", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(distance_img, f"Down: {distance_moved['Down']:.2f} px", (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(distance_img, f"Left: {distance_moved['Left']:.2f} px", (10, 200), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    
    cv.imshow('Distance Moved', distance_img)
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
    old_gray = frame_gray.copy()
    if len(points_to_track) < 10:
        points_to_track = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)

cv.destroyAllWindows()
cap.release()
