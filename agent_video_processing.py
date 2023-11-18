import cv2
import numpy as np
import time
import threading
import queue

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, alpha=1.5, beta=20):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# Function for automatic brightness and contrast optimization using CLAHE
def auto_adjust_brightness_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return adjusted

# Function to reduce noise
def reduce_noise(image):
    denoised = cv2.medianBlur(image, 3)
    return denoised

# Function to sharpen the image
def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def is_frame_different(frame1, frame2, threshold):
    # Convert frames to grayscale for comparison
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)

    # Check if the average difference across the frame is above the threshold
    return np.mean(diff) > threshold

# Capturing video from webcam
vid = cv2.VideoCapture(0)

frame_interval = 1 / 30 # For 30 fps
stop_threads = False
frame_queue = queue.Queue()

# Define the video capture thread
def video_capture_thread():
    global stop_threads
    vid = cv2.VideoCapture(0)

    while True:
        success, frame = vid.read()
        if not success:
            break

        # Auto adjusting brightness and contrast
        frame = auto_adjust_brightness_contrast(frame)

        frame_queue.put(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break

    vid.release()
    cv2.destroyAllWindows()

# Define the frame processing thread
def frame_processing_thread():
    global stop_threads
    last_time = time.time()
    last_processed_frame = None

    while not stop_threads:
        if not frame_queue.empty() and time.time() - last_time >= frame_interval:
            frame = frame_queue.get()

            # Auto adjusting brightness and contrast
            frame = auto_adjust_brightness_contrast(frame)

            # Sharpening the image
            frame = sharpen_image(frame)

            if last_processed_frame is None or is_frame_different(last_processed_frame, frame, 30):
                print("Frame changed")

            last_processed_frame = frame.copy()

    last_time = time.time()

capture_thread = threading.Thread(target=video_capture_thread)
processing_thread = threading.Thread(target=frame_processing_thread)

capture_thread.start()
processing_thread.start()

# Wait for threads to finish
capture_thread.join()
processing_thread.join()