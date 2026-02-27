import cv2
import time

def main():
    print("Testing OBS Virtual Camera on /dev/video4...")
    # Open the virtual camera
    cap = cv2.VideoCapture(4)
    
    if not cap.isOpened():
        print("Error: Could not open video device /dev/video4.")
        print("Please ensure OBS Virtual Camera is running.")
        return
        
    print("Successfully opened camera. Waiting 2 seconds for it to warm up...")
    time.sleep(2)
    
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from camera.")
    else:
        output_file = "obs_snapshot.png"
        cv2.imwrite(output_file, frame)
        print(f"Success! Captured frame with shape {frame.shape} and saved to {output_file}.")
        
    cap.release()

if __name__ == "__main__":
    main()
