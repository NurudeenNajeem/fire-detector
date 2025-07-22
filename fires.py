from ultralytics import YOLO
import cvzone
import cv2
import math

# Running real time from webcam
cap = cv2.VideoCapture('fire2.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

model = YOLO('fire.pt')

# Reading the classes
classnames = ['fire']

print("Starting video processing...")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1
    # print(f"Processing frame {frame_count}, shape: {frame.shape}")

    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    detections_in_frame = 0
    # Getting bbox,confidence and class names informations to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence_percent = math.ceil(confidence * 100)
            Class = int(box.cls[0])

            # Try a very low confidence threshold initially to see if anything is detected
            if confidence_percent > 10: # <-- CHANGE THIS TO A LOWER VALUE (e.g., 10 or 5)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence_percent}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)
                detections_in_frame += 1
                print(f"  Detected: {classnames[Class]} with {confidence_percent}% confidence at [{x1},{y1},{x2},{y2}]")
            # else:
            #     print(f"  Skipped detection with {confidence_percent}% confidence (below threshold)")


    if detections_in_frame == 0:
        print(f"No fire detections in frame {frame_count}.")


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
print("Video processing finished.")