from ultralytics import YOLO
model = YOLO(r"d:\F47\GA_DSB\EXCERSIZES\Week9\bahrain_currency\best.pt")

import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame) # prediction
    annotated_frame = results[0].plot()  # draw boxes

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
