import cv2

pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1,format=NV12 ! nvvidconv ! video/x-raw,width=640,height=480,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera failed to open.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received.")
        break

    cv2.imshow("Jetson IMX219 Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()