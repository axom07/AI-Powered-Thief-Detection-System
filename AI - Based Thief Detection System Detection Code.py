from ultralytics import YOLO
import cv2
import RPi.GPIO as GPIO
import time
from flask import Flask, Response
import threading

# === GPIO Setup ===
LED_PIN = 18
BUZZER_PIN = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# === YOLO Model Setup ===
model = YOLO("/home/raspi/Desktop/best.pt")

THREAT_CLASSES = ['person', 'knife', 'gun', 'crowbar', 'baseball-bat']
CONF_THRESHOLD = 0.60

# === Flask App Setup ===
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# === Webcam Setup ===
cap = cv2.VideoCapture(0)

# === Threat Flag ===
threat_active = False

# === LED Blinking Thread ===
def led_blink_worker():
    global threat_active
    while True:
        if threat_active:
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.3)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(0.3)
        else:
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(0.1)

# === Detection and Alert Logic ===
def detect_and_alert():
    global output_frame, lock, cap, threat_active

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Couldn't read frame from camera.")
            time.sleep(1)
            continue

        resized_frame = cv2.resize(frame, (640, 640))
        results = model.predict(resized_frame, conf=0.53, verbose=False)

        threat_detected = False
        detected_labels = []

        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label in THREAT_CLASSES and conf >= CONF_THRESHOLD:
                print(f"⚠️ THREAT DETECTED: {label} ({conf:.2f})")
                threat_detected = True
                detected_labels.append(label)

                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(resized_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(resized_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Handle LED and Buzzer functioning
        if threat_detected:
            threat_active = True
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
        else:
            threat_active = False
            GPIO.output(BUZZER_PIN, GPIO.LOW)

        with lock:
            output_frame = resized_frame.copy()

        # Display on Raspberry PI Remote Desktop
        cv2.imshow("Raspberry Pi Camera Feed", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

# === LIve FLask Server (Video STreaming) ===
def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask():
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True, use_reloader=False)

# === Main ===
if __name__ == "__main__":
    detection_thread = threading.Thread(target=detect_and_alert, daemon=True)
    led_thread = threading.Thread(target=led_blink_worker, daemon=True)

    detection_thread.start()
    led_thread.start()

    start_flask()