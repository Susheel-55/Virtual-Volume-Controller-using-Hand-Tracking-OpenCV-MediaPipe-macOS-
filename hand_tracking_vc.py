import cv2
import time
import numpy as np
import HandTracking_Module as htm
import math
import subprocess
import threading


wCam, hCam = 960, 540
smoothing = 0.15
volumeStep = 2          
updateDelay = 0.05      

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.6)

pTime = 0
smoothVol = 0
prevVol = -1
lastUpdateTime = 0


def set_volume_mac(vol):
    subprocess.Popen(
        ["osascript", "-e", f"set volume output volume {vol}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def async_volume(vol):
    thread = threading.Thread(target=set_volume_mac, args=(vol,))
    thread.daemon = True
    thread.start()


while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.line(img, (x1, y1),(x2,y2), (255,0,0), 3)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        if length<50:
            cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)
        if length>250:
            cv2.line(img, (x1, y1),(x2,y2), (0,255,0), 3)


        

        vol = np.interp(length, [40, 250], [0, 100])

        smoothVol += smoothing * (vol - smoothVol)
        finalVol = int(smoothVol // volumeStep * volumeStep)

        currentTime = time.time()

        if finalVol != prevVol and (currentTime - lastUpdateTime) > updateDelay:
            async_volume(finalVol)
            prevVol = finalVol
            lastUpdateTime = currentTime



        sliderX = 100
        sliderTop = 100
        sliderBottom = 400

        cv2.rectangle(img,
                      (sliderX - 20, sliderTop - 20),
                      (sliderX + 20, sliderBottom + 20),
                      (30, 30, 30), -1)


        cv2.line(img,
                 (sliderX, sliderTop), 
                 (sliderX, sliderBottom),
                 (80, 80, 80), 6)


        knobY = int(np.interp(finalVol, [0, 100],
                              [sliderBottom, sliderTop]))


        cv2.circle(img, (sliderX, knobY), 20, (255, 100, 200), -1)
        cv2.circle(img, (sliderX, knobY), 10, (255, 255, 255), -1)


        cv2.putText(img, f"{finalVol}%",
                    (sliderX - 40, sliderBottom + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Volume Control - Ultra Smooth", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()