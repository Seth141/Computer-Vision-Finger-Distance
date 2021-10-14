import cv2
import cvzone


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = cvzone.HandDetector(detectionCon=0.5, maxHands=1)
detector2 = cvzone.FaceMeshDetector(maxFaces=1)


while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    
    #Face det:
    img, faces = detector2.findFaceMesh(img)
    if faces:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(faces[0])

   
    if lmList:
        # Find how many fingers are up
        fingers = detector.fingersUp()
        totalFingers = fingers.count(1)
        cv2.putText(img, f'Total fingers: {totalFingers}', (bbox[0] + 350, bbox[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    if lmList:
        # Find distance between a given two fingers
        distance, img, info = detector.findDistance(4, 8, img)
        cv2.putText(img, f'1-2 in mm:{int(distance)}', (bbox[0] + 350, bbox[1] + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    if lmList:
        # Find dist
        distance, img, info = detector.findDistance(8, 12, img)
        cv2.putText(img, f'2-3 in mm:{int(distance)}', (bbox[0] + 350, bbox[1] + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    if lmList:
        # Find dist
        distance, img, info = detector.findDistance(12, 16, img)
        cv2.putText(img, f'3-4 in mm:{int(distance)}', (bbox[0] + 350, bbox[1] + 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    if lmList:
        # Find dist
        distance, img, info = detector.findDistance(16, 20, img)
        cv2.putText(img, f'4-5 in mm:{int(distance)}', (bbox[0] + 350, bbox[1] + 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
 
    # Display output here: 
    cv2.imshow("Image", img)
    cv2.waitKey(1)








