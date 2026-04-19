import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# Landmarks for the tips of the fingers (Index, Middle, Ring, Pinky)
tip_ids = [8, 12, 16, 20]

print("Finger Counter Started! Press 'q' to quit.")

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1) # Flip for a mirror effect
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            fingers = []

            # 1. Check Thumb (Horizontal movement)
            # Compare tip (4) with the joint below it (3)
            if hand_lms.landmark[4].x < hand_lms.landmark[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # 2. Check 4 Fingers (Vertical movement)
            for tid in tip_ids:
                # If Tip Y is less than Knuckle Y (less is higher on screen)
                if hand_lms.landmark[tid].y < hand_lms.landmark[tid - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)
            
            # Display the count
            cv2.rectangle(img, (20, 300), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total_fingers), (45, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    cv2.imshow("Task 4: Finger Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
