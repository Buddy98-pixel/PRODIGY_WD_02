import cv2
import mediapipe as mp

# 1. Setup Mediapipe Hand Engine
mp_hands = mp.solutions.hands
# We use a lower model_complexity to make it run faster on your laptop
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=1, 
                       model_complexity=0, 
                       min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 2. Open Camera
cap = cv2.VideoCapture(0)

print("🚀 AI Gesture Engine Started! Look for the 'Gesture Recognition' window.")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Flip image for natural 'mirror' feel
    img = cv2.flip(img, 1)
    
    # Process the hand landmarks
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Draw skeleton
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # --- INTERNSHIP LOGIC: Pointing Detection ---
            # Landmark 8 is the index finger tip. Landmark 6 is the joint.
            # If tip is higher (lower Y) than joint, you are pointing!
            if hand_lms.landmark[8].y < hand_lms.landmark[6].y:
                cv2.putText(img, "POINTING UP", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "FIST / IDLE", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 3. Show the output
    cv2.imshow("Gesture Recognition", img)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp

# Use this specific way to access the solutions
try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw
except ImportError:
    # Fallback for older versions
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

# Initialize
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# ... rest of your camera code ...