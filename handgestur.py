import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
 
# Video capture from the webcam
cap = cv2.VideoCapture(0)

# Set screen width and height for cursor control
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe Hand Detection
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        # Read frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for natural movement
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB, as required by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hand landmarks
        result = hands.process(rgb_frame)

        # If hands are detected, process them
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract the x, y coordinates of the index finger tip (landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                x = int(index_finger_tip.x * screen_width)
                y = int(index_finger_tip.y * screen_height)

                # Move the mouse to the index finger tip position
                pyautogui.moveTo(x, y)

                # Example for clicking when thumb tip (landmark 4) is near the index finger tip (landmark 8)
                thumb_tip = hand_landmarks.landmark[4]
                thumb_x = int(thumb_tip.x * screen_width)
                thumb_y = int(thumb_tip.y * screen_height)

                # Calculate the distance between thumb and index finger tip
                distance = ((thumb_x - x) ** 2 + (thumb_y - y) ** 2) ** 0.5

                # If the distance is below a threshold, perform a click action
                if distance < 50:
                    pyautogui.click()

        # Display the frame
        cv2.imshow("Hand Gesture Cursor Control", frame)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
