import cv2 
import mediapipe as mp
import numpy as np 
mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle 

if __name__ == '__main__':
    # captures video from device number indicates which device 
    cap = cv2.VideoCapture(0)
    
    count = 0
    up = False
    # mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:   
        while cap.isOpened():
            # read returns the return variables and the image which we store as frame 
            ret, frame = cap.read()
            
            # Detect and render 
            # change colour to mediapipe format of rgb
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection and store it in result 
            results = pose.process(image)
            
            image.flags.writeable = True 
            # recolour back to bgr for opencv
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # extract landmarks 
            try: 
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x , landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x , landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x , landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                bicep_angle = calculate_angle(shoulder, elbow, wrist)
                if (bicep_angle < 45 and up):
                    count += 1
                    up = False
                elif (bicep_angle > 170 and not up): 
                    up = True 
                cv2.rectangle(image, (0,0), (175, 50) , (255, 64, 20), -1)
                cv2.putText(image, str(bicep_angle), tuple(np.multiply(elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(count), (30, 30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                if (up):
                    cv2.putText(image, str("Up Stage"), (60, 30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, str("Down Stage"), (60, 30) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass
            
            # draw detections we pass in the image, landmarks and connections between landmarks 
            
            
            # creates a popup that visualises the image, name the pop up and what we pass in
            cv2.imshow('Mediapipe Feed', image)
            # if q is pressed break feed 
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
