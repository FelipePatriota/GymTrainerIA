import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
from playsound import playsound

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def checkHandInBox(handX, handY, tlX, tlY, brX, brY):
    if handX > tlX and handX < brX and handY > tlY and handY < brY:
        return True
    else:
        return False

def createMenu():
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)

    menu_size = 120  # Tamanho do menu reduzido
    menu_x, menu_y = 20, 480 - menu_size - 20  # Coordenadas do canto inferior esquerdo reduzidas

    deadliftTlx, deadliftTly = menu_x, menu_y - menu_size - 10
    deadliftBrx, deadliftBry = menu_x + menu_size, menu_y - 10
    squatTlx, squatTly = menu_x, menu_y - 2 * menu_size - 20
    squatBrx, squatBry = menu_x + menu_size, menu_y - menu_size - 20

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates

                leftHand = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
                rightHand = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]

                if(checkHandInBox(leftHand[0]*640,leftHand[1]*480 ,deadliftTlx, deadliftTly, deadliftBrx, deadliftBry) or checkHandInBox(rightHand[0]*640,rightHand[1]*480 ,deadliftTlx, deadliftTly, deadliftBrx, deadliftBry)):
                    cap.release()
                    cv2.destroyAllWindows()
                    deadlift()
                    break

                if(checkHandInBox(leftHand[0]*640,leftHand[1]*480 ,squatTlx, squatTly, squatBrx, squatBry) or checkHandInBox(rightHand[0]*640,rightHand[1]*480 ,squatTlx, squatTly, squatBrx, squatBry)):
                    cap.release()
                    cv2.destroyAllWindows()
                    squat()
                    break
                        
            except:
                pass

            # Draw menu rectangles
            cv2.rectangle(image, (deadliftTlx, deadliftTly), (deadliftBrx, deadliftBry), (245, 117, 16), -1)
            text_width, text_height = cv2.getTextSize('DEADLIFT', cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
            text_x = int((menu_size - text_width) / 2) + menu_x
            text_y = int((menu_size + text_height) / 2) + (menu_y - menu_size - 30)
            cv2.putText(image, 'DEADLIFT', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.rectangle(image, (squatTlx, squatTly), (squatBrx, squatBry), (245, 117, 16), -1)
            text_width, text_height = cv2.getTextSize('SQUAT', cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
            text_x = int((menu_size - text_width) / 2) + menu_x
            text_y = int((menu_size + text_height) / 2) + (menu_y - 2 * menu_size - 70)
            cv2.putText(image, 'SQUAT', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2) 
                                     )  
            cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def checkDeadlift(a,b,c,d,checkUp=False):
    if c.visibility>0.5 and d.visibility>0.5:
        print("visible")
        if checkUp==True:
            if a.y<c.y and b.y<d.y:
                return True
            else:
                return False
        else:
            if a.y>c.y and b.y>d.y:
                return True
            else:
                return False
    else:
        print("not visible")
        return False


def deadlift():
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)

    # Deadlift counter variables
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # left_hand = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
                # right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                # left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                # right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                leftHand = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
                rightHand = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]

                if(checkHandInBox(leftHand[0]*640,leftHand[1]*480 ,590,20,670,80) or checkHandInBox(rightHand[0]*640,rightHand[1]*480,600,0,680,60)):
                    cap.release()
                    cv2.destroyAllWindows()
                    createMenu()
                    break

                left_hand = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                if checkDeadlift(left_hand,right_hand,left_knee,right_knee):
                    stage = "down"
                if stage=="down" and checkDeadlift(left_hand,right_hand,left_knee,right_knee,checkUp=True):
                    stage="up"
                    counter+=1
                    mytext=str(counter)
                    language='en'
                    myobj=gTTS(text=mytext,lang=language,slow=True)
                    myobj.save("welcome"+str(counter)+".mp3")
                    playsound("welcome"+str(counter)+".mp3")
                    print(counter)
                        
            except:
                pass
            
            # Setup status box
            cv2.rectangle(image, (0,0), (240,73), (245,117,16), -1)
            cv2.rectangle(image, (590,20), (670,80), (245,117,16), -1)
            cv2.putText(image, 'S', (590,79), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            # cv2.putText(image, 'DEADLIFT', 
            #             (460,60), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (85,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def checkSquat(a,b,c,d,checkUp=False):
    if a.visibilty>0.4 and b.visibility>0.4 and c.visibility>0.4 and d.visibility>0.4:
        print("visible")
        # Cup20=c.y+c.y*0.2
        Cdown30=c.y-c.y*0.6
        # Dup20=d.y+d.y*0.2
        Ddown30=d.y-d.y*0.6
        if checkUp==True:
            if a.y<Cdown30 and b.y<Ddown30:
                return True
            else:
                return False
        else:
            if a.y>Cdown30 and b.y>Ddown30:
                return True
            else:
                return False
    else:
        print("not visible")
        return False


def squat():
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(0)

    # Deadlift counter variables
    counter = 0 
    stage = "None"

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                leftHand = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
                rightHand = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]

                if(checkHandInBox(leftHand[0]*640,leftHand[1]*480 ,590,20,670,80) or checkHandInBox(rightHand[0]*640,rightHand[1]*480,600,0,680,60)):
                    cap.release()
                    cv2.destroyAllWindows()
                    createMenu()
                    break

                if checkSquat(left_hip,right_hip,left_knee,right_knee):
                    stage = "down"
                if stage=="down" and checkSquat(left_hip,right_hip,left_knee,right_knee,checkUp=True):
                    stage="up"
                    counter+=1
                    mytext=str(counter)
                    language='en'
                    myobj=gTTS(text=mytext,lang=language,slow=True)
                    myobj.save("welcome"+str(counter)+".mp3")
                    playsound("welcome"+str(counter)+".mp3")
                    print(counter)
                        
            except:
                pass
            
            # Setup status box
            cv2.rectangle(image, (0,0), (240,73), (245,117,16), -1)
            cv2.rectangle(image, (590,20), (670,80), (245,117,16), -1)
            cv2.putText(image, 'S', (590,79), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (85,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (80,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            # cv2.putText(image, 'SQUAT', 
            #             (470,60), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

createMenu()

