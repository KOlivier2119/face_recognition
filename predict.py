# predict.py
import cv2
import mediapipe as mp
import json
import os

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(1)

MODEL_DIR = "models"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(MODEL_DIR, "lbph.yml"))
with open(os.path.join(MODEL_DIR, "label_map.json"), "r") as f:
    label_map = json.load(f)
inv_map = {v:k for k,v in label_map.items()}

# Face mesh connections for drawing
face_mesh_connections = mp.solutions.face_mesh.FACEMESH_CONTOURS

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector, \
     mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.flip(frame, 1)
        h, w = img.shape[:2]
        
        # Process for face detection (for recognition)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detection_results = detector.process(rgb_img)
        
        # Process for face mesh (for visualization)
        mesh_results = face_mesh.process(rgb_img)
        
        # Draw face mesh landmarks
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # Draw face mesh tessellation (full mesh structure)
                mp_drawing.draw_landmarks(
                    img,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    None,
                    mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # Draw face contours
                mp_drawing.draw_landmarks(
                    img,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    mp_drawing_styles.get_default_face_mesh_contours_style())
                
                # Highlight eye regions with different color (cyan)
                # Left eye iris and landmarks
                left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                # Right eye iris and landmarks  
                right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                
                # Draw left eye landmarks with cyan
                for idx in left_eye_indices:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(img, (x, y), 3, (255, 255, 0), -1)
                
                # Draw right eye landmarks with cyan
                for idx in right_eye_indices:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(img, (x, y), 3, (255, 255, 0), -1)
        
        # Face recognition using detection results
        if detection_results.detections:
            for det in detection_results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w); y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w); y2 = y1 + int(bbox.height * h)
                pad = 10
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                face = img[y1:y2, x1:x2]
                if face.size == 0: continue
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (200,200))
                label_id, conf = recognizer.predict(face_resized)
                name = inv_map.get(label_id, "unknown")
                txt = f"{name} ({conf:.1f})"
                # Draw bounding box in green
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        cv2.imshow("Predict", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Press any key in the OpenCV window to exit")
cv2.waitKey(0)
cv2.destroyAllWindows()