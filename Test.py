import mediapipe as mp
import cv2

# Inicializar utilitários de desenho e módulo Holistic
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Inicializar a captura de vídeo
cam = cv2.VideoCapture(0)

# Usar a classe Holistic dentro de um contexto with
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        # Converter a imagem para RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar a imagem
        results = holistic.process(image)
        
        # Converter a imagem de volta para BGR
        image2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Desenhar rosto
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image2, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
        
        # Desenhar corpo
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image2, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        # Desenhar mãos
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image2, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image2, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Exibir a imagem
        cv2.imshow('Webcam', image2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar a captura de vídeo e fechar janelas
cam.release()
cv2.destroyAllWindows()
