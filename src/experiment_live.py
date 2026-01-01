import cv2 #Pour HC
import time
import torch
from torchvision import transforms
from model import EmotionCNN
from config import MODEL_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"


model = EmotionCNN(input_shape=1, hidden_units=128, output_shape=6).to(device)
model.load_state_dict(torch.load(f"{MODEL_DIR} / model_final_lr0.1_bs32.pth", map_location=device))
model.eval()  

# Transformation image pour compatibilité avec le CNN (a mettre en module)
transform = transforms.Compose([
    transforms.ToPILImage(),          # convertit NumPy → PIL
    transforms.Resize((48, 48)),      # FER2013 attend 48x48
    transforms.Grayscale(1),          # 1 canal
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# ================================
# Initialisation Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================================
# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

# ================================
# Variables de métriques
prev_time = time.time()
fps_list = []
latency_list = []

print("Appuyer sur 'Q' pour quitter")

while True:
    capture_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Conversion niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Amélioration du contraste (important pour Haar)
    gray = cv2.equalizeHist(gray)

    # ================================
    # Détection Haar optimisée
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,          # plus précis que 1.2
        minNeighbors=6,           # réduit les faux positifs
        minSize=(60, 60),         # ignore les petits visages bruités
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # ===== ROI carré centré =====
        size = max(w, h)
        cx, cy = x + w // 2, y + h // 2
        x1 = max(cx - size // 2, 0)
        y1 = max(cy - size // 2, 0)
        x2 = min(cx + size // 2, gray.shape[1])
        y2 = min(cy + size // 2, gray.shape[0])

        face_roi = gray[y1:y2, x1:x2]

        if face_roi.size == 0:
            continue

        # CNN preprocessing
        face_tensor = transform(face_roi).unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs = model(face_tensor)
            pred = torch.argmax(outputs, dim=1)

        class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        emotion = class_names[int(pred.item())]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, emotion, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ================================
    # Calcul métriques de fluidité
    curr_time = time.time()
    frame_time = curr_time - prev_time
    prev_time = curr_time

    fps = 1 / frame_time if frame_time > 0 else 0
    latency = (curr_time - capture_time) * 1000  # ms

    fps_list.append(fps)
    latency_list.append(latency)

    # ================================
    # Affichage HUD performance
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(frame, f"Latency: {latency:.1f} ms", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Haar Cascade + CNN (Optimized)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================================
# Statistiques finales
print("\n===== PERFORMANCE =====")
print(f"FPS moyen      : {sum(fps_list)/len(fps_list):.2f}")
print(f"Latence moyenne: {sum(latency_list)/len(latency_list):.2f} ms")
print(f"Jitter FPS     : {max(fps_list) - min(fps_list):.2f}")

cap.release()
cv2.destroyAllWindows()
