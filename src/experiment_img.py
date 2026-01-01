import cv2 #Pour HC
import time
import torch
from torchvision import transforms
from model import EmotionCNN
from config import MODEL_DIR, DATA_DIR
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


model = EmotionCNN(input_shape=1, hidden_units=128, output_shape=6).to(device)
model.load_state_dict(torch.load(os.path.join(f"{MODEL_DIR}model_final_lr0.1_bs32.pth"), map_location=device))
model.eval()  

# Transformation image pour compatibilité avec le CNN (a mettre en module)
transform = transforms.Compose([
    transforms.ToPILImage(),          # convertit NumPy → PIL
    transforms.Resize((48, 48)),      # FER2013 attend 48x48
    transforms.Grayscale(1),          # 1 canal
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

#Class Names
class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ================================
# Initialisation Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================================
# Load image
img_path = "data/sample/happy.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image non trouvée: {img_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)


# -------------------------------
# Détection des visages
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=6,
    minSize=(60, 60),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# -------------------------------
# Prédiction pour chaque visage détecté
for (x, y, w, h) in faces:
    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, gray.shape[1])
    y2 = min(cy + size // 2, gray.shape[0])

    face_roi = gray[y1:y2, x1:x2]
    if face_roi.size == 0:
        continue

    face_tensor = transform(face_roi).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(face_tensor)
        pred = torch.argmax(outputs, dim=1)
        emotion = class_names[int(pred.item())]

    # Affichage rectangle et label
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(img, emotion, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# -------------------------------
# Affichage résultat
cv2.imshow("Emotion Detection", img)
cv2.imwrite(f"{DATA_DIR}sample/sample1.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()