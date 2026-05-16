from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import uvicorn
import torch
import io


app = FastAPI()


class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GarbageClassifier(num_classes=6)
model.load_state_dict(torch.load('model_GarbageClassification.pth', map_location=device))
model.to(device)
model.eval()

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


@app.post('/predict')
async def predict(file: UploadFile = File()):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(400, detail='Файл не загружен')

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            y_prediction = model(image_tensor)
            prediction = y_prediction.argmax(dim=1).item()

        return {
            'Номер класса': prediction,
            'Название класса': class_names[prediction]
        }

    except Exception as e:
        raise HTTPException(500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
