from torchvision import transforms
from pytorch.model import mobilnetv2
from pytorch import nn
from PIL import image
def mobilnetv2(img_path="img1.jpg"):
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.2229,0.224, 0.225])

    ])


def result(img):
    image_tensor = img
    return image_tensor