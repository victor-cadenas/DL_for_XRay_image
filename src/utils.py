import torch
import streamlit as st
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from torchvision import transforms
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
import io

@st.cache_resource
def load_model(path, device):
    

    modelo = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = modelo.fc.in_features
    modelo.fc = nn.Linear(in_features, 2)  # nº de clases: 2
    modelo = modelo.to(device)

    ## Cargar checkpoint
    state_dict = torch.load(path, map_location=device)

    modelo.load_state_dict(state_dict)

    ## Poner en modo evaluación
    modelo.eval()
    return modelo

def preprocess_image(image):

    ## Misma normalización que las imágenes de entrenamiento
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Aplicar transformaciones
    image_tensor = transform(image)

    # Añadir dimensión batch -> (1,3,224,224)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def generate_prediction(model, img, device):

    model.eval()

    img = img.to(device)

    with torch.no_grad():

        logits = model(img)

        probs = F.softmax(logits, dim=1)

        pred_idx = probs.argmax(1).item()

        confidence = probs[0][pred_idx].item()*100

    clases = ["NORMAL", "NEUMONIA"]

    pred_class = clases[pred_idx]

    return pred_class, confidence

def generate_heatmap(model, img_tensor, device):

    model.eval()

    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_()

    # Crear extractor
    cam_extractor = SmoothGradCAMpp(
        model,
        target_layer='layer4'
    )

    # Forward
    logits = model(img_tensor)

    pred_class = logits.argmax(1).item()

    # Heatmap
    activation_map = cam_extractor(
        pred_class,
        logits
    )

    heatmap = activation_map[0].squeeze(0)

    # Redimensionar
    heatmap_tensor = heatmap.unsqueeze(0).unsqueeze(0)

    heatmap_resized = F.interpolate(
        heatmap_tensor,
        size=(224,224),
        mode='bilinear',
        align_corners=False
    )

    heatmap_resized = heatmap_resized.squeeze().cpu()

    # Imagen original
    img = img_tensor[0].cpu()

    img = (img - img.min()) / (
        img.max() - img.min()
    )

    img_pil = to_pil_image(img).convert("RGB")

    # Overlay final
    overlay = overlay_mask(
        img_pil,
        to_pil_image(heatmap_resized, mode='F'),
        alpha=0.5
    )

    # Mismo tamaño exacto
    overlay = overlay.resize(img_pil.size)

    cam_extractor.remove_hooks()

    return overlay