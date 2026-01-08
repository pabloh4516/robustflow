"""
Model Loading and Preprocessing Module
=======================================

Este módulo gerencia o carregamento de modelos pré-treinados
e as transformações necessárias para pré-processamento de imagens.

MODELOS DISPONÍVEIS:
- ResNet-50: Arquitetura residual profunda (50 camadas)
- InceptionV3: Arquitetura com módulos inception paralelos
- VGG16: Arquitetura sequencial clássica
- MobileNetV2: Arquitetura leve para dispositivos móveis

Todos os modelos são pré-treinados no ImageNet (1000 classes).
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import (
    ResNet50_Weights,
    Inception_V3_Weights,
    VGG16_Weights,
    MobileNet_V2_Weights
)
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import json
import os

# Labels do ImageNet (carregados sob demanda)
IMAGENET_LABELS = None


def load_imagenet_labels() -> list:
    """
    Carrega os labels do ImageNet.

    Returns:
        Lista com 1000 nomes de classes do ImageNet
    """
    global IMAGENET_LABELS
    if IMAGENET_LABELS is not None:
        return IMAGENET_LABELS

    # Labels simplificados das 1000 classes do ImageNet
    # Em produção, carregar de um arquivo JSON completo
    try:
        labels_path = os.path.join(os.path.dirname(__file__), "imagenet_labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                IMAGENET_LABELS = json.load(f)
        else:
            # Fallback: usar labels dos weights do torchvision
            weights = ResNet50_Weights.IMAGENET1K_V1
            IMAGENET_LABELS = weights.meta["categories"]
    except Exception:
        # Último fallback: nomes genéricos
        IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]

    return IMAGENET_LABELS


class ModelManager:
    """
    Gerenciador de modelos para o RobustnessLab.

    Responsável por:
    - Carregar modelos pré-treinados do Torchvision
    - Configurar transformações de pré-processamento
    - Gerenciar cache de modelos para evitar recarregamento
    """

    SUPPORTED_MODELS = {
        "resnet50": {
            "name": "ResNet-50",
            "input_size": 224,
            "description": "Rede residual com 50 camadas. Boa precisão e velocidade.",
            "weights_class": ResNet50_Weights,
        },
        "inception_v3": {
            "name": "Inception V3",
            "input_size": 299,  # InceptionV3 requer 299x299
            "description": "Arquitetura com módulos inception. Alta precisão.",
            "weights_class": Inception_V3_Weights,
        },
        "vgg16": {
            "name": "VGG-16",
            "input_size": 224,
            "description": "Arquitetura sequencial clássica. Mais vulnerável a ataques.",
            "weights_class": VGG16_Weights,
        },
        "mobilenet_v2": {
            "name": "MobileNet V2",
            "input_size": 224,
            "description": "Arquitetura leve. Menor robustez adversária.",
            "weights_class": MobileNet_V2_Weights,
        }
    }

    def __init__(self, device: str = "cpu"):
        """
        Inicializa o gerenciador de modelos.

        Args:
            device: Dispositivo de computação ('cpu' ou 'cuda')
        """
        self.device = device
        self._model_cache = {}
        self._transform_cache = {}

    def get_available_models(self) -> list:
        """
        Retorna lista de modelos disponíveis.

        Returns:
            Lista de dicts com informações de cada modelo
        """
        return [
            {
                "id": model_id,
                "name": info["name"],
                "input_size": info["input_size"],
                "description": info["description"]
            }
            for model_id, info in self.SUPPORTED_MODELS.items()
        ]

    def load_model(self, model_id: str) -> nn.Module:
        """
        Carrega um modelo pré-treinado.

        NOTA: Modelos são cacheados para evitar downloads repetidos.

        Args:
            model_id: Identificador do modelo ('resnet50', 'inception_v3', etc.)

        Returns:
            Modelo PyTorch em modo de avaliação

        Raises:
            ValueError: Se model_id não for suportado
        """
        if model_id not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Modelo '{model_id}' não suportado. "
                f"Opções: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # Verifica cache
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        # Carrega modelo com pesos pré-treinados
        weights_class = self.SUPPORTED_MODELS[model_id]["weights_class"]
        weights = weights_class.IMAGENET1K_V1

        if model_id == "resnet50":
            model = models.resnet50(weights=weights)
        elif model_id == "inception_v3":
            model = models.inception_v3(weights=weights)
            model.aux_logits = False  # Desativa logits auxiliares
        elif model_id == "vgg16":
            model = models.vgg16(weights=weights)
        elif model_id == "mobilenet_v2":
            model = models.mobilenet_v2(weights=weights)

        # Configura para avaliação e move para dispositivo
        model.eval()
        model.to(self.device)

        # Cacheia para uso futuro
        self._model_cache[model_id] = model

        return model

    def get_transform(self, model_id: str) -> transforms.Compose:
        """
        Retorna as transformações de pré-processamento para um modelo.

        IMPORTANTE: Diferentes modelos podem requerer diferentes
        tamanhos de entrada e normalizações.

        Args:
            model_id: Identificador do modelo

        Returns:
            Composição de transformações do torchvision
        """
        if model_id in self._transform_cache:
            return self._transform_cache[model_id]

        input_size = self.SUPPORTED_MODELS[model_id]["input_size"]

        # Normalização padrão do ImageNet
        # Estes valores são a média e desvio padrão dos pixels do ImageNet
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        transform = transforms.Compose([
            transforms.Resize(input_size + 32),  # Margem para crop
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # NÃO aplicamos normalize aqui pois o PGD opera em [0,1]
        ])

        self._transform_cache[model_id] = transform
        return transform

    def get_normalize_transform(self, model_id: str) -> transforms.Normalize:
        """
        Retorna apenas a transformação de normalização.

        Útil para aplicar normalização separadamente após o ataque PGD.
        """
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )


def preprocess_image(
    image: Image.Image,
    model_id: str,
    manager: ModelManager
) -> torch.Tensor:
    """
    Pré-processa uma imagem PIL para inferência.

    Args:
        image: Imagem PIL (RGB)
        model_id: ID do modelo alvo
        manager: Instância do ModelManager

    Returns:
        Tensor [1, 3, H, W] normalizado em [0, 1]
    """
    # Converte para RGB se necessário
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = manager.get_transform(model_id)
    tensor = transform(image)

    # Adiciona dimensão de batch
    return tensor.unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Converte um tensor PyTorch de volta para imagem PIL.

    Args:
        tensor: Tensor [1, 3, H, W] ou [3, H, W] em [0, 1]

    Returns:
        Imagem PIL RGB
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Garante que está em [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Converte para numpy [H, W, C]
    array = tensor.cpu().numpy().transpose(1, 2, 0)

    # Converte para uint8 [0, 255]
    array = (array * 255).astype(np.uint8)

    return Image.fromarray(array)


def create_normalized_model(
    base_model: nn.Module,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225]
) -> nn.Module:
    """
    Encapsula um modelo com normalização integrada.

    Isso permite que o PGD opere em imagens [0, 1] enquanto
    o modelo recebe entradas normalizadas corretamente.

    Args:
        base_model: Modelo PyTorch original
        mean: Média para normalização
        std: Desvio padrão para normalização

    Returns:
        Modelo wrapper com normalização automática
    """
    class NormalizedModel(nn.Module):
        def __init__(self, model, mean, std):
            super().__init__()
            self.model = model
            self.register_buffer(
                'mean',
                torch.tensor(mean).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'std',
                torch.tensor(std).view(1, 3, 1, 1)
            )

        def forward(self, x):
            # Normaliza entrada de [0,1] para distribuição ImageNet
            x_normalized = (x - self.mean) / self.std
            return self.model(x_normalized)

    return NormalizedModel(base_model, mean, std)
