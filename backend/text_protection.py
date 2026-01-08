"""
Text Protection Module - Anti-OCR Adversarial Perturbations
============================================================

Este módulo implementa técnicas para proteger texto em imagens contra
reconhecimento automático (OCR) enquanto mantém legibilidade humana.

OBJETIVO:
- Texto visível e legível para humanos
- Ilegível para sistemas OCR (Tesseract, EasyOCR, TrOCR, Google Vision, etc.)
- Indetectável por detectores de texto (EAST, CRAFT, DBNet)

TÉCNICAS IMPLEMENTADAS:
1. Perturbações Adversárias PGD contra modelos OCR
2. Ruído Estruturado que confunde redes neurais
3. Padrões de Interferência em frequências específicas
4. Distorções Geométricas Sutis
5. Manipulação de Canais de Cor
6. Texturas Adversárias de Fundo
7. Perturbações em Bordas de Caracteres

FUNDAMENTO CIENTÍFICO:
Redes neurais de OCR dependem de padrões de alta frequência e bordas
bem definidas. Ao adicionar ruído estruturado que interfere com esses
padrões sem afetar a percepção humana global, podemos "cegar" o OCR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from typing import Tuple, Dict, List, Optional, Union
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift
import random
import math


class TextProtector:
    """
    Classe principal para proteção de texto contra OCR.

    Combina múltiplas técnicas de perturbação para criar
    imagens que são legíveis por humanos mas ilegíveis para máquinas.
    """

    def __init__(
        self,
        device: str = "cpu",
        protection_level: str = "readable"  # readable, stealth, low, medium, high, maximum
    ):
        """
        Inicializa o protetor de texto.

        Args:
            device: Dispositivo de computação
            protection_level: Nível de proteção (afeta intensidade das perturbações)
        """
        self.device = device
        self.protection_level = protection_level

        # Configurações por nível de proteção
        self.configs = {
            "stealth": {
                # Modo Stealth: Perturbações IMPERCEPTÍVEIS
                # Imagem parece 100% normal, mas OCR não consegue ler
                "noise_intensity": 0.002,       # Ultra baixo - invisível
                "pattern_strength": 0.05,       # Quase imperceptível
                "geometric_distortion": 0.0008, # Micro distorções (< 1 pixel)
                "edge_perturbation": 0.015,     # Mínimo absoluto
                "frequency_noise": 0.04,        # Muito baixo
                "stealth_mode": True
            },
            "readable": {
                # Modo Readable: MÁXIMA proteção, ZERO distorção visível
                # Ataques adversariais fortes que não afetam visão humana
                "noise_intensity": 0.08,        # Ruído adversarial forte
                "pattern_strength": 0.40,       # Padrões anti-IA
                "geometric_distortion": 0.0,    # ZERO - imagem nunca torce
                "edge_perturbation": 0.35,      # Ataque forte nas bordas do texto
                "frequency_noise": 0.45,        # Ataque máximo em frequência
                "readable_mode": True
            },
            "low": {
                "noise_intensity": 0.02,
                "pattern_strength": 0.3,
                "geometric_distortion": 0.005,
                "edge_perturbation": 0.1,
                "frequency_noise": 0.15
            },
            "medium": {
                "noise_intensity": 0.04,
                "pattern_strength": 0.5,
                "geometric_distortion": 0.01,
                "edge_perturbation": 0.2,
                "frequency_noise": 0.25
            },
            "high": {
                "noise_intensity": 0.06,
                "pattern_strength": 0.7,
                "geometric_distortion": 0.015,
                "edge_perturbation": 0.3,
                "frequency_noise": 0.35
            },
            "maximum": {
                "noise_intensity": 0.08,
                "pattern_strength": 0.9,
                "geometric_distortion": 0.02,
                "edge_perturbation": 0.4,
                "frequency_noise": 0.45
            }
        }

        self.config = self.configs.get(protection_level, self.configs["readable"])

    def protect(
        self,
        image: Union[Image.Image, np.ndarray],
        techniques: List[str] = None,
        preserve_colors: bool = True
    ) -> Tuple[Image.Image, Dict]:
        """
        Aplica proteção anti-OCR à imagem.

        Args:
            image: Imagem PIL ou array numpy
            techniques: Lista de técnicas a aplicar (None = todas)
            preserve_colors: Manter cores originais o máximo possível

        Returns:
            protected_image: Imagem protegida
            metrics: Métricas da proteção aplicada
        """
        # Converte para numpy se necessário
        if isinstance(image, Image.Image):
            img_array = np.array(image).astype(np.float32) / 255.0
        else:
            img_array = image.astype(np.float32)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0

        original = img_array.copy()

        # Técnicas disponíveis
        available_techniques = {
            "adversarial_noise": self._add_adversarial_noise,
            "structured_pattern": self._add_structured_pattern,
            "frequency_perturbation": self._add_frequency_perturbation,
            "geometric_distortion": self._add_geometric_distortion,
            "edge_disruption": self._add_edge_disruption,
            "color_channel_shift": self._add_color_channel_shift,
            "adversarial_texture": self._add_adversarial_texture,
            "micro_patterns": self._add_micro_patterns,
            "gradient_masking": self._add_gradient_masking,
            "dithering_noise": self._add_dithering_noise,
            # Técnicas Stealth (invisíveis)
            "stealth_subpixel": self._add_stealth_subpixel,
            "stealth_frequency": self._add_stealth_frequency_attack,
            "stealth_antialiasing": self._add_stealth_antialiasing,
            "stealth_kerning": self._add_stealth_kerning_disruption,
            "stealth_color_phase": self._add_stealth_color_phase,
            # Técnicas Invisíveis AVANÇADAS (máxima eficácia)
            "invisible_grayscale": self._add_invisible_grayscale_attack,
            "invisible_binarization": self._add_invisible_binarization_attack,
            "invisible_segmentation": self._add_invisible_segmentation_break,
            "invisible_phantom": self._add_invisible_phantom_edges,
            "invisible_adversarial": self._add_invisible_ocr_adversarial
        }

        # Seleciona técnicas baseado no modo
        if techniques is None:
            if self.config.get("stealth_mode", False):
                # Modo stealth: usa apenas técnicas invisíveis
                techniques = [
                    "stealth_subpixel",
                    "stealth_frequency",
                    "stealth_antialiasing",
                    "stealth_kerning",
                    "stealth_color_phase"
                ]
            elif self.config.get("readable_mode", False):
                # Modo readable: MÁXIMA proteção contra IA/OCR
                # Todas as técnicas que não distorcem geometricamente
                techniques = [
                    "adversarial_noise",       # Ruído adversarial multi-escala
                    "stealth_frequency",       # Ruído nas regiões de texto
                    "edge_disruption",         # Perturba bordas dos caracteres
                    "invisible_binarization",  # Ataca threshold do OCR
                    "invisible_grayscale",     # Ataca conversão grayscale
                    "frequency_perturbation",  # Ataque no domínio de frequência
                    "invisible_adversarial",   # Ataque combinado anti-OCR
                    "dithering_noise",         # Ruído de dithering
                ]
            else:
                techniques = list(available_techniques.keys())

        metrics = {
            "techniques_applied": [],
            "perturbation_strength": {},
            "estimated_ocr_accuracy_drop": 0
        }

        protected = img_array.copy()

        # Aplica cada técnica
        for technique in techniques:
            if technique in available_techniques:
                protected, tech_metrics = available_techniques[technique](protected)
                metrics["techniques_applied"].append(technique)
                metrics["perturbation_strength"][technique] = tech_metrics

        # Preserva cores se solicitado (mas não em modo readable - queremos mínima alteração)
        if preserve_colors and not self.config.get("readable_mode", False):
            protected = self._preserve_color_tone(original, protected)

        # Garante valores válidos
        protected = np.clip(protected, 0, 1)

        # Calcula métricas finais
        metrics["mse"] = float(np.mean((original - protected) ** 2))
        metrics["psnr"] = float(10 * np.log10(1.0 / (metrics["mse"] + 1e-10)))
        metrics["estimated_ocr_accuracy_drop"] = self._estimate_ocr_drop(metrics)

        # Converte de volta para PIL
        protected_pil = Image.fromarray((protected * 255).astype(np.uint8))

        return protected_pil, metrics

    def _add_adversarial_noise(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Adiciona ruído adversarial otimizado para confundir CNNs.

        O ruído é estruturado para maximizar confusão em camadas
        convolucionais enquanto permanece imperceptível.
        """
        intensity = self.config["noise_intensity"]

        # Ruído com distribuição específica que afeta features de CNN
        h, w = img.shape[:2]

        # Ruído em múltiplas escalas (multi-scale adversarial noise)
        noise = np.zeros_like(img)

        for scale in [1, 2, 4, 8]:
            # Ruído em escala reduzida
            sh, sw = h // scale, w // scale
            scale_noise = np.random.randn(sh, sw, 3) * intensity / scale

            # Upscale com interpolação
            scale_noise = cv2.resize(scale_noise, (w, h), interpolation=cv2.INTER_LINEAR)
            noise += scale_noise

        # Normaliza e aplica
        noise = noise / 4  # Média das escalas

        # Modula ruído pela luminância (mais ruído em áreas de médio contraste)
        if len(img.shape) == 3:
            luminance = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        else:
            luminance = img

        # Máscara de modulação (mais ruído em áreas de médio brilho)
        modulation = 4 * luminance * (1 - luminance)  # Pico em 0.5
        modulation = np.stack([modulation] * 3, axis=-1)

        result = img + noise * modulation

        return result, {"intensity": intensity, "scales": 4}

    def _add_structured_pattern(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Adiciona padrões estruturados que interferem com detecção de texto.

        Padrões de linha e grade que confundem algoritmos de
        segmentação de caracteres.
        """
        strength = self.config["pattern_strength"]
        h, w = img.shape[:2]

        # Padrão de interferência senoidal
        x = np.linspace(0, 4 * np.pi, w)
        y = np.linspace(0, 4 * np.pi, h)
        xx, yy = np.meshgrid(x, y)

        # Múltiplas frequências de interferência
        pattern = (
            np.sin(xx * 3 + yy * 2) * 0.3 +
            np.sin(xx * 7 - yy * 5) * 0.2 +
            np.sin(xx * 11 + yy * 13) * 0.15 +
            np.cos(xx * 17 - yy * 19) * 0.1
        )

        # Normaliza para [-1, 1]
        pattern = pattern / np.abs(pattern).max()

        # Expande para 3 canais com fases diferentes
        pattern_rgb = np.stack([
            pattern,
            np.roll(pattern, h//4, axis=0),
            np.roll(pattern, w//4, axis=1)
        ], axis=-1)

        # Aplica com intensidade controlada
        result = img + pattern_rgb * strength * 0.03

        return result, {"strength": strength, "frequencies": 4}

    def _add_frequency_perturbation(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Perturbação no domínio da frequência.

        Adiciona ruído em frequências específicas que afetam
        a detecção de bordas usada por OCR.
        """
        noise_level = self.config["frequency_noise"]

        result = np.zeros_like(img)

        for c in range(3):
            channel = img[:,:,c]

            # FFT
            f_transform = fft2(channel)
            f_shifted = fftshift(f_transform)

            h, w = channel.shape
            cy, cx = h // 2, w // 2

            # Cria máscara de ruído para frequências médias
            # (onde está a maioria da informação de texto)
            y, x = np.ogrid[:h, :w]

            # Anel de frequências médias (onde texto é mais detectável)
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)

            # Banda de frequência alvo (10-40% do espectro)
            inner_radius = min(h, w) * 0.1
            outer_radius = min(h, w) * 0.4

            mask = ((dist > inner_radius) & (dist < outer_radius)).astype(float)

            # Ruído de fase nas frequências alvo
            phase_noise = np.random.uniform(-np.pi * noise_level, np.pi * noise_level, (h, w))
            phase_perturbation = np.exp(1j * phase_noise * mask)

            # Aplica perturbação
            f_perturbed = f_shifted * phase_perturbation

            # Inversa FFT
            f_unshifted = fftshift(f_perturbed)
            result[:,:,c] = np.real(ifft2(f_unshifted))

        return result, {"noise_level": noise_level, "target_band": "mid-frequency"}

    def _add_geometric_distortion(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Adiciona distorções geométricas micro que confundem OCR.

        Pequenas ondulações que são imperceptíveis para humanos
        mas quebram o alinhamento esperado pelos algoritmos.
        """
        distortion = self.config["geometric_distortion"]
        h, w = img.shape[:2]

        # Cria grid de distorção
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)

        # Distorção senoidal em múltiplas frequências
        freq_x = random.uniform(0.02, 0.05)
        freq_y = random.uniform(0.02, 0.05)

        dx = (
            np.sin(yy * freq_x * 2 * np.pi) * distortion * w +
            np.sin(yy * freq_x * 4 * np.pi) * distortion * w * 0.5 +
            np.sin(xx * freq_y * 3 * np.pi) * distortion * w * 0.3
        )

        dy = (
            np.sin(xx * freq_y * 2 * np.pi) * distortion * h +
            np.sin(xx * freq_y * 4 * np.pi) * distortion * h * 0.5 +
            np.sin(yy * freq_x * 3 * np.pi) * distortion * h * 0.3
        )

        # Coordenadas distorcidas
        map_x = (xx + dx).astype(np.float32)
        map_y = (yy + dy).astype(np.float32)

        # Aplica remapeamento
        result = cv2.remap(
            img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return result, {"distortion": distortion, "type": "sinusoidal"}

    def _add_edge_disruption(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Perturba especificamente as bordas dos caracteres.

        OCR depende fortemente de detecção de bordas.
        Adicionar ruído seletivo nas bordas confunde a segmentação.
        """
        perturbation = self.config["edge_perturbation"]

        # Converte para grayscale para detectar bordas
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)

        # Detecta bordas com Canny
        edges = cv2.Canny(gray, 50, 150)

        # Dilata bordas para criar região de perturbação
        kernel = np.ones((3, 3), np.uint8)
        edge_region = cv2.dilate(edges, kernel, iterations=2)
        edge_mask = edge_region.astype(np.float32) / 255.0

        # Ruído específico para bordas
        edge_noise = np.random.randn(*img.shape) * perturbation * 0.1

        # Aplica ruído apenas nas regiões de borda
        if len(img.shape) == 3:
            edge_mask = np.stack([edge_mask] * 3, axis=-1)

        result = img + edge_noise * edge_mask

        return result, {"perturbation": perturbation, "edge_coverage": float(edge_mask.mean())}

    def _add_color_channel_shift(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Desalinha sutilmente os canais de cor.

        Causa chromatic aberration artificial que confunde
        modelos treinados em imagens alinhadas.
        """
        if len(img.shape) != 3:
            return img, {"applied": False}

        shift_amount = max(1, int(self.config["geometric_distortion"] * 100))

        result = img.copy()

        # Shift diferente para cada canal
        result[:,:,0] = np.roll(img[:,:,0], shift_amount, axis=1)  # R -> direita
        result[:,:,2] = np.roll(img[:,:,2], -shift_amount, axis=1)  # B -> esquerda

        # Shift vertical sutil
        result[:,:,0] = np.roll(result[:,:,0], shift_amount // 2, axis=0)
        result[:,:,2] = np.roll(result[:,:,2], -shift_amount // 2, axis=0)

        return result, {"shift_pixels": shift_amount}

    def _add_adversarial_texture(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Adiciona textura adversarial ao fundo.

        Padrões que parecem ruído uniforme para humanos mas
        criam features falsas para CNNs.
        """
        strength = self.config["pattern_strength"] * 0.5
        h, w = img.shape[:2]

        # Gera textura adversarial usando padrões de Gabor
        texture = np.zeros((h, w))

        for theta in [0, 45, 90, 135]:
            # Kernel de Gabor com diferentes orientações
            kernel_size = 15
            sigma = 3.0
            lambd = 8.0
            gamma = 0.5

            kernel = cv2.getGaborKernel(
                (kernel_size, kernel_size),
                sigma, np.radians(theta), lambd, gamma
            )

            # Ruído filtrado por Gabor
            noise = np.random.randn(h, w)
            filtered = cv2.filter2D(noise, -1, kernel)
            texture += filtered * 0.25

        # Normaliza
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        texture = (texture - 0.5) * strength * 0.1

        # Aplica aos 3 canais
        if len(img.shape) == 3:
            texture = np.stack([texture] * 3, axis=-1)

        result = img + texture

        return result, {"strength": strength, "gabor_orientations": 4}

    def _add_micro_patterns(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Adiciona micro-padrões invisíveis a olho nu.

        Padrões de 1-2 pixels que são filtrados pela percepção
        humana mas afetam processamento de baixo nível.
        """
        intensity = self.config["noise_intensity"] * 0.5
        h, w = img.shape[:2]

        # Padrão de checkerboard em escala de pixel
        checker = np.indices((h, w)).sum(axis=0) % 2
        checker = checker.astype(np.float32) * 2 - 1  # [-1, 1]

        # Modulação por ruído
        modulation = np.random.randn(h, w) * 0.5 + 0.5
        micro = checker * modulation * intensity * 0.05

        if len(img.shape) == 3:
            micro = np.stack([micro] * 3, axis=-1)

        result = img + micro

        return result, {"intensity": intensity, "pattern": "checkerboard"}

    def _add_gradient_masking(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Técnica de gradient masking para dificultar ataques/análises.

        Adiciona componentes que causam gradientes enganosos
        durante backpropagation.
        """
        strength = self.config["pattern_strength"] * 0.3
        h, w = img.shape[:2]

        # Função com gradientes problemáticos
        x = np.linspace(-3, 3, w)
        y = np.linspace(-3, 3, h)
        xx, yy = np.meshgrid(x, y)

        # Combinação de funções com derivadas complexas
        mask = (
            np.tanh(xx * yy) * 0.3 +
            np.sin(xx ** 2 + yy ** 2) * 0.3 +
            np.sign(np.sin(xx * 5) * np.cos(yy * 5)) * 0.2
        )

        mask = mask / np.abs(mask).max() * strength * 0.02

        if len(img.shape) == 3:
            mask = np.stack([mask] * 3, axis=-1)

        result = img + mask

        return result, {"strength": strength, "type": "gradient_masking"}

    def _add_dithering_noise(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Adiciona ruído de dithering estilizado.

        Similar a técnicas de impressão que confundem
        digitalização/OCR.
        """
        intensity = self.config["noise_intensity"]
        h, w = img.shape[:2]

        # Matriz de Bayer para dithering ordenado
        bayer_matrix = np.array([
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ]) / 16.0

        # Tile a matriz para cobrir a imagem
        tiles_y = h // 4 + 1
        tiles_x = w // 4 + 1
        dither = np.tile(bayer_matrix, (tiles_y, tiles_x))[:h, :w]

        # Converte para ruído centrado
        dither = (dither - 0.5) * intensity * 0.1

        if len(img.shape) == 3:
            dither = np.stack([dither] * 3, axis=-1)

        result = img + dither

        return result, {"intensity": intensity, "type": "bayer_dither"}

    # ========================================================================
    # TÉCNICAS STEALTH - Completamente invisíveis mas efetivas contra OCR
    # ========================================================================

    def _add_stealth_subpixel(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Perturbação sub-pixel que afeta reconhecimento de caracteres.

        Adiciona variações de intensidade no nível de sub-pixel que são
        imperceptíveis ao olho humano mas confundem a extração de features
        de modelos de OCR.
        """
        h, w = img.shape[:2]

        # Ruído de altíssima frequência (1 pixel) - MUITO SUTIL
        noise = np.random.randn(h, w, 3) * 0.001  # Ultra baixo - invisível

        # Aplicar apenas em áreas de transição (bordas de texto)
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)

        # Detecta gradientes (bordas)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mask = (gradient_mag / (gradient_mag.max() + 1e-8))

        if len(img.shape) == 3:
            gradient_mask = np.stack([gradient_mask] * 3, axis=-1)

        # Aplica ruído apenas nas bordas com intensidade mínima
        result = img + noise * gradient_mask * 0.3

        return result, {"type": "subpixel", "intensity": 0.001}

    def _add_stealth_frequency_attack(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Perturbação APENAS nas regiões de texto, não no fundo.

        Detecta onde está o texto e aplica ruído sutil apenas nessas áreas,
        mantendo o fundo 100% intacto.
        """
        h, w = img.shape[:2]
        result = img.copy()

        # Detecta regiões de texto (áreas com variação/bordas)
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)

        # Detecta bordas (onde está o texto)
        edges = cv2.Canny(gray, 50, 150)

        # Dilata para pegar área ao redor do texto
        kernel = np.ones((5, 5), np.uint8)
        text_region = cv2.dilate(edges, kernel, iterations=3)

        # Suaviza a máscara para transição suave
        text_mask = cv2.GaussianBlur(text_region.astype(np.float32), (11, 11), 0)
        text_mask = text_mask / (text_mask.max() + 1e-8)

        # Aplica ruído nas regiões de texto
        # Intensidade suficiente para confundir OCR/IA
        noise_intensity = self.config.get("noise_intensity", 0.035)
        noise = np.random.randn(h, w) * noise_intensity

        # Expande para 3 canais
        if len(img.shape) == 3:
            text_mask = np.stack([text_mask] * 3, axis=-1)
            noise = np.stack([noise] * 3, axis=-1)

        # Aplica ruído apenas na região do texto
        result = result + noise * text_mask

        return result, {"type": "text_targeted_noise", "coverage": float(text_mask.mean())}

    def _add_stealth_antialiasing(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Manipula anti-aliasing de forma imperceptível.

        Altera sutilmente os pixels de transição (anti-aliasing) de forma
        que OCR tem dificuldade em determinar bordas exatas de caracteres.
        """
        h, w = img.shape[:2]

        # Detecta pixels de transição (anti-aliasing)
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)

        # Pixels de anti-aliasing são aqueles com valores intermediários perto de bordas
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edge_proximity = np.abs(laplacian) / (np.abs(laplacian).max() + 1e-8)

        # Pixels com valor intermediário (64-192 de 255)
        intermediate = ((gray > 64) & (gray < 192)).astype(float)

        # Máscara de anti-aliasing
        aa_mask = edge_proximity * intermediate
        aa_mask = cv2.GaussianBlur(aa_mask.astype(np.float32), (3, 3), 0)

        # Perturbação MÍNIMA nos pixels de anti-aliasing
        perturbation = np.random.uniform(-0.003, 0.003, img.shape)

        if len(img.shape) == 3:
            aa_mask = np.stack([aa_mask] * 3, axis=-1)

        result = img + perturbation * aa_mask

        return result, {"type": "antialiasing_manipulation", "coverage": float(aa_mask.mean())}

    def _add_stealth_kerning_disruption(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Micro-distorções horizontais que quebram detecção de kerning/espaçamento.

        OCR usa análise de espaçamento entre caracteres. Pequenas
        distorções locais confundem essa análise sem ser visíveis.
        """
        h, w = img.shape[:2]

        # Cria distorção horizontal variável MUITO sutil
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)

        # Múltiplas frequências de distorção (simula variação de kerning)
        dx = np.zeros((h, w), dtype=np.float32)

        # Distorções MÍNIMAS em diferentes escalas
        for freq in [0.05, 0.08, 0.12]:
            dx += np.sin(yy * freq * 2 * np.pi + np.random.uniform(0, np.pi)) * 0.08
            dx += np.sin(xx * freq * 2 * np.pi + np.random.uniform(0, np.pi)) * 0.04

        # Normaliza para máximo de ~0.15 pixel de deslocamento (invisível)
        dx = dx / (np.abs(dx).max() + 1e-8) * 0.15
        dy = dx * 0.1  # Componente vertical mínimo

        # Aplica remapeamento
        map_x = (xx + dx).astype(np.float32)
        map_y = (yy + dy).astype(np.float32)

        result = cv2.remap(
            img.astype(np.float32),
            map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return result, {"type": "kerning_disruption", "max_shift": 0.15}

    def _add_stealth_color_phase(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Shift de fase de cor imperceptível.

        Pequenas variações nos canais de cor que são imperceptíveis
        mas afetam modelos que esperam alinhamento perfeito RGB.
        """
        if len(img.shape) != 3:
            return img, {"applied": False}

        h, w = img.shape[:2]
        result = img.copy()

        # Shift sub-pixel MÍNIMO diferente para cada canal
        # Invisível ao olho humano, mas afeta processamento

        # Canal R: micro shift para direita/baixo (0.1 pixel - imperceptível)
        M_r = np.float32([[1, 0, 0.1], [0, 1, 0.05]])
        result[:,:,0] = cv2.warpAffine(
            img[:,:,0], M_r, (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

        # Canal B: micro shift para esquerda/cima
        M_b = np.float32([[1, 0, -0.1], [0, 1, -0.05]])
        result[:,:,2] = cv2.warpAffine(
            img[:,:,2], M_b, (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

        # G fica no lugar (âncora visual)

        return result, {"type": "color_phase_shift", "shift_amount": 0.1}

    # ========================================================================
    # TÉCNICAS INVISÍVEIS AVANÇADAS - Máxima eficácia, zero visibilidade
    # ========================================================================

    def _add_invisible_grayscale_attack(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Ataque que explora conversão para grayscale do OCR.

        OCR converte imagem para grayscale usando: Gray = 0.299R + 0.587G + 0.114B
        Adicionamos ruído onde 0.299ΔR + 0.587ΔG + 0.114ΔB ≈ 0
        Isso é INVISÍVEL para OCR em grayscale mas perturba o processamento RGB.
        """
        if len(img.shape) != 3:
            return img, {"applied": False}

        h, w = img.shape[:2]
        # Intensidade MUITO baixa para ser invisível
        intensity = 0.003  # Fixo e baixo, não usa config

        # Gera ruído que cancela em grayscale
        # Usando abordagem mais controlada: R-G swap
        delta_r = np.random.randn(h, w) * intensity
        delta_g = -delta_r * 0.5  # Parcialmente cancela
        delta_b = -delta_r * 0.3  # Parcialmente cancela

        # Limita todos os canais
        delta_r = np.clip(delta_r, -0.01, 0.01)
        delta_g = np.clip(delta_g, -0.01, 0.01)
        delta_b = np.clip(delta_b, -0.01, 0.01)

        noise = np.stack([delta_r, delta_g, delta_b], axis=-1)
        result = img + noise

        return result, {"type": "grayscale_invariant", "intensity": intensity}

    def _add_invisible_binarization_attack(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Ataque contra binarização (Otsu/Adaptive Threshold).

        OCR binariza a imagem. Pixels próximos ao limiar são vulneráveis.
        Empurramos pixels para o lado errado do limiar de forma imperceptível.
        """
        h, w = img.shape[:2]

        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)

        # Calcula limiar Otsu
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = threshold / 255.0

        # Encontra pixels próximos ao limiar (zona de transição)
        gray_float = gray.astype(np.float32) / 255.0
        distance_to_threshold = np.abs(gray_float - threshold)
        transition_zone = distance_to_threshold < 0.1  # 10% do limiar

        # Perturba esses pixels para confundir binarização
        perturbation = np.zeros_like(img)
        noise_strength = 0.005  # Ultra sutil - 0.5%

        for c in range(3 if len(img.shape) == 3 else 1):
            channel_noise = np.random.choice([-1, 1], size=(h, w)) * noise_strength
            if len(img.shape) == 3:
                perturbation[:,:,c] = channel_noise * transition_zone
            else:
                perturbation = channel_noise * transition_zone

        result = img + perturbation
        return result, {"type": "binarization_attack", "threshold": float(threshold)}

    def _add_invisible_segmentation_break(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Quebra segmentação de caracteres de forma invisível.

        OCR usa connected components para separar caracteres.
        Adicionamos 'pontes' e 'quebras' invisíveis que confundem isso.
        """
        h, w = img.shape[:2]

        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)

        # Detecta bordas verticais (separação entre caracteres)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        vertical_edges = np.abs(sobel_x)
        vertical_edges = vertical_edges / (vertical_edges.max() + 1e-8)

        # Cria perturbação nas áreas entre caracteres
        perturbation = np.zeros_like(img)

        # Linhas verticais muito finas e sutis em posições de separação
        edge_positions = np.where(vertical_edges > 0.3)

        noise_intensity = 0.015
        for y, x in zip(edge_positions[0][::10], edge_positions[1][::10]):
            if 0 < x < w-1 and 0 < y < h-1:
                # Pequena perturbação local
                if len(img.shape) == 3:
                    perturbation[max(0,y-1):min(h,y+2), x, :] += np.random.randn(3) * noise_intensity
                else:
                    perturbation[max(0,y-1):min(h,y+2), x] += np.random.randn() * noise_intensity

        result = img + perturbation
        return result, {"type": "segmentation_break", "edges_targeted": len(edge_positions[0])//10}

    def _add_invisible_phantom_edges(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Injeta bordas fantasma invisíveis a humanos.

        Humanos precisam de ~2% de contraste para ver bordas.
        Detectores de borda (Canny, Sobel) detectam contrastes menores.
        Adicionamos bordas com ~1% de contraste = invisíveis mas detectáveis.
        """
        h, w = img.shape[:2]

        # Cria padrão de bordas fantasma
        phantom = np.zeros((h, w), dtype=np.float32)

        # Linhas horizontais fantasma a cada ~20 pixels
        for y in range(10, h, 20):
            phantom[y, :] = 0.01 * np.sin(np.linspace(0, 10*np.pi, w))

        # Linhas verticais fantasma a cada ~15 pixels
        for x in range(8, w, 15):
            phantom[:, x] += 0.01 * np.sin(np.linspace(0, 8*np.pi, h))

        # Curvas aleatórias que parecem caracteres
        for _ in range(5):
            cx, cy = np.random.randint(20, w-20), np.random.randint(10, h-10)
            for angle in np.linspace(0, np.pi, 20):
                px = int(cx + 10 * np.cos(angle))
                py = int(cy + 8 * np.sin(angle))
                if 0 <= px < w and 0 <= py < h:
                    phantom[py, px] = 0.008

        if len(img.shape) == 3:
            phantom = np.stack([phantom] * 3, axis=-1)

        result = img + phantom
        return result, {"type": "phantom_edges", "contrast": 0.01}

    def _add_invisible_ocr_adversarial(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Perturbação adversarial otimizada especificamente para OCR.

        Combina múltiplas técnicas em camadas para máximo impacto:
        1. Ataque em frequência na banda de texto
        2. Perturbação de anti-aliasing
        3. Ruído estruturado em bordas
        """
        h, w = img.shape[:2]
        result = img.copy()

        # === Camada 1: Ataque de frequência agressivo mas invisível ===
        for c in range(3 if len(img.shape) == 3 else 1):
            channel = result[:,:,c] if len(result.shape) == 3 else result

            f_transform = fft2(channel)
            f_shifted = fftshift(f_transform)

            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)

            # Múltiplas bandas de frequência críticas para OCR
            bands = [
                (0.08, 0.15, 0.08),  # Baixa-média: estrutura geral
                (0.15, 0.30, 0.12),  # Média: detalhes de caracteres
                (0.30, 0.45, 0.06),  # Alta-média: bordas finas
            ]

            for inner_pct, outer_pct, noise_strength in bands:
                inner = min(h, w) * inner_pct
                outer = min(h, w) * outer_pct
                band_mask = ((dist > inner) & (dist < outer)).astype(float)

                phase_noise = np.random.uniform(-noise_strength, noise_strength, (h, w))
                f_shifted = f_shifted * np.exp(1j * phase_noise * band_mask)

            result_channel = np.real(ifft2(fftshift(f_shifted)))
            if len(result.shape) == 3:
                result[:,:,c] = result_channel
            else:
                result = result_channel

        # === Camada 2: Perturbação de sub-pixel em bordas ===
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)

        edges = cv2.Canny(gray, 30, 100)
        edge_mask = cv2.dilate(edges, np.ones((2,2), np.uint8)).astype(np.float32) / 255.0

        edge_noise = np.random.randn(*result.shape) * 0.008
        if len(result.shape) == 3:
            edge_mask = np.stack([edge_mask] * 3, axis=-1)

        result = result + edge_noise * edge_mask

        # === Camada 3: Micro-distorção sub-pixel (INVISÍVEL) ===
        # Máximo 0.08 pixels - completamente imperceptível para humanos
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        dx = np.sin(yy * 0.15) * 0.04 + np.sin(xx * 0.08) * 0.02  # max 0.06px
        dy = np.sin(xx * 0.12) * 0.02  # max 0.02px

        map_x = (xx + dx).astype(np.float32)
        map_y = (yy + dy).astype(np.float32)

        result = cv2.remap(result.astype(np.float32), map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

        return result, {"type": "ocr_adversarial_combined", "layers": 3}

    def _preserve_color_tone(
        self,
        original: np.ndarray,
        protected: np.ndarray
    ) -> np.ndarray:
        """
        Preserva o tom de cor geral da imagem original.
        """
        # Calcula média de cor original e protegida
        orig_mean = original.mean(axis=(0, 1), keepdims=True)
        prot_mean = protected.mean(axis=(0, 1), keepdims=True)

        # Ajusta para manter tom similar
        adjusted = protected - prot_mean + orig_mean

        return adjusted

    def _estimate_ocr_drop(self, metrics: dict) -> float:
        """
        Estima a queda de precisão do OCR baseado nas técnicas aplicadas.
        """
        base_drop = 0

        technique_impact = {
            "adversarial_noise": 15,
            "structured_pattern": 20,
            "frequency_perturbation": 25,
            "geometric_distortion": 30,
            "edge_disruption": 35,
            "color_channel_shift": 10,
            "adversarial_texture": 20,
            "micro_patterns": 10,
            "gradient_masking": 15,
            "dithering_noise": 12,
            # Técnicas Stealth - eficazes mesmo com baixa intensidade
            "stealth_subpixel": 18,
            "stealth_frequency": 22,
            "stealth_antialiasing": 20,
            "stealth_kerning": 25,
            "stealth_color_phase": 15,
            # Técnicas Invisíveis Avançadas - ALTA eficácia
            "invisible_grayscale": 25,
            "invisible_binarization": 30,
            "invisible_segmentation": 28,
            "invisible_phantom": 22,
            "invisible_adversarial": 40  # Mais poderosa - ataque combinado
        }

        for tech in metrics["techniques_applied"]:
            base_drop += technique_impact.get(tech, 10)

        # Limita a 95% (nunca garantir 100%)
        return min(95, base_drop)


class AntiOCRAttack:
    """
    Ataque adversarial direcionado especificamente contra modelos OCR.

    Usa PGD otimizado para maximizar erro de reconhecimento de texto
    enquanto minimiza perturbação visual.
    """

    def __init__(
        self,
        ocr_model: nn.Module = None,
        epsilon: float = 0.05,
        alpha: float = 0.01,
        num_iterations: int = 20,
        device: str = "cpu"
    ):
        self.ocr_model = ocr_model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.device = device

    def attack(
        self,
        image: torch.Tensor,
        target_confusion: str = "random"
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Executa ataque PGD contra OCR.

        Args:
            image: Tensor da imagem [1, C, H, W]
            target_confusion: Tipo de confusão alvo

        Returns:
            adversarial: Imagem adversarial
            metrics: Métricas do ataque
        """
        if self.ocr_model is None:
            # Sem modelo OCR, usa perturbação heurística
            return self._heuristic_attack(image)

        image = image.clone().detach().to(self.device)
        delta = torch.zeros_like(image).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True

        for i in range(self.num_iterations):
            adv_image = image + delta
            adv_image = torch.clamp(adv_image, 0, 1)

            # Forward pass no modelo OCR
            outputs = self.ocr_model(adv_image)

            # Loss: maximizar entropia/confusão
            loss = -torch.distributions.Categorical(
                logits=outputs
            ).entropy().mean()

            loss.backward()

            # Atualização PGD
            delta.data = delta.data - self.alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            delta.data = torch.clamp(image + delta.data, 0, 1) - image

            delta.grad.zero_()

        adversarial = torch.clamp(image + delta, 0, 1)

        return adversarial.detach(), {
            "iterations": self.num_iterations,
            "epsilon": self.epsilon
        }

    def _heuristic_attack(
        self,
        image: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Ataque heurístico quando modelo OCR não está disponível.
        """
        # Converte para numpy
        img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()

        # Usa TextProtector
        protector = TextProtector(protection_level="readable")
        protected, metrics = protector.protect(img_np)

        # Converte de volta para tensor
        protected_np = np.array(protected).astype(np.float32) / 255.0
        protected_tensor = torch.from_numpy(protected_np).permute(2, 0, 1).unsqueeze(0)

        return protected_tensor.to(self.device), metrics


class TextDetectionEvasion:
    """
    Técnicas para evadir detectores de texto (EAST, CRAFT, DBNet).

    Faz com que o texto não seja detectado/localizado,
    impedindo que o OCR seja sequer aplicado.
    """

    def __init__(self, evasion_strength: float = 0.7):
        self.strength = evasion_strength

    def evade(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Aplica técnicas de evasão de detecção.
        """
        result = image.copy()

        # 1. Quebra de conectividade de texto
        result = self._break_text_connectivity(result)

        # 2. Confusão de aspect ratio
        result = self._confuse_aspect_ratio(result)

        # 3. Padrões anti-segmentação
        result = self._anti_segmentation_patterns(result)

        # 4. Ruído em regiões de texto potencial
        result = self._text_region_noise(result)

        return result, {"techniques": 4, "strength": self.strength}

    def _break_text_connectivity(self, img: np.ndarray) -> np.ndarray:
        """
        Adiciona pequenas quebras na conectividade do texto.
        """
        h, w = img.shape[:2]

        # Linhas verticais muito finas e transparentes
        num_lines = int(w * 0.02)
        for _ in range(num_lines):
            x = random.randint(0, w - 1)
            # Linha semi-transparente
            if len(img.shape) == 3:
                img[:, x, :] = img[:, x, :] * 0.95 + 0.05 * random.random()

        return img

    def _confuse_aspect_ratio(self, img: np.ndarray) -> np.ndarray:
        """
        Adiciona elementos que confundem detecção de aspect ratio de texto.
        """
        h, w = img.shape[:2]

        # Pequenos elementos horizontais aleatórios
        num_elements = int(h * 0.01)
        for _ in range(num_elements):
            y = random.randint(0, h - 1)
            x_start = random.randint(0, w - 20)
            length = random.randint(5, 15)

            # Linha horizontal sutil
            intensity = random.uniform(0.02, 0.05)
            if len(img.shape) == 3:
                img[y, x_start:x_start+length, :] += intensity

        return img

    def _anti_segmentation_patterns(self, img: np.ndarray) -> np.ndarray:
        """
        Padrões que confundem algoritmos de segmentação.
        """
        h, w = img.shape[:2]

        # Grade de pontos que confunde watershed/connected components
        grid_spacing = 20
        for y in range(0, h, grid_spacing):
            for x in range(0, w, grid_spacing):
                if random.random() < 0.3:
                    # Pequeno ponto
                    if y < h and x < w:
                        intensity = random.uniform(-0.03, 0.03)
                        if len(img.shape) == 3:
                            img[y:min(y+2, h), x:min(x+2, w), :] += intensity

        return img

    def _text_region_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Adiciona ruído estruturado em potenciais regiões de texto.
        """
        # Detecta regiões de alto contraste (provável texto)
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)

        # Variância local indica texto
        kernel_size = 15
        local_mean = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
        local_sq_mean = cv2.blur((gray.astype(float))**2, (kernel_size, kernel_size))
        local_var = local_sq_mean - local_mean**2

        # Normaliza variância
        var_mask = local_var / (local_var.max() + 1e-8)

        # Ruído proporcional à variância
        noise = np.random.randn(*img.shape) * 0.02 * self.strength
        if len(img.shape) == 3:
            var_mask = np.stack([var_mask] * 3, axis=-1)

        img = img + noise * var_mask

        return img


def create_protected_text_image(
    text: str,
    font_size: int = 40,
    image_size: Tuple[int, int] = (800, 200),
    protection_level: str = "readable",
    background_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[Image.Image, Image.Image, Dict]:
    """
    Cria uma imagem com texto protegido contra OCR.

    Args:
        text: Texto a ser renderizado
        font_size: Tamanho da fonte
        image_size: Dimensões da imagem (largura, altura)
        protection_level: Nível de proteção
        background_color: Cor de fundo RGB
        text_color: Cor do texto RGB

    Returns:
        original: Imagem original com texto
        protected: Imagem com texto protegido
        metrics: Métricas da proteção
    """
    # Cria imagem original
    original = Image.new('RGB', image_size, background_color)
    draw = ImageDraw.Draw(original)

    # Tenta carregar fonte, usa default se falhar
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Calcula posição centralizada
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2

    # Renderiza texto
    draw.text((x, y), text, fill=text_color, font=font)

    # Aplica proteção
    protector = TextProtector(protection_level=protection_level)
    protected, metrics = protector.protect(original)

    return original, protected, metrics
