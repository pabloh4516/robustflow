"""
OCR Models Integration Module
=============================

Integração com múltiplos engines de OCR para testar
a eficácia da proteção anti-OCR.

MODELOS SUPORTADOS:
1. Tesseract OCR (tesseract-ocr)
2. EasyOCR (deep learning based)
3. PaddleOCR (alta precisão)
4. TrOCR (transformer based - opcional)

Este módulo permite comparar o texto reconhecido antes
e depois da proteção, quantificando a eficácia.
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Resultado de uma execução de OCR."""
    text: str
    confidence: float
    engine: str
    bounding_boxes: List[dict] = None
    word_confidences: List[float] = None


class OCRManager:
    """
    Gerenciador de múltiplos engines OCR.

    Permite testar imagens contra vários OCRs simultaneamente
    para avaliar robustez da proteção.
    """

    def __init__(self):
        self.engines = {}
        self._init_engines()

    def _init_engines(self):
        """Inicializa os engines OCR disponíveis."""

        # Tesseract
        try:
            import pytesseract
            self.engines["tesseract"] = TesseractEngine()
            logger.info("Tesseract OCR inicializado")
        except ImportError:
            logger.warning("Tesseract não disponível. Instale: pip install pytesseract")

        # EasyOCR
        try:
            import easyocr
            self.engines["easyocr"] = EasyOCREngine()
            logger.info("EasyOCR inicializado")
        except ImportError:
            logger.warning("EasyOCR não disponível. Instale: pip install easyocr")

        # PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.engines["paddleocr"] = PaddleOCREngine()
            logger.info("PaddleOCR inicializado")
        except ImportError:
            logger.warning("PaddleOCR não disponível. Instale: pip install paddleocr")

        # Fallback: engine simulado para testes
        if not self.engines:
            logger.warning("Nenhum OCR real disponível. Usando simulador.")
            self.engines["simulator"] = SimulatorEngine()

    def get_available_engines(self) -> List[str]:
        """Retorna lista de engines disponíveis."""
        return list(self.engines.keys())

    def recognize(
        self,
        image: Union[Image.Image, np.ndarray],
        engines: List[str] = None,
        language: str = "eng"
    ) -> Dict[str, OCRResult]:
        """
        Executa OCR com múltiplos engines.

        Args:
            image: Imagem para reconhecimento
            engines: Lista de engines a usar (None = todos)
            language: Código do idioma

        Returns:
            Dict mapeando nome do engine para resultado
        """
        if engines is None:
            engines = list(self.engines.keys())

        results = {}

        for engine_name in engines:
            if engine_name in self.engines:
                try:
                    result = self.engines[engine_name].recognize(image, language)
                    results[engine_name] = result
                except Exception as e:
                    logger.error(f"Erro no {engine_name}: {e}")
                    results[engine_name] = OCRResult(
                        text="[ERROR]",
                        confidence=0.0,
                        engine=engine_name
                    )

        return results

    def compare_protection(
        self,
        original_image: Union[Image.Image, np.ndarray],
        protected_image: Union[Image.Image, np.ndarray],
        ground_truth: str = None,
        engines: List[str] = None
    ) -> Dict:
        """
        Compara reconhecimento antes e depois da proteção.

        Args:
            original_image: Imagem original
            protected_image: Imagem protegida
            ground_truth: Texto real (para cálculo de acurácia)
            engines: Engines a utilizar

        Returns:
            Dict com métricas de comparação
        """
        # OCR nas duas versões
        original_results = self.recognize(original_image, engines)
        protected_results = self.recognize(protected_image, engines)

        comparison = {
            "engines": {},
            "summary": {
                "average_accuracy_drop": 0,
                "average_confidence_drop": 0,
                "protection_success_rate": 0
            }
        }

        total_accuracy_drop = 0
        total_confidence_drop = 0
        success_count = 0

        for engine_name in original_results:
            orig = original_results[engine_name]
            prot = protected_results.get(engine_name)

            if prot is None:
                continue

            engine_comparison = {
                "original_text": orig.text,
                "protected_text": prot.text,
                "original_confidence": orig.confidence,
                "protected_confidence": prot.confidence,
                "confidence_drop": orig.confidence - prot.confidence
            }

            # Similaridade de texto
            similarity = SequenceMatcher(
                None,
                orig.text.lower(),
                prot.text.lower()
            ).ratio()

            engine_comparison["text_similarity"] = similarity
            engine_comparison["text_change"] = 1 - similarity

            # Se ground truth fornecido, calcula acurácia
            if ground_truth:
                orig_accuracy = SequenceMatcher(
                    None,
                    orig.text.lower(),
                    ground_truth.lower()
                ).ratio()

                prot_accuracy = SequenceMatcher(
                    None,
                    prot.text.lower(),
                    ground_truth.lower()
                ).ratio()

                engine_comparison["original_accuracy"] = orig_accuracy
                engine_comparison["protected_accuracy"] = prot_accuracy
                engine_comparison["accuracy_drop"] = orig_accuracy - prot_accuracy

                total_accuracy_drop += engine_comparison["accuracy_drop"]

            # Proteção bem-sucedida se texto mudou significativamente
            # ou confiança caiu muito
            protection_success = (
                similarity < 0.5 or
                (orig.confidence - prot.confidence) > 0.3 or
                prot.confidence < 0.3
            )

            engine_comparison["protection_success"] = protection_success

            if protection_success:
                success_count += 1

            total_confidence_drop += engine_comparison["confidence_drop"]

            comparison["engines"][engine_name] = engine_comparison

        # Calcula médias
        num_engines = len(comparison["engines"])
        if num_engines > 0:
            comparison["summary"]["average_confidence_drop"] = (
                total_confidence_drop / num_engines
            )
            comparison["summary"]["protection_success_rate"] = (
                success_count / num_engines
            )

            if ground_truth:
                comparison["summary"]["average_accuracy_drop"] = (
                    total_accuracy_drop / num_engines
                )

        return comparison


class TesseractEngine:
    """Engine baseado em Tesseract OCR."""

    def __init__(self):
        import pytesseract
        self.pytesseract = pytesseract

    def recognize(
        self,
        image: Union[Image.Image, np.ndarray],
        language: str = "eng"
    ) -> OCRResult:
        """Executa reconhecimento com Tesseract."""

        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Reconhecimento com dados detalhados
        data = self.pytesseract.image_to_data(
            image,
            lang=language,
            output_type=self.pytesseract.Output.DICT
        )

        # Extrai texto e confiança
        text_parts = []
        confidences = []
        boxes = []

        for i, word in enumerate(data['text']):
            if word.strip():
                text_parts.append(word)
                conf = int(data['conf'][i])
                if conf > 0:
                    confidences.append(conf / 100.0)
                    boxes.append({
                        'text': word,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': conf / 100.0
                    })

        text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return OCRResult(
            text=text,
            confidence=float(avg_confidence),
            engine="tesseract",
            bounding_boxes=boxes,
            word_confidences=confidences
        )


class EasyOCREngine:
    """Engine baseado em EasyOCR (Deep Learning)."""

    def __init__(self):
        import easyocr
        self.reader = None
        self._languages = {}

    def _get_reader(self, language: str):
        """Obtém reader para o idioma (lazy loading)."""
        lang_map = {
            'eng': 'en',
            'por': 'pt',
            'spa': 'es',
            'fra': 'fr',
            'deu': 'de'
        }

        lang = lang_map.get(language, language)

        if lang not in self._languages:
            import easyocr
            self._languages[lang] = easyocr.Reader([lang], gpu=False)

        return self._languages[lang]

    def recognize(
        self,
        image: Union[Image.Image, np.ndarray],
        language: str = "eng"
    ) -> OCRResult:
        """Executa reconhecimento com EasyOCR."""

        reader = self._get_reader(language)

        if isinstance(image, Image.Image):
            image = np.array(image)

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Reconhecimento
        results = reader.readtext(image)

        text_parts = []
        confidences = []
        boxes = []

        for (bbox, text, conf) in results:
            text_parts.append(text)
            confidences.append(conf)

            # Converte bbox para formato padrão
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]

            boxes.append({
                'text': text,
                'x': min(x_coords),
                'y': min(y_coords),
                'width': max(x_coords) - min(x_coords),
                'height': max(y_coords) - min(y_coords),
                'confidence': conf
            })

        text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return OCRResult(
            text=text,
            confidence=float(avg_confidence),
            engine="easyocr",
            bounding_boxes=boxes,
            word_confidences=confidences
        )


class PaddleOCREngine:
    """Engine baseado em PaddleOCR."""

    def __init__(self):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def recognize(
        self,
        image: Union[Image.Image, np.ndarray],
        language: str = "eng"
    ) -> OCRResult:
        """Executa reconhecimento com PaddleOCR."""

        if isinstance(image, Image.Image):
            image = np.array(image)

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Reconhecimento
        results = self.ocr.ocr(image, cls=True)

        text_parts = []
        confidences = []
        boxes = []

        if results and results[0]:
            for line in results[0]:
                bbox, (text, conf) = line
                text_parts.append(text)
                confidences.append(conf)

                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]

                boxes.append({
                    'text': text,
                    'x': min(x_coords),
                    'y': min(y_coords),
                    'width': max(x_coords) - min(x_coords),
                    'height': max(y_coords) - min(y_coords),
                    'confidence': conf
                })

        text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return OCRResult(
            text=text,
            confidence=float(avg_confidence),
            engine="paddleocr",
            bounding_boxes=boxes,
            word_confidences=confidences
        )


class SimulatorEngine:
    """
    Engine simulador para testes quando OCR real não está disponível.

    Simula comportamento de OCR com base em análise de imagem.
    """

    def recognize(
        self,
        image: Union[Image.Image, np.ndarray],
        language: str = "eng"
    ) -> OCRResult:
        """Simula reconhecimento OCR."""

        if isinstance(image, Image.Image):
            image = np.array(image)

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Análise básica da imagem para simular confiança
        # Imagens com alto contraste e pouco ruído = alta confiança

        # Calcula contraste
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        contrast = gray.std() / 128.0  # Normalizado
        noise_level = self._estimate_noise(gray)

        # Confiança simulada baseada em contraste e ruído
        base_confidence = min(0.95, contrast)
        confidence = max(0.0, base_confidence - noise_level)

        # Texto simulado (em produção seria texto real)
        simulated_text = "[SIMULATED - Install OCR engines for real results]"

        return OCRResult(
            text=simulated_text,
            confidence=float(confidence),
            engine="simulator",
            bounding_boxes=[],
            word_confidences=[confidence]
        )

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estima nível de ruído na imagem."""
        # Usa Laplaciano para detectar ruído de alta frequência
        import cv2
        laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
        noise = laplacian.var() / 10000.0  # Normalizado
        return min(0.9, noise)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calcula Character Error Rate (CER).

    CER = (S + D + I) / N

    Onde:
        S = substituições
        D = deleções
        I = inserções
        N = caracteres na referência
    """
    if not reference:
        return 1.0 if hypothesis else 0.0

    # Remove espaços extras e normaliza
    ref = re.sub(r'\s+', ' ', reference.strip().lower())
    hyp = re.sub(r'\s+', ' ', hypothesis.strip().lower())

    # Distância de edição de caracteres
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    edit_distance = dp[m][n]
    cer = edit_distance / max(1, len(ref))

    return min(1.0, cer)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calcula Word Error Rate (WER).

    Similar ao CER mas opera em nível de palavras.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 1.0 if hyp_words else 0.0

    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    wer = dp[m][n] / max(1, len(ref_words))

    return min(1.0, wer)
