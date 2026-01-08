"""
RobustnessLab - FastAPI Backend
===============================

API REST para análise de robustez adversária de modelos de visão computacional.

Este backend fornece endpoints para:
- Upload e processamento de imagens
- Execução de ataques PGD
- Análise de confiança e predições
- Geração de visualizações de perturbação

Uso:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import numpy as np
from PIL import Image
import io
import base64
import logging

from pgd_attack import PGDAttack, get_top_k_predictions
from models import (
    ModelManager,
    preprocess_image,
    tensor_to_image,
    create_normalized_model,
    load_imagenet_labels
)
from text_protection import (
    TextProtector,
    TextDetectionEvasion,
    create_protected_text_image
)
from ocr_models import OCRManager, calculate_cer, calculate_wer
from text_camouflage import (
    TextCamouflage,
    VisualTextCamouflage,
    CamouflageMode,
    get_camouflage_modes,
    demonstrate_camouflage
)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# INICIALIZAÇÃO DA APLICAÇÃO
# ================================================================
app = FastAPI(
    title="RobustnessLab API",
    description="API para análise de robustez adversária de DNNs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuração de CORS para permitir requests do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detecta dispositivo disponível
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Dispositivo de computação: {DEVICE}")

# Inicializa gerenciador de modelos
model_manager = ModelManager(device=DEVICE)

# Inicializa gerenciador de OCR
ocr_manager = OCRManager()
logger.info(f"Engines OCR disponíveis: {ocr_manager.get_available_engines()}")


# ================================================================
# SCHEMAS PYDANTIC
# ================================================================
class AttackParams(BaseModel):
    """Parâmetros para o ataque PGD."""
    epsilon: float = Field(
        default=8/255,
        ge=0,
        le=1,
        description="Magnitude da perturbação (norma L∞). Valor típico: 8/255 ≈ 0.031"
    )
    alpha: float = Field(
        default=2/255,
        ge=0,
        le=1,
        description="Step size por iteração. Valor típico: 2/255 ≈ 0.008"
    )
    num_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Número de iterações do PGD"
    )
    model_id: str = Field(
        default="resnet50",
        description="ID do modelo alvo"
    )


class AttackResponse(BaseModel):
    """Resposta do endpoint de ataque."""
    success: bool
    original_image: str  # Base64
    adversarial_image: str  # Base64
    perturbation_heatmap: str  # Base64
    original_predictions: List[dict]
    adversarial_predictions: List[dict]
    metrics: dict
    attack_params: dict


class ModelInfo(BaseModel):
    """Informações sobre um modelo."""
    id: str
    name: str
    input_size: int
    description: str


# ================================================================
# FUNÇÕES AUXILIARES
# ================================================================
def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Converte imagem PIL para string base64."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def numpy_to_base64_heatmap(array: np.ndarray) -> str:
    """
    Converte array numpy de heatmap para imagem base64.
    Aplica colormap para visualização.
    """
    import matplotlib
    matplotlib.use('Agg')  # Backend não-interativo
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # Aplica colormap 'hot' para visualização clara
    colored = cm.hot(array)
    colored = (colored[:, :, :3] * 255).astype(np.uint8)

    image = Image.fromarray(colored)
    return image_to_base64(image)


# ================================================================
# ENDPOINTS DA API
# ================================================================
@app.get("/")
async def root():
    """Endpoint raiz - health check."""
    return {
        "status": "online",
        "service": "RobustnessLab API",
        "device": DEVICE,
        "torch_version": torch.__version__
    }


@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """
    Lista todos os modelos disponíveis para ataque.

    Returns:
        Lista de modelos com suas especificações
    """
    return model_manager.get_available_models()


@app.post("/attack")
async def perform_attack(
    file: UploadFile = File(..., description="Imagem para atacar"),
    epsilon: float = Form(default=0.031, description="Magnitude da perturbação"),
    alpha: float = Form(default=0.008, description="Step size"),
    num_iterations: int = Form(default=10, description="Número de iterações"),
    model_id: str = Form(default="resnet50", description="ID do modelo")
):
    """
    Executa ataque PGD em uma imagem.

    PROCESSO:
    1. Carrega a imagem do upload
    2. Pré-processa para o modelo selecionado
    3. Obtém predição original
    4. Executa ataque PGD
    5. Gera visualizações comparativas
    6. Retorna métricas e imagens

    Args:
        file: Arquivo de imagem (JPEG, PNG)
        epsilon: Orçamento de perturbação L∞
        alpha: Tamanho do passo por iteração
        num_iterations: Iterações do PGD
        model_id: Modelo a ser atacado

    Returns:
        AttackResponse com imagens e métricas
    """
    try:
        # Valida tipo de arquivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Arquivo deve ser uma imagem"
            )

        # Carrega e converte imagem
        logger.info(f"Processando imagem: {file.filename}")
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Carrega modelo
        logger.info(f"Carregando modelo: {model_id}")
        base_model = model_manager.load_model(model_id)

        # Cria modelo com normalização integrada
        # Isso permite que o PGD opere em [0,1] enquanto
        # o modelo recebe entradas normalizadas
        model = create_normalized_model(base_model)
        model.to(DEVICE)
        model.eval()

        # Pré-processa imagem
        image_tensor = preprocess_image(image, model_id, model_manager)
        image_tensor = image_tensor.to(DEVICE)

        # Obtém predição original para usar como label no ataque
        with torch.no_grad():
            original_output = model(image_tensor)
            original_pred = original_output.argmax(dim=1)

        # ============================================================
        # EXECUTA ATAQUE PGD
        # ============================================================
        logger.info(
            f"Iniciando ataque PGD: ε={epsilon}, α={alpha}, "
            f"iterações={num_iterations}"
        )

        attacker = PGDAttack(
            model=model,
            epsilon=epsilon,
            alpha=alpha,
            num_iterations=num_iterations,
            device=DEVICE
        )

        # Executa ataque não-direcionado
        # O objetivo é maximizar o erro na classe original
        adversarial_tensor, metrics = attacker.attack(
            images=image_tensor,
            labels=original_pred,
            targeted=False
        )

        # ============================================================
        # GERA VISUALIZAÇÕES
        # ============================================================
        # Converte tensores para imagens
        original_pil = tensor_to_image(image_tensor)
        adversarial_pil = tensor_to_image(adversarial_tensor)

        # Gera heatmap de perturbação
        heatmap = attacker.get_perturbation_heatmap(
            image_tensor, adversarial_tensor
        )

        # Carrega labels do ImageNet
        labels = load_imagenet_labels()

        # Obtém top-5 predições
        original_top5 = get_top_k_predictions(
            metrics["original_probs"][0], labels, k=5
        )
        adversarial_top5 = get_top_k_predictions(
            metrics["adversarial_probs"][0], labels, k=5
        )

        # ============================================================
        # PREPARA RESPOSTA
        # ============================================================
        response = {
            "success": True,
            "original_image": image_to_base64(original_pil),
            "adversarial_image": image_to_base64(adversarial_pil),
            "perturbation_heatmap": numpy_to_base64_heatmap(heatmap),
            "original_predictions": original_top5,
            "adversarial_predictions": adversarial_top5,
            "metrics": {
                "perturbation_linf": float(metrics["perturbation_linf"]),
                "perturbation_l2": float(metrics["perturbation_l2"]),
                "attack_success": bool(metrics["attack_success"]),
                "original_class": int(metrics["original_pred"][0]),
                "adversarial_class": int(metrics["adversarial_pred"][0]),
                "original_confidence": float(metrics["original_confidence"][0]),
                "adversarial_confidence": float(metrics["adversarial_confidence"][0]),
                "loss_history": metrics["loss_history"]
            },
            "attack_params": {
                "epsilon": epsilon,
                "alpha": alpha,
                "num_iterations": num_iterations,
                "model_id": model_id
            }
        }

        logger.info(
            f"Ataque concluído. Sucesso: {metrics['attack_success']}, "
            f"Classe original: {original_top5[0]['label']} → "
            f"Classe adversária: {adversarial_top5[0]['label']}"
        )

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Erro durante ataque: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante processamento: {str(e)}"
        )


@app.post("/attack/download")
async def download_adversarial(
    file: UploadFile = File(..., description="Imagem para atacar"),
    epsilon: float = Form(default=0.031),
    alpha: float = Form(default=0.008),
    num_iterations: int = Form(default=10),
    model_id: str = Form(default="resnet50"),
    output_width: int = Form(default=1024, description="Largura de saída"),
    output_height: int = Form(default=1024, description="Altura de saída"),
    output_format: str = Form(default="PNG", description="Formato: PNG, JPEG, WEBP")
):
    """
    Executa ataque e retorna imagem adversária em ALTA RESOLUÇÃO.
    A imagem é upscaled para as dimensões especificadas.
    """
    try:
        image_data = await file.read()
        original_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        original_size = original_image.size  # Guarda tamanho original

        # Usa tamanho original se não especificado
        if output_width == 1024 and output_height == 1024:
            output_width, output_height = original_size

        # Carrega modelo e executa ataque
        base_model = model_manager.load_model(model_id)
        model = create_normalized_model(base_model)
        model.to(DEVICE)
        model.eval()

        image_tensor = preprocess_image(original_image, model_id, model_manager)
        image_tensor = image_tensor.to(DEVICE)

        with torch.no_grad():
            original_pred = model(image_tensor).argmax(dim=1)

        attacker = PGDAttack(
            model=model, epsilon=epsilon, alpha=alpha,
            num_iterations=num_iterations, device=DEVICE
        )

        adversarial_tensor, _ = attacker.attack(
            images=image_tensor, labels=original_pred, targeted=False
        )

        # Converte para PIL e faz upscale para alta resolução
        adversarial_pil = tensor_to_image(adversarial_tensor)
        adversarial_hires = adversarial_pil.resize(
            (output_width, output_height),
            Image.Resampling.LANCZOS
        )

        # Salva em buffer
        buffer = io.BytesIO()
        if output_format.upper() == "JPEG":
            adversarial_hires.save(buffer, format="JPEG", quality=100, subsampling=0)
            media_type = "image/jpeg"
        elif output_format.upper() == "WEBP":
            adversarial_hires.save(buffer, format="WEBP", quality=100, lossless=True)
            media_type = "image/webp"
        else:
            adversarial_hires.save(buffer, format="PNG", compress_level=1)
            media_type = "image/png"

        buffer.seek(0)

        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            buffer,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=adversarial_{output_width}x{output_height}.{output_format.lower()}"
            }
        )

    except Exception as e:
        logger.error(f"Erro no download: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = Form(default="resnet50")
):
    """
    Realiza apenas predição (sem ataque) em uma imagem.

    Útil para obter baseline antes do ataque.
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        base_model = model_manager.load_model(model_id)
        model = create_normalized_model(base_model)
        model.to(DEVICE)
        model.eval()

        image_tensor = preprocess_image(image, model_id, model_manager)
        image_tensor = image_tensor.to(DEVICE)

        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)

        labels = load_imagenet_labels()
        top5 = get_top_k_predictions(probs[0].cpu().numpy(), labels, k=5)

        return {
            "success": True,
            "predictions": top5,
            "model_id": model_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Verifica saúde do serviço."""
    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": list(model_manager._model_cache.keys()),
        "ocr_engines": ocr_manager.get_available_engines()
    }


# ================================================================
# ENDPOINTS DE PROTEÇÃO DE TEXTO ANTI-OCR
# ================================================================

@app.get("/text-protection/levels")
async def get_protection_levels():
    """
    Retorna níveis de proteção disponíveis.
    """
    return {
        "levels": [
            {
                "id": "readable",
                "name": "Legível (Recomendado)",
                "description": "Texto 100% perfeito para humanos. Usa ataques invisíveis avançados que devastam OCR.",
                "estimated_ocr_drop": "70-90%",
                "recommended": True
            },
            {
                "id": "stealth",
                "name": "Stealth",
                "description": "Perturbações 100% invisíveis. Imagem idêntica ao original para humanos.",
                "estimated_ocr_drop": "40-60%"
            },
            {
                "id": "low",
                "name": "Baixo",
                "description": "Perturbação sutil. Pode não confundir todos os OCRs.",
                "estimated_ocr_drop": "30-50%"
            },
            {
                "id": "medium",
                "name": "Médio",
                "description": "Balanço entre invisibilidade e eficácia.",
                "estimated_ocr_drop": "50-70%"
            },
            {
                "id": "high",
                "name": "Alto",
                "description": "Alta eficácia contra maioria dos OCRs. Pode ter artefatos leves.",
                "estimated_ocr_drop": "70-85%"
            },
            {
                "id": "maximum",
                "name": "Máximo",
                "description": "Proteção máxima. TERÁ artefatos visíveis.",
                "estimated_ocr_drop": "85-95%"
            }
        ]
    }


@app.get("/text-protection/techniques")
async def get_available_techniques():
    """
    Lista todas as técnicas de proteção disponíveis.
    """
    return {
        "techniques": [
            {
                "id": "adversarial_noise",
                "name": "Ruído Adversarial",
                "description": "Ruído multi-escala otimizado para confundir CNNs"
            },
            {
                "id": "structured_pattern",
                "name": "Padrão Estruturado",
                "description": "Interferência senoidal que afeta segmentação"
            },
            {
                "id": "frequency_perturbation",
                "name": "Perturbação de Frequência",
                "description": "Ruído no domínio FFT em frequências médias"
            },
            {
                "id": "geometric_distortion",
                "name": "Distorção Geométrica",
                "description": "Micro-ondulações que quebram alinhamento"
            },
            {
                "id": "edge_disruption",
                "name": "Disrupção de Bordas",
                "description": "Ruído seletivo nas bordas dos caracteres"
            },
            {
                "id": "color_channel_shift",
                "name": "Deslocamento de Canais",
                "description": "Aberração cromática artificial"
            },
            {
                "id": "adversarial_texture",
                "name": "Textura Adversarial",
                "description": "Padrões Gabor que criam features falsas"
            },
            {
                "id": "micro_patterns",
                "name": "Micro Padrões",
                "description": "Checkerboard em nível de pixel"
            },
            {
                "id": "gradient_masking",
                "name": "Mascaramento de Gradiente",
                "description": "Componentes com derivadas problemáticas"
            },
            {
                "id": "dithering_noise",
                "name": "Ruído Dithering",
                "description": "Padrão Bayer similar a impressão"
            }
        ]
    }


@app.post("/text-protection/protect")
async def protect_text_image(
    file: UploadFile = File(..., description="Imagem contendo texto"),
    protection_level: str = Form(default="readable", description="Nível de proteção"),
    techniques: str = Form(default="all", description="Técnicas a usar (comma-separated ou 'all')"),
    preserve_colors: bool = Form(default=True, description="Preservar tons de cor"),
    ground_truth: str = Form(default="", description="Texto real para cálculo de métricas")
):
    """
    Protege texto em uma imagem contra reconhecimento por OCR.

    O texto permanece legível para humanos mas se torna
    ilegível para sistemas de OCR automatizados.

    TÉCNICAS APLICADAS:
    - Ruído adversarial multi-escala
    - Padrões de interferência estruturada
    - Perturbação no domínio da frequência
    - Distorções geométricas micro
    - Disrupção seletiva de bordas
    - Desalinhamento de canais de cor
    - Texturas adversariais
    - Micro-padrões de pixel
    - Mascaramento de gradiente
    - Ruído dithering

    Returns:
        Imagem protegida + métricas de eficácia
    """
    try:
        # Valida arquivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")

        logger.info(f"Protegendo imagem: {file.filename}, nível: {protection_level}")

        # Carrega imagem
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Parse técnicas
        if techniques.lower() == "all":
            technique_list = None  # Usa todas
        else:
            technique_list = [t.strip() for t in techniques.split(",")]

        # Aplica proteção
        protector = TextProtector(
            device=DEVICE,
            protection_level=protection_level
        )

        protected_image, protection_metrics = protector.protect(
            image,
            techniques=technique_list,
            preserve_colors=preserve_colors
        )

        # Aplica evasão de detecção adicional APENAS para níveis não-readable
        evasion_metrics = {}
        if protection_level not in ["readable", "stealth"]:
            evasion = TextDetectionEvasion(evasion_strength=0.5)
            protected_array = np.array(protected_image).astype(np.float32) / 255.0
            protected_array, evasion_metrics = evasion.evade(protected_array)
            protected_image = Image.fromarray((protected_array * 255).astype(np.uint8))

        # Testa eficácia com OCR
        ocr_comparison = ocr_manager.compare_protection(
            original_image=image,
            protected_image=protected_image,
            ground_truth=ground_truth if ground_truth else None
        )

        # Gera heatmap de diferença
        original_array = np.array(image).astype(np.float32) / 255.0
        protected_array = np.array(protected_image).astype(np.float32) / 255.0
        diff = np.abs(protected_array - original_array)
        diff_heatmap = diff.mean(axis=2)  # Média dos canais
        diff_heatmap = diff_heatmap / (diff_heatmap.max() + 1e-8)

        # Prepara resposta
        response = {
            "success": True,
            "original_image": image_to_base64(image),
            "protected_image": image_to_base64(protected_image),
            "difference_heatmap": numpy_to_base64_heatmap(diff_heatmap),
            "protection_metrics": {
                "techniques_applied": protection_metrics["techniques_applied"],
                "mse": protection_metrics["mse"],
                "psnr": protection_metrics["psnr"],
                "estimated_ocr_accuracy_drop": protection_metrics["estimated_ocr_accuracy_drop"]
            },
            "ocr_comparison": ocr_comparison,
            "parameters": {
                "protection_level": protection_level,
                "techniques": technique_list or "all",
                "preserve_colors": preserve_colors
            }
        }

        logger.info(
            f"Proteção concluída. Taxa de sucesso: "
            f"{ocr_comparison['summary']['protection_success_rate']*100:.1f}%"
        )

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Erro na proteção: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text-protection/create")
async def create_protected_text(
    text: str = Form(..., description="Texto a renderizar"),
    font_size: int = Form(default=40, description="Tamanho da fonte"),
    width: int = Form(default=800, description="Largura da imagem"),
    height: int = Form(default=200, description="Altura da imagem"),
    protection_level: str = Form(default="readable", description="Nível de proteção"),
    background_color: str = Form(default="#FFFFFF", description="Cor de fundo (hex)"),
    text_color: str = Form(default="#000000", description="Cor do texto (hex)")
):
    """
    Cria uma imagem com texto já protegido contra OCR.

    Útil para gerar textos anti-scraping, CAPTCHAs leves,
    ou proteger informações sensíveis em imagens.
    """
    try:
        logger.info(f"Criando texto protegido: '{text[:20]}...'")

        # Parse cores hex
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        bg_rgb = hex_to_rgb(background_color)
        text_rgb = hex_to_rgb(text_color)

        # Cria imagens
        original, protected, metrics = create_protected_text_image(
            text=text,
            font_size=font_size,
            image_size=(width, height),
            protection_level=protection_level,
            background_color=bg_rgb,
            text_color=text_rgb
        )

        # Testa OCR
        ocr_comparison = ocr_manager.compare_protection(
            original_image=original,
            protected_image=protected,
            ground_truth=text
        )

        response = {
            "success": True,
            "original_image": image_to_base64(original),
            "protected_image": image_to_base64(protected),
            "text": text,
            "protection_metrics": metrics,
            "ocr_comparison": ocr_comparison,
            "parameters": {
                "font_size": font_size,
                "dimensions": f"{width}x{height}",
                "protection_level": protection_level
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Erro ao criar texto: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text-protection/test-ocr")
async def test_ocr(
    file: UploadFile = File(..., description="Imagem para teste de OCR"),
    engines: str = Form(default="all", description="Engines a usar (comma-separated ou 'all')")
):
    """
    Testa uma imagem contra múltiplos engines de OCR.

    Útil para verificar eficácia da proteção ou
    comparar diferentes engines.
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Parse engines
        if engines.lower() == "all":
            engine_list = None
        else:
            engine_list = [e.strip() for e in engines.split(",")]

        # Executa OCR
        results = ocr_manager.recognize(image, engines=engine_list)

        # Formata resposta
        formatted_results = {}
        for engine, result in results.items():
            formatted_results[engine] = {
                "text": result.text,
                "confidence": result.confidence,
                "word_count": len(result.text.split()) if result.text else 0,
                "bounding_boxes": result.bounding_boxes
            }

        return {
            "success": True,
            "engines_used": list(results.keys()),
            "results": formatted_results,
            "image_size": f"{image.width}x{image.height}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/text-protection/ocr-engines")
async def get_ocr_engines():
    """
    Lista engines de OCR disponíveis no sistema.
    """
    return {
        "available_engines": ocr_manager.get_available_engines(),
        "note": "Instale pacotes adicionais para mais engines: pytesseract, easyocr, paddleocr"
    }


@app.post("/text-protection/download")
async def download_protected_image(
    file: UploadFile = File(..., description="Imagem contendo texto"),
    protection_level: str = Form(default="readable"),
    quality: int = Form(default=100, ge=1, le=100, description="Qualidade (1-100)"),
    format: str = Form(default="PNG", description="Formato: PNG, JPEG, WEBP"),
    scale: float = Form(default=1.0, ge=0.5, le=4.0, description="Escala (0.5x a 4x)")
):
    """
    Protege imagem e retorna para download direto em alta qualidade.

    Parâmetros:
    - quality: 1-100 (100 = máxima qualidade)
    - format: PNG (lossless), JPEG, WEBP
    - scale: Multiplica resolução (2.0 = dobra tamanho)
    """
    try:
        # Carrega imagem
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Aplica escala se necessário
        if scale != 1.0:
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Aplica proteção
        protector = TextProtector(device=DEVICE, protection_level=protection_level)
        protected_image, _ = protector.protect(image, preserve_colors=True)

        # Aplica evasão de detecção
        evasion = TextDetectionEvasion(evasion_strength=0.5)
        protected_array = np.array(protected_image).astype(np.float32) / 255.0
        protected_array, _ = evasion.evade(protected_array)
        protected_image = Image.fromarray((protected_array * 255).astype(np.uint8))

        # Prepara para download
        buffer = io.BytesIO()
        format_upper = format.upper()

        if format_upper == "PNG":
            protected_image.save(buffer, format="PNG", optimize=False)
            media_type = "image/png"
            extension = "png"
        elif format_upper == "JPEG" or format_upper == "JPG":
            protected_image.save(buffer, format="JPEG", quality=quality, subsampling=0)
            media_type = "image/jpeg"
            extension = "jpg"
        elif format_upper == "WEBP":
            protected_image.save(buffer, format="WEBP", quality=quality, lossless=(quality == 100))
            media_type = "image/webp"
            extension = "webp"
        else:
            protected_image.save(buffer, format="PNG")
            media_type = "image/png"
            extension = "png"

        buffer.seek(0)

        filename = f"protected_image_{protection_level}_{int(scale*100)}pct.{extension}"

        return StreamingResponse(
            buffer,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Image-Width": str(protected_image.width),
                "X-Image-Height": str(protected_image.height),
                "X-Protection-Level": protection_level
            }
        )

    except Exception as e:
        logger.error(f"Erro no download: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text-protection/create-download")
async def create_and_download_protected_text(
    text: str = Form(..., description="Texto a renderizar"),
    font_size: int = Form(default=60, description="Tamanho da fonte"),
    width: int = Form(default=1920, description="Largura da imagem"),
    height: int = Form(default=400, description="Altura da imagem"),
    protection_level: str = Form(default="readable"),
    background_color: str = Form(default="#FFFFFF"),
    text_color: str = Form(default="#000000"),
    quality: int = Form(default=100, ge=1, le=100),
    format: str = Form(default="PNG")
):
    """
    Cria texto protegido e retorna imagem para download em alta qualidade.

    Ideal para criar banners, cupons, e anúncios protegidos.
    """
    try:
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        bg_rgb = hex_to_rgb(background_color)
        text_rgb = hex_to_rgb(text_color)

        # Cria imagem protegida
        _, protected, _ = create_protected_text_image(
            text=text,
            font_size=font_size,
            image_size=(width, height),
            protection_level=protection_level,
            background_color=bg_rgb,
            text_color=text_rgb
        )

        # Prepara para download
        buffer = io.BytesIO()
        format_upper = format.upper()

        if format_upper == "PNG":
            protected.save(buffer, format="PNG", optimize=False)
            media_type = "image/png"
            extension = "png"
        elif format_upper in ["JPEG", "JPG"]:
            protected.save(buffer, format="JPEG", quality=quality, subsampling=0)
            media_type = "image/jpeg"
            extension = "jpg"
        elif format_upper == "WEBP":
            protected.save(buffer, format="WEBP", quality=quality, lossless=(quality == 100))
            media_type = "image/webp"
            extension = "webp"
        else:
            protected.save(buffer, format="PNG")
            media_type = "image/png"
            extension = "png"

        buffer.seek(0)

        # Sanitiza nome do arquivo
        safe_text = "".join(c if c.isalnum() else "_" for c in text[:20])
        filename = f"protected_text_{safe_text}.{extension}"

        return StreamingResponse(
            buffer,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Image-Width": str(width),
                "X-Image-Height": str(height),
                "X-Protection-Level": protection_level
            }
        )

    except Exception as e:
        logger.error(f"Erro ao criar download: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# ENDPOINTS DE CAMUFLAGEM DE TEXTO
# ================================================================

# Inicializa sistema de camuflagem
text_camouflage = TextCamouflage()
visual_camouflage = VisualTextCamouflage()


@app.get("/camouflage/modes")
async def get_camouflage_modes_endpoint():
    """
    Retorna todos os modos de camuflagem disponíveis.

    Cada modo tem diferentes características:
    - Homoglyphs: Substitui por caracteres Unicode idênticos visualmente
    - Zero-Width: Injeta caracteres invisíveis
    - Leetspeak: Estilo hacker (A→4, E→3)
    - Mixed Scripts: Mistura Latin/Cirílico/Grego
    - Combining Marks: Adiciona diacríticos invisíveis
    - Direction Trick: Manipula direção do texto
    - Full Camouflage: Todas as técnicas combinadas
    """
    return {
        "modes": get_camouflage_modes(),
        "recommendation": "full_camouflage para máxima proteção"
    }


@app.post("/camouflage/text")
async def camouflage_text(
    text: str = Form(..., description="Texto para camuflar"),
    mode: str = Form(default="full_camouflage", description="Modo de camuflagem"),
    intensity: float = Form(default=0.7, ge=0.0, le=1.0, description="Intensidade (0-1)")
):
    """
    Camufla texto para ser legível apenas por humanos.

    O texto resultante:
    - Parece idêntico ao original para humanos
    - É ilegível ou incorretamente interpretado por máquinas
    - Copia/cola resulta em texto diferente
    - OCR não consegue ler corretamente

    COMO FUNCIONA:
    - Substitui caracteres por homoglyphs (visualmente idênticos, código diferente)
    - Injeta caracteres invisíveis (zero-width) entre letras
    - Adiciona marcas combinantes Unicode sutis
    - Manipula marcadores de direção de texto

    Returns:
        Texto camuflado + análise Unicode + instruções de uso
    """
    try:
        logger.info(f"Camuflando texto: '{text[:30]}...' modo: {mode}")

        # Converte string para enum
        try:
            camouflage_mode = CamouflageMode(mode)
        except ValueError:
            camouflage_mode = CamouflageMode.FULL_CAMOUFLAGE

        # Aplica camuflagem
        result = text_camouflage.camouflage(text, camouflage_mode, intensity)

        # Tenta decodificar (para mostrar que é possível reverter parcialmente)
        decoded = text_camouflage.decode(result.camouflaged_text)

        return {
            "success": True,
            "original": {
                "text": result.original_text,
                "length": len(result.original_text),
                "bytes": len(result.original_text.encode('utf-8'))
            },
            "camouflaged": {
                "text": result.camouflaged_text,
                "length": len(result.camouflaged_text),
                "bytes": len(result.camouflaged_text.encode('utf-8')),
                "visible_length": result.unicode_analysis['visible_length']
            },
            "decoded_attempt": decoded,
            "technique": result.technique,
            "description": result.description,
            "unicode_analysis": result.unicode_analysis,
            "usage_tips": [
                "Copie o texto camuflado - ele parecerá normal",
                "Cole em qualquer lugar - humanos lerão normalmente",
                "Máquinas/OCR não conseguirão interpretar corretamente",
                "Buscas de texto não encontrarão o conteúdo",
                "Copy/paste pode resultar em caracteres estranhos"
            ]
        }

    except Exception as e:
        logger.error(f"Erro na camuflagem: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/camouflage/image")
async def camouflage_text_to_image(
    text: str = Form(..., description="Texto para camuflar"),
    mode: str = Form(default="full_camouflage", description="Modo de camuflagem"),
    font_size: int = Form(default=40, description="Tamanho da fonte"),
    width: int = Form(default=800, description="Largura da imagem"),
    height: int = Form(default=200, description="Altura da imagem"),
    background_color: str = Form(default="#FFFFFF", description="Cor de fundo"),
    text_color: str = Form(default="#000000", description="Cor do texto"),
    add_visual_noise: bool = Form(default=True, description="Adicionar perturbação visual")
):
    """
    Cria imagem com texto camuflado.

    Combina camuflagem de caracteres Unicode com perturbações visuais
    para máxima proteção contra OCR.
    """
    try:
        # Parse cores
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        bg_rgb = hex_to_rgb(background_color)
        text_rgb = hex_to_rgb(text_color)

        # Converte modo
        try:
            camouflage_mode = CamouflageMode(mode)
        except ValueError:
            camouflage_mode = CamouflageMode.FULL_CAMOUFLAGE

        # Cria imagem camuflada
        image, result = visual_camouflage.create_camouflaged_image(
            text=text,
            mode=camouflage_mode,
            font_size=font_size,
            image_size=(width, height),
            bg_color=bg_rgb,
            text_color=text_rgb,
            add_visual_noise=add_visual_noise
        )

        # Converte para base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return {
            "success": True,
            "image": image_base64,
            "original_text": result.original_text,
            "camouflaged_text": result.camouflaged_text,
            "technique": result.technique,
            "unicode_analysis": result.unicode_analysis,
            "parameters": {
                "mode": mode,
                "font_size": font_size,
                "dimensions": f"{width}x{height}",
                "visual_noise": add_visual_noise
            }
        }

    except Exception as e:
        logger.error(f"Erro ao criar imagem: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/camouflage/decode")
async def decode_camouflaged_text(
    text: str = Form(..., description="Texto camuflado para decodificar")
):
    """
    Tenta decodificar texto camuflado.

    NOTA: A decodificação não é 100% reversível para todas as técnicas.
    Caracteres zero-width e marcas combinantes são removidos,
    e homoglyphs são revertidos quando possível.
    """
    try:
        decoded = text_camouflage.decode(text)

        # Análise do texto original
        original_analysis = text_camouflage._analyze_unicode(text)
        decoded_analysis = text_camouflage._analyze_unicode(decoded)

        return {
            "success": True,
            "original_camouflaged": text,
            "decoded": decoded,
            "original_analysis": original_analysis,
            "decoded_analysis": decoded_analysis,
            "chars_removed": len(text) - len(decoded),
            "note": "Decodificação pode não ser 100% precisa para todas as técnicas"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/camouflage/demo")
async def demo_all_camouflage(
    text: str = "Senha Secreta: ABC123"
):
    """
    Demonstra todas as técnicas de camuflagem em um texto de exemplo.

    Útil para comparar os diferentes modos e escolher o mais adequado.
    """
    try:
        results = demonstrate_camouflage(text)

        return {
            "success": True,
            "original_text": text,
            "demonstrations": results,
            "recommendation": {
                "for_copy_paste": "homoglyph",
                "for_search_evasion": "zero_width",
                "for_maximum_protection": "full_camouflage",
                "for_visual_effect": "leetspeak"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/camouflage/analyze")
async def analyze_text_unicode(
    text: str = Form(..., description="Texto para analisar")
):
    """
    Analisa a composição Unicode de um texto.

    Detecta:
    - Caracteres de diferentes scripts (Latin, Cirílico, Grego)
    - Caracteres invisíveis (zero-width)
    - Marcas combinantes
    - Possível camuflagem existente
    """
    try:
        analysis = text_camouflage._analyze_unicode(text)

        # Detecta se texto parece camuflado
        is_camouflaged = (
            analysis['has_invisible_chars'] or
            analysis['has_mixed_scripts'] or
            analysis['length'] != analysis['visible_length']
        )

        return {
            "success": True,
            "text": text,
            "analysis": analysis,
            "likely_camouflaged": is_camouflaged,
            "camouflage_indicators": {
                "invisible_characters": analysis['has_invisible_chars'],
                "mixed_scripts": analysis['has_mixed_scripts'],
                "length_mismatch": analysis['length'] != analysis['visible_length']
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# EXECUÇÃO LOCAL
# ================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
