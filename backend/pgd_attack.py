"""
PGD (Projected Gradient Descent) Attack Implementation
=======================================================

Este módulo implementa o algoritmo PGD para geração de exemplos adversários.

FUNDAMENTOS MATEMÁTICOS:
------------------------
O PGD é um ataque iterativo que busca maximizar a função de perda L(θ, x+δ, y)
onde:
    - θ: parâmetros do modelo
    - x: imagem original
    - δ: perturbação adversária
    - y: label verdadeiro

A cada iteração, calculamos:
    δ_{t+1} = Π_{ε}(δ_t + α * sign(∇_x L(θ, x + δ_t, y)))

Onde:
    - α (step_size): magnitude do passo de atualização
    - sign(∇_x L): direção do gradiente (maximiza o erro)
    - Π_{ε}: projeção na bola L∞ de raio ε (garante ||δ||∞ ≤ ε)

Por que usamos o SINAL do gradiente?
------------------------------------
O gradiente ∇_x L aponta na direção de maior aumento da perda.
Ao usar sign(∇_x L), garantimos que cada pixel seja perturbado
com magnitude uniforme (±α), maximizando o impacto dentro do
orçamento de perturbação L∞.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class PGDAttack:
    """
    Implementação do ataque PGD (Projected Gradient Descent).

    O PGD é considerado um dos ataques mais fortes de primeira ordem,
    sendo amplamente utilizado para avaliar a robustez adversária de modelos.

    Attributes:
        model: Modelo PyTorch a ser atacado
        epsilon: Magnitude máxima da perturbação (norma L∞)
        alpha: Tamanho do passo por iteração
        num_iterations: Número de iterações do ataque
        device: Dispositivo de computação (CPU/GPU)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_iterations: int = 10,
        device: str = "cpu"
    ):
        """
        Inicializa o atacante PGD.

        Args:
            model: Modelo de classificação pré-treinado
            epsilon: Orçamento de perturbação L∞ (padrão: 8/255 ≈ 0.031)
                    Valores típicos: 4/255, 8/255, 16/255
            alpha: Step size por iteração (padrão: 2/255)
                   Regra prática: α ≈ ε / (num_iterations / 4)
            num_iterations: Iterações do PGD (mais = ataque mais forte)
            device: 'cpu' ou 'cuda'
        """
        self.model = model
        self.model.eval()  # Modo de avaliação (desativa dropout, batchnorm em modo eval)
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.device = device
        self.model.to(device)

        # Função de perda: Cross-Entropy Loss
        # Usamos ela porque queremos MAXIMIZAR a perda para confundir o modelo
        self.criterion = nn.CrossEntropyLoss()

    def attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        targeted: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Executa o ataque PGD não-direcionado.

        ALGORITMO PGD:
        1. Inicializa perturbação δ com ruído uniforme em [-ε, ε]
        2. Para cada iteração t:
           a. Calcula logits = model(x + δ)
           b. Calcula perda L = CrossEntropy(logits, y)
           c. Calcula gradiente ∇_δ L via backpropagation
           d. Atualiza δ = δ + α * sign(∇_δ L)  [Maximiza a perda]
           e. Projeta δ = clamp(δ, -ε, ε)       [Restrição L∞]
           f. Projeta x_adv = clamp(x + δ, 0, 1) [Valores válidos de pixel]
        3. Retorna x_adv = x + δ

        Args:
            images: Tensor de imagens [B, C, H, W] normalizado em [0, 1]
            labels: Labels verdadeiros [B]
            targeted: Se True, minimiza perda (ataque direcionado)
                     Se False, maximiza perda (não-direcionado)

        Returns:
            adv_images: Imagens adversárias
            metrics: Dicionário com métricas do ataque
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Guarda a imagem original para calcular a perturbação final
        original_images = images.clone()

        # ================================================================
        # PASSO 1: Inicialização aleatória dentro da bola ε
        # ================================================================
        # Começamos com perturbação aleatória para escapar de mínimos locais
        # Isso torna o PGD mais forte que o FGSM iterativo básico
        delta = torch.zeros_like(images).uniform_(-self.epsilon, self.epsilon)
        delta = delta.to(self.device)
        delta.requires_grad = True

        # Métricas para acompanhar o progresso do ataque
        loss_history = []

        # ================================================================
        # PASSO 2: Loop iterativo do PGD
        # ================================================================
        for iteration in range(self.num_iterations):
            # Imagem adversária atual
            adv_images = images + delta

            # Garante que os valores de pixel estejam em [0, 1]
            adv_images = torch.clamp(adv_images, 0, 1)

            # Forward pass: obtém predições do modelo
            outputs = self.model(adv_images)

            # Calcula a perda (queremos MAXIMIZÁ-LA para confundir o modelo)
            loss = self.criterion(outputs, labels)
            loss_history.append(loss.item())

            # ============================================================
            # PASSO 2c: Backpropagation para calcular gradiente
            # ============================================================
            # O gradiente ∇_δ L nos diz como mudar δ para aumentar a perda
            loss.backward()

            # ============================================================
            # PASSO 2d: Atualização da perturbação usando FGSM step
            # ============================================================
            # sign(grad): direção que mais aumenta a perda
            # alpha: magnitude do passo
            #
            # POR QUE SIGN()?
            # - Garante perturbação uniforme por pixel
            # - Maximiza impacto na norma L∞
            # - Mais eficiente que usar gradiente raw
            grad_sign = delta.grad.data.sign()

            if targeted:
                # Ataque direcionado: MINIMIZA perda para classe alvo
                delta.data = delta.data - self.alpha * grad_sign
            else:
                # Ataque não-direcionado: MAXIMIZA perda para classe original
                delta.data = delta.data + self.alpha * grad_sign

            # ============================================================
            # PASSO 2e: Projeção na bola L∞ de raio ε
            # ============================================================
            # Isso garante que ||δ||∞ ≤ ε (perturbação imperceptível)
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)

            # ============================================================
            # PASSO 2f: Garante pixels válidos em [0, 1]
            # ============================================================
            delta.data = torch.clamp(
                images + delta.data, 0, 1
            ) - images

            # Zera gradientes para próxima iteração
            delta.grad.zero_()

        # ================================================================
        # PASSO 3: Gera imagem adversária final
        # ================================================================
        adv_images = torch.clamp(images + delta, 0, 1)

        # Calcula métricas finais
        metrics = self._compute_metrics(
            original_images, adv_images, labels, loss_history
        )

        return adv_images.detach(), metrics

    def _compute_metrics(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor,
        labels: torch.Tensor,
        loss_history: list
    ) -> Dict:
        """
        Calcula métricas detalhadas do ataque.

        Returns:
            Dict contendo:
                - original_probs: probabilidades originais por classe
                - adversarial_probs: probabilidades após ataque
                - original_pred: predição original
                - adversarial_pred: predição após ataque
                - perturbation_linf: norma L∞ da perturbação
                - perturbation_l2: norma L2 da perturbação
                - attack_success: se o ataque mudou a predição
                - loss_history: evolução da perda durante iterações
        """
        with torch.no_grad():
            # Predições originais
            orig_outputs = self.model(original)
            orig_probs = F.softmax(orig_outputs, dim=1)
            orig_pred = orig_outputs.argmax(dim=1)

            # Predições adversárias
            adv_outputs = self.model(adversarial)
            adv_probs = F.softmax(adv_outputs, dim=1)
            adv_pred = adv_outputs.argmax(dim=1)

            # Perturbação
            perturbation = adversarial - original
            linf_norm = perturbation.abs().max().item()
            l2_norm = perturbation.pow(2).sum().sqrt().item()

            # Sucesso do ataque
            attack_success = (orig_pred != adv_pred).float().mean().item()

            return {
                "original_probs": orig_probs.cpu().numpy(),
                "adversarial_probs": adv_probs.cpu().numpy(),
                "original_pred": orig_pred.cpu().numpy(),
                "adversarial_pred": adv_pred.cpu().numpy(),
                "original_confidence": orig_probs.max(dim=1)[0].cpu().numpy(),
                "adversarial_confidence": adv_probs.max(dim=1)[0].cpu().numpy(),
                "perturbation_linf": linf_norm,
                "perturbation_l2": l2_norm,
                "attack_success": attack_success,
                "loss_history": loss_history,
                "true_label": labels.cpu().numpy()
            }

    def get_perturbation_heatmap(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor
    ) -> np.ndarray:
        """
        Gera um heatmap da perturbação para visualização.

        A perturbação é amplificada para ser visível, já que valores
        típicos de ε (8/255) são imperceptíveis ao olho humano.

        Args:
            original: Imagem original [1, C, H, W]
            adversarial: Imagem adversária [1, C, H, W]

        Returns:
            heatmap: Array numpy [H, W] com magnitude da perturbação
        """
        perturbation = (adversarial - original).abs()
        # Soma sobre canais para obter magnitude total
        heatmap = perturbation.sum(dim=1).squeeze().cpu().numpy()
        # Normaliza para [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap


def get_top_k_predictions(
    probs: np.ndarray,
    labels: list,
    k: int = 5
) -> list:
    """
    Retorna as top-k predições com suas probabilidades.

    Args:
        probs: Array de probabilidades [num_classes]
        labels: Lista de nomes das classes
        k: Número de predições a retornar

    Returns:
        Lista de dicts com 'class', 'label' e 'probability'
    """
    top_k_idx = probs.argsort()[-k:][::-1]
    results = []
    for idx in top_k_idx:
        results.append({
            "class_idx": int(idx),
            "label": labels[idx] if idx < len(labels) else f"Class {idx}",
            "probability": float(probs[idx])
        })
    return results
