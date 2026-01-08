# RobustnessLab

Ferramenta de diagnóstico para análise de robustez adversária de modelos de visão computacional e proteção de texto contra OCR.

## Visão Geral

O RobustnessLab oferece duas funcionalidades principais:

1. **Ataques Adversários em Classificadores**: Testa a vulnerabilidade de redes neurais profundas usando o algoritmo PGD
2. **Proteção Anti-OCR**: Torna texto em imagens legível apenas para humanos, ilegível para sistemas OCR

## Funcionalidades

### 1. Ataques PGD contra Classificadores

Implementa o algoritmo **PGD (Projected Gradient Descent)** para gerar perturbações adversárias:

```
δ_{t+1} = Π_ε(δ_t + α · sign(∇_x L(θ, x + δ_t, y)))
```

### 2. Proteção Anti-OCR (NOVO)

Sistema de proteção que mantém texto legível para humanos mas ilegível para máquinas:

**10 Técnicas de Proteção:**
| Técnica | Descrição |
|---------|-----------|
| Ruído Adversarial | Ruído multi-escala otimizado para confundir CNNs |
| Padrão Estruturado | Interferência senoidal que afeta segmentação |
| Perturbação FFT | Ruído no domínio da frequência em bandas médias |
| Distorção Geométrica | Micro-ondulações que quebram alinhamento |
| Disrupção de Bordas | Ruído seletivo nas bordas dos caracteres |
| Shift de Canais | Aberração cromática artificial |
| Textura Adversarial | Padrões Gabor que criam features falsas |
| Micro Padrões | Checkerboard em nível de pixel |
| Mascaramento de Gradiente | Componentes com derivadas problemáticas |
| Ruído Dithering | Padrão Bayer similar a impressão |

**4 Níveis de Proteção:**
| Nível | Eficácia Estimada | Visibilidade |
|-------|-------------------|--------------|
| Baixo | 30-50% | Imperceptível |
| Médio | 50-70% | Muito sutil |
| Alto | 70-85% | Sutil |
| Máximo | 85-95% | Possíveis artefatos |

## Estrutura do Projeto

```
RobustnessLab/
├── backend/
│   ├── main.py              # API FastAPI
│   ├── pgd_attack.py        # Implementação do PGD
│   ├── text_protection.py   # Sistema Anti-OCR
│   ├── ocr_models.py        # Integração OCR
│   ├── models.py            # Carregamento de modelos
│   └── requirements.txt     # Dependências Python
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── TextProtection.jsx  # Interface Anti-OCR
│       │   └── ...outros componentes
│       ├── App.jsx
│       └── main.jsx
└── README.md
```

## Instalação

### Backend (Python)

```bash
cd RobustnessLab/backend

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente (Windows)
venv\Scripts\activate

# Ativar ambiente (Linux/Mac)
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# (Opcional) Instalar engines OCR para teste
pip install pytesseract easyocr
```

### Frontend (Node.js)

```bash
cd RobustnessLab/frontend
npm install
```

## Execução

### 1. Iniciar Backend

```bash
cd RobustnessLab/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Iniciar Frontend

```bash
cd RobustnessLab/frontend
npm run dev
```

Acesse: `http://localhost:3000`

## API Endpoints

### Classificadores (PGD)

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/models` | Lista modelos disponíveis |
| POST | `/attack` | Executa ataque PGD |
| POST | `/predict` | Predição sem ataque |

### Proteção Anti-OCR

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/text-protection/levels` | Níveis de proteção |
| GET | `/text-protection/techniques` | Técnicas disponíveis |
| POST | `/text-protection/protect` | Protege imagem com texto |
| POST | `/text-protection/create` | Cria texto protegido |
| POST | `/text-protection/test-ocr` | Testa OCR em imagem |
| GET | `/text-protection/ocr-engines` | Engines OCR disponíveis |

### Exemplos de Request

**Proteger imagem existente:**
```bash
curl -X POST "http://localhost:8000/text-protection/protect" \
  -F "file=@documento.png" \
  -F "protection_level=high" \
  -F "techniques=all"
```

**Criar texto protegido:**
```bash
curl -X POST "http://localhost:8000/text-protection/create" \
  -F "text=Informação Confidencial" \
  -F "protection_level=maximum" \
  -F "font_size=40"
```

## Como Funciona a Proteção Anti-OCR

### Fundamento Científico

Sistemas OCR (Tesseract, EasyOCR, Google Vision, etc.) dependem de:
- Detecção de bordas bem definidas
- Padrões de alta frequência
- Conectividade de caracteres
- Análise de contraste local

A proteção funciona adicionando perturbações estruturadas que:
1. **Interferem com detecção de bordas** sem afetar percepção humana global
2. **Adicionam ruído em frequências específicas** que confundem CNNs
3. **Quebram conectividade** em nível de pixel
4. **Criam features falsas** que enganam redes neurais

### Por que humanos ainda conseguem ler?

O sistema visual humano:
- Processa informação em nível global, não local
- É robusto a ruído estruturado de baixa magnitude
- Usa contexto semântico para interpretação
- Tolera distorções geométricas sutis

## Casos de Uso Legítimos

- **Proteção de documentos** contra scraping automático
- **CAPTCHAs leves** que não irritam usuários
- **Watermarks invisíveis** para rastreamento
- **Proteção de informações sensíveis** em screenshots
- **Pesquisa em robustez** de sistemas OCR

## Modelos de Classificação Suportados

| Modelo | Input | Accuracy | Robustez |
|--------|-------|----------|----------|
| ResNet-50 | 224×224 | 76.1% | Média |
| InceptionV3 | 299×299 | 77.3% | Alta |
| VGG-16 | 224×224 | 71.6% | Baixa |
| MobileNetV2 | 224×224 | 71.9% | Baixa |

## Tecnologias

### Backend
- FastAPI
- PyTorch / Torchvision
- OpenCV / SciPy
- NumPy / Pillow
- Matplotlib

### Frontend
- React 18
- Vite
- Tailwind CSS
- Recharts
- Lucide Icons
- Axios

### OCR (Opcional)
- Tesseract
- EasyOCR
- PaddleOCR

## Referências

1. Madry, A., et al. (2017). "Towards Deep Learning Models Resistant to Adversarial Attacks"
2. Goodfellow, I., et al. (2014). "Explaining and Harnessing Adversarial Examples"
3. Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks"
4. Akhtar, N., & Mian, A. (2018). "Threat of Adversarial Attacks on Deep Learning in Computer Vision"

## Considerações Éticas

Esta ferramenta é destinada **exclusivamente** para:
- Pesquisa em robustez de IA
- Teste de modelos próprios
- Educação em segurança de ML
- Proteção defensiva de informações

**Não use** para atacar sistemas de terceiros sem autorização.

## Licença

MIT License - Use responsavelmente para fins de pesquisa.
