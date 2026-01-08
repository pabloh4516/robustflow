# Deploy Gratuito do RobustnessLab

Guia para fazer deploy usando servicos gratuitos.

**Custo: R$ 0 (com limites generosos)**

---

## Arquitetura

```
[Vercel - Frontend]  <-->  [Railway - Backend]
   (React/Vite)              (FastAPI/Python)
      GRATIS                    GRATIS
```

---

## Parte 1: Deploy do Backend no Railway

Railway oferece $5 de credito gratis por mes (suficiente para projetos pequenos).

### 1.1 Criar Conta no Railway

1. Acesse [railway.app](https://railway.app)
2. Clique em **Login** > **Login with GitHub**
3. Autorize o acesso

### 1.2 Preparar o Backend

Primeiro, crie os arquivos necessarios na pasta `backend`:

**Arquivo: `backend/Procfile`**
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Arquivo: `backend/runtime.txt`**
```
python-3.11.0
```

**Arquivo: `backend/nixpacks.toml`**
```toml
[phases.setup]
nixPkgs = ["python311", "tesseract"]

[phases.install]
cmds = ["pip install -r requirements.txt"]

[start]
cmd = "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
```

### 1.3 Subir para o GitHub

Se ainda nao tem o projeto no GitHub:

```bash
cd C:\Users\55319\RobustnessLab
git init
git add .
git commit -m "Initial commit"
```

Crie um repositorio no GitHub e faca push:

```bash
git remote add origin https://github.com/SEU_USUARIO/RobustnessLab.git
git branch -M main
git push -u origin main
```

### 1.4 Deploy no Railway

1. No Railway, clique em **New Project**
2. Selecione **Deploy from GitHub repo**
3. Escolha o repositorio `RobustnessLab`
4. Railway vai detectar automaticamente
5. Clique em **Add variables** e adicione:
   - Nao precisa de variaveis por enquanto

6. Vá em **Settings** > **Networking** > **Generate Domain**
7. Anote a URL gerada (ex: `robustnesslab-backend.up.railway.app`)

### 1.5 Configurar Root Directory (Importante!)

1. No projeto Railway, va em **Settings**
2. Em **Root Directory**, coloque: `backend`
3. Clique em **Redeploy**

Aguarde o deploy terminar. Teste acessando:
```
https://SUA-URL.up.railway.app/docs
```

---

## Parte 2: Deploy do Frontend na Vercel

### 2.1 Criar Conta na Vercel

1. Acesse [vercel.com](https://vercel.com)
2. Clique em **Sign Up** > **Continue with GitHub**
3. Autorize o acesso

### 2.2 Configurar URL do Backend

Crie o arquivo de producao:

**Arquivo: `frontend/.env.production`**
```
VITE_API_URL=https://SUA-URL-DO-RAILWAY.up.railway.app
```

Faca commit e push:

```bash
git add .
git commit -m "Add production config"
git push
```

### 2.3 Deploy na Vercel

1. Na Vercel, clique em **Add New** > **Project**
2. Selecione o repositorio `RobustnessLab`
3. Configure:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

4. Em **Environment Variables**, adicione:
   - `VITE_API_URL` = `https://SUA-URL-DO-RAILWAY.up.railway.app`

5. Clique em **Deploy**

Aguarde o deploy terminar. Sua URL sera algo como:
```
https://robustnesslab.vercel.app
```

---

## Parte 3: Configurar CORS no Backend

Edite `backend/main.py` para permitir requisicoes da Vercel:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://robustnesslab.vercel.app",  # Adicione sua URL da Vercel
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Faca commit e push - o Railway vai fazer redeploy automaticamente.

---

## Parte 4: Configurar Headers para FFmpeg.wasm

O FFmpeg.wasm precisa de headers especiais. Crie o arquivo:

**Arquivo: `frontend/vercel.json`**
```json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "Cross-Origin-Opener-Policy",
          "value": "same-origin"
        },
        {
          "key": "Cross-Origin-Embedder-Policy",
          "value": "require-corp"
        }
      ]
    }
  ]
}
```

Faca commit e push.

---

## Resumo das URLs

Apos o deploy, voce tera:

| Servico | URL | Funcao |
|---------|-----|--------|
| Frontend | `https://robustnesslab.vercel.app` | Interface do usuario |
| Backend | `https://xxx.up.railway.app` | API Python |
| API Docs | `https://xxx.up.railway.app/docs` | Documentacao Swagger |

---

## Limites dos Planos Gratuitos

### Railway (Backend)
- $5 de credito/mes (suficiente para ~500 horas)
- 512 MB RAM
- Dorme apos inatividade (demora alguns segundos para acordar)

### Vercel (Frontend)
- 100 GB de bandwidth/mes
- Ilimitado para projetos pessoais
- Builds ilimitados

---

## Alternativas Gratuitas

Se Railway nao funcionar bem, alternativas para o backend:

### Render.com
1. Crie conta em [render.com](https://render.com)
2. New > Web Service
3. Conecte o GitHub
4. Root Directory: `backend`
5. Build Command: `pip install -r requirements.txt`
6. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Fly.io
```bash
# Instalar flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Deploy
cd backend
flyctl launch
flyctl deploy
```

---

## Solucao de Problemas

### Erro de CORS
Verifique se a URL da Vercel esta na lista de `allow_origins` no backend.

### Backend nao inicia
Verifique os logs no Railway:
1. Clique no servico
2. Va em **Deployments**
3. Clique no deploy mais recente
4. Veja os logs

### FFmpeg.wasm nao funciona
Verifique se o `vercel.json` esta correto e foi feito deploy.

### Timeout no processamento
O plano gratuito tem limite de tempo. Tente videos menores.

---

## Atualizando o Projeto

Sempre que fizer mudancas:

```bash
git add .
git commit -m "Descricao da mudanca"
git push
```

Tanto Railway quanto Vercel vao fazer deploy automaticamente!

---

## Dominio Personalizado (Opcional)

### Na Vercel:
1. Va em **Settings** > **Domains**
2. Adicione seu dominio
3. Configure o DNS conforme instrucoes

### No Railway:
1. Va em **Settings** > **Networking**
2. Adicione Custom Domain
3. Configure o DNS conforme instrucoes

---

## Custos se Precisar Escalar

| Servico | Plano Gratis | Plano Pago |
|---------|--------------|------------|
| Vercel | Suficiente | $20/mes (Pro) |
| Railway | $5 credito | $5+ por uso |
| Render | 750h/mes | $7/mes |

Para a maioria dos usos pessoais, o plano gratuito e suficiente!
