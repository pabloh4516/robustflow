# Deploy do RobustnessLab na Hostinger

Guia completo para fazer deploy do projeto na Hostinger usando VPS.

---

## Requisitos

- Conta na Hostinger com plano VPS (recomendado: VPS 2 ou superior)
- Dominio configurado (opcional, pode usar IP)
- Conhecimento basico de terminal Linux

---

## Parte 1: Preparar o VPS na Hostinger

### 1.1 Contratar VPS

1. Acesse [hostinger.com.br](https://www.hostinger.com.br)
2. Va em **VPS Hosting**
3. Escolha o plano **VPS 2** ou superior (minimo 2GB RAM para o backend com PyTorch)
4. Selecione **Ubuntu 22.04** como sistema operacional
5. Finalize a compra

### 1.2 Acessar o Painel VPS

1. Acesse o **hPanel** da Hostinger
2. Va em **VPS** > Seu servidor
3. Anote o **IP do servidor** e **senha root**

### 1.3 Conectar via SSH

No Windows, use o PowerShell ou Terminal:

```bash
ssh root@SEU_IP_DO_VPS
```

Digite a senha quando solicitado.

---

## Parte 2: Configurar o Servidor

### 2.1 Atualizar o Sistema

```bash
apt update && apt upgrade -y
```

### 2.2 Instalar Dependencias

```bash
# Instalar Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Instalar Python 3.11
apt install -y python3.11 python3.11-venv python3-pip

# Instalar Nginx e outras ferramentas
apt install -y nginx git certbot python3-certbot-nginx

# Instalar Tesseract OCR (para funcionalidade OCR)
apt install -y tesseract-ocr tesseract-ocr-por
```

### 2.3 Criar Usuario para a Aplicacao

```bash
# Criar usuario
adduser robustnesslab
usermod -aG sudo robustnesslab

# Mudar para o usuario
su - robustnesslab
```

---

## Parte 3: Enviar os Arquivos do Projeto

### Opcao A: Via Git (Recomendado)

Se o projeto estiver no GitHub:

```bash
cd /home/robustnesslab
git clone https://github.com/SEU_USUARIO/RobustnessLab.git
cd RobustnessLab
```

### Opcao B: Via SCP (Upload Direto)

No seu computador Windows, abra o terminal na pasta do projeto:

```bash
# Compactar o projeto (no Windows)
tar -czvf robustnesslab.tar.gz RobustnessLab

# Enviar para o servidor
scp robustnesslab.tar.gz root@SEU_IP:/home/robustnesslab/
```

No servidor:

```bash
cd /home/robustnesslab
tar -xzvf robustnesslab.tar.gz
```

### Opcao C: Via FileZilla (Interface Grafica)

1. Baixe o FileZilla: https://filezilla-project.org/
2. Conecte com:
   - Host: `sftp://SEU_IP`
   - Usuario: `root`
   - Senha: sua senha
   - Porta: `22`
3. Arraste a pasta `RobustnessLab` para `/home/robustnesslab/`

---

## Parte 4: Configurar o Backend (FastAPI)

### 4.1 Criar Ambiente Virtual

```bash
cd /home/robustnesslab/RobustnessLab/backend
python3.11 -m venv venv
source venv/bin/activate
```

### 4.2 Instalar Dependencias Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.3 Testar o Backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Acesse `http://SEU_IP:8000/docs` para ver se funciona. Depois pare com `Ctrl+C`.

### 4.4 Criar Servico Systemd

Volte para root:

```bash
exit  # sair do usuario robustnesslab
```

Crie o arquivo de servico:

```bash
nano /etc/systemd/system/robustnesslab-backend.service
```

Cole o conteudo:

```ini
[Unit]
Description=RobustnessLab Backend API
After=network.target

[Service]
User=robustnesslab
Group=robustnesslab
WorkingDirectory=/home/robustnesslab/RobustnessLab/backend
Environment="PATH=/home/robustnesslab/RobustnessLab/backend/venv/bin"
ExecStart=/home/robustnesslab/RobustnessLab/backend/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Salve com `Ctrl+O`, `Enter`, `Ctrl+X`.

Ative o servico:

```bash
systemctl daemon-reload
systemctl enable robustnesslab-backend
systemctl start robustnesslab-backend
systemctl status robustnesslab-backend
```

---

## Parte 5: Configurar o Frontend (React)

### 5.1 Build de Producao

```bash
su - robustnesslab
cd /home/robustnesslab/RobustnessLab/frontend
```

Edite o arquivo de configuracao para apontar para o backend correto:

```bash
nano src/config.js
```

Se nao existir, crie com:

```javascript
export const API_URL = 'https://SEU_DOMINIO/api';
// ou se nao tiver dominio:
// export const API_URL = 'http://SEU_IP:8000';
```

Agora faca o build:

```bash
npm install
npm run build
```

Os arquivos de producao estarao em `/home/robustnesslab/RobustnessLab/frontend/dist`

---

## Parte 6: Configurar Nginx

### 6.1 Criar Configuracao do Site

Volte para root:

```bash
exit
nano /etc/nginx/sites-available/robustnesslab
```

Cole o conteudo (substitua `SEU_DOMINIO` pelo seu dominio ou IP):

```nginx
server {
    listen 80;
    server_name SEU_DOMINIO;  # ou seu IP

    # Frontend - arquivos estaticos
    root /home/robustnesslab/RobustnessLab/frontend/dist;
    index index.html;

    # Servir o frontend
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy para o backend API
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;

        # Aumentar timeout para uploads grandes
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;

        # Aumentar tamanho maximo de upload (100MB)
        client_max_body_size 100M;
    }

    # Headers para permitir SharedArrayBuffer (necessario para FFmpeg.wasm)
    add_header Cross-Origin-Opener-Policy same-origin;
    add_header Cross-Origin-Embedder-Policy require-corp;
}
```

### 6.2 Ativar o Site

```bash
ln -s /etc/nginx/sites-available/robustnesslab /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default  # remover site padrao
nginx -t  # testar configuracao
systemctl restart nginx
```

---

## Parte 7: Configurar SSL (HTTPS)

Se voce tem um dominio configurado:

```bash
certbot --nginx -d SEU_DOMINIO
```

Siga as instrucoes e escolha redirecionar HTTP para HTTPS.

---

## Parte 8: Configurar Firewall

```bash
ufw allow 22      # SSH
ufw allow 80      # HTTP
ufw allow 443     # HTTPS
ufw enable
```

---

## Parte 9: Ajustar o Frontend para Producao

O frontend precisa saber onde esta o backend. Edite o arquivo de chamadas API.

### 9.1 Verificar arquivos que chamam o backend

```bash
grep -r "localhost:8000" /home/robustnesslab/RobustnessLab/frontend/src/
```

### 9.2 Criar arquivo de configuracao de ambiente

```bash
nano /home/robustnesslab/RobustnessLab/frontend/.env.production
```

Conteudo:

```
VITE_API_URL=https://SEU_DOMINIO/api
```

### 9.3 Atualizar chamadas no codigo

Se houver URLs hardcoded como `http://localhost:8000`, substitua por:

```javascript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

### 9.4 Rebuild

```bash
cd /home/robustnesslab/RobustnessLab/frontend
npm run build
systemctl restart nginx
```

---

## Parte 10: Verificar se Esta Funcionando

1. Acesse `http://SEU_DOMINIO` ou `http://SEU_IP`
2. Teste o upload de imagem para ataque adversarial
3. Teste a protecao de texto
4. Teste a protecao de video

---

## Comandos Uteis

### Ver logs do backend

```bash
journalctl -u robustnesslab-backend -f
```

### Reiniciar servicos

```bash
systemctl restart robustnesslab-backend
systemctl restart nginx
```

### Ver status

```bash
systemctl status robustnesslab-backend
systemctl status nginx
```

### Atualizar o projeto

```bash
cd /home/robustnesslab/RobustnessLab
git pull  # se usando git

# Reinstalar dependencias se necessario
cd backend
source venv/bin/activate
pip install -r requirements.txt

cd ../frontend
npm install
npm run build

# Reiniciar
sudo systemctl restart robustnesslab-backend
sudo systemctl restart nginx
```

---

## Solucao de Problemas

### Erro 502 Bad Gateway

O backend nao esta rodando:

```bash
systemctl status robustnesslab-backend
journalctl -u robustnesslab-backend -n 50
```

### Erro de CORS

Verifique se o backend esta configurado para aceitar requisicoes do seu dominio.
Edite `/home/robustnesslab/RobustnessLab/backend/main.py` e ajuste o CORS:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://SEU_DOMINIO"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Video nao processa (SharedArrayBuffer)

O FFmpeg.wasm requer headers especiais. Verifique se o Nginx tem:

```nginx
add_header Cross-Origin-Opener-Policy same-origin;
add_header Cross-Origin-Embedder-Policy require-corp;
```

### Memoria insuficiente

Se o servidor travar ao processar imagens grandes, aumente a RAM do VPS ou adicione swap:

```bash
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

---

## Estrutura Final

```
/home/robustnesslab/RobustnessLab/
├── backend/
│   ├── venv/              # Ambiente virtual Python
│   ├── main.py            # API FastAPI
│   └── ...
├── frontend/
│   ├── dist/              # Build de producao (servido pelo Nginx)
│   ├── src/
│   └── ...
└── DEPLOY_HOSTINGER.md    # Este arquivo
```

---

## Custos Estimados (Hostinger)

| Plano | RAM | Preco/mes | Recomendacao |
|-------|-----|-----------|--------------|
| VPS 1 | 1GB | ~R$20 | Nao recomendado (pouca RAM) |
| VPS 2 | 2GB | ~R$35 | Minimo para funcionar |
| VPS 4 | 4GB | ~R$60 | Recomendado |
| VPS 8 | 8GB | ~R$100 | Ideal para muitos usuarios |

---

## Suporte

Se encontrar problemas, verifique:
1. Logs do backend: `journalctl -u robustnesslab-backend -f`
2. Logs do Nginx: `tail -f /var/log/nginx/error.log`
3. Status dos servicos: `systemctl status robustnesslab-backend nginx`
