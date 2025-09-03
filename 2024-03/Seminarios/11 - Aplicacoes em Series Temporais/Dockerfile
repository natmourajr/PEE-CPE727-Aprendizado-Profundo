# Usar uma imagem base do Python
FROM python:3.9-slim

# Atualizar e instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Definir o diretório de trabalho no container
WORKDIR /app

# Copiar o arquivo requirements.txt para dentro do container
COPY requirements.txt /app/

# Instalar as dependências do Python do requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar o código Python para dentro do container
COPY . /app

# Definir o comando para rodar o script Python
CMD ["python3", "Aplicacao em Series Temporais.ipynb"]