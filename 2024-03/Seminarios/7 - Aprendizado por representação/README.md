# Seminário de Aprendizado por Representação

O seminário é baseado no Capítulo 15 do livro [Dive into Deep Learning](https://d2l.ai/).

## Instruções para executar o projeto

Siga os passos abaixo para configurar e executar o ambiente do seminário de Aprendizado por Representação utilizando Docker.

### 1. Construir a imagem Docker
Execute o comando a seguir para construir a imagem Docker:
```bash
docker build -t image_representation_learning:v1 -f DockefileRepresentationLearning .
```

### 2. Executar o contêiner Docker
Inicie o contêiner com o seguinte comando:
```bash
docker run --name container_representation_learning --detach -p 8888:8888 -it image_representation_learning:v1
```

### 3. Verificar o contêiner
Confira o status do contêiner:
```bash
docker ps -a
```

### 4. Acessar o contêiner
Para acessar o contêiner em execução, utilize:
```bash
docker container exec -u 0 -it container_representation_learning /bin/bash
```

### 5. Executar o Jupyter Notebook
Dentro do contêiner, execute o seguinte comando para iniciar o Jupyter Notebook:
```bash
source /opt/conda/etc/profile.d/conda.sh && jupyter notebook --ip=0.0.0.0 --allow-root --no-browser --port=8888
```

### 6. Acessar o Jupyter Notebook pela web
Abra o navegador e acesse o seguinte link:
```
http://localhost:8888
```

### 7. Abrir outro terminal e acessar o contêiner
Caso precise acessar o contêiner novamente, abra outro terminal e execute:
```bash
docker container exec -u 0 -it container_representation_learning /bin/bash
```

### 8. Listar servidores Jupyter e copiar o token
Dentro do contêiner, liste os servidores Jupyter em execução para obter o token de acesso:
```bash
jupyter server list
```
Copie o token exibido e cole na URL do navegador para acessar o Notebook.