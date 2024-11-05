# YouTube Transcriber

O **YouTube Transcriber** é uma aplicação que permite transcrever vídeos do YouTube em texto, utilizando modelos de transcrição baseados em inteligência artificial. A aplicação é construída com Flask para o backend, e uma interface web para facilitar o uso.

## Funcionalidades

- **Transcrição de Vídeos do YouTube**: Converte o áudio dos vídeos em texto.
- **Detecção de Idioma**: Detecta automaticamente o idioma do áudio.
- **Integração com Obsidian**: Cria notas no Obsidian com as transcrições.
- **Interface Web**: Interface amigável para facilitar o uso.
- **CLI**: Interface de linha de comando para usuários avançados.
- **Notificações e Progresso**: Exibe o progresso da transcrição e notifica o usuário.

## Tecnologias Utilizadas

- **Backend**: Flask, yt-dlp, ffmpeg-python, PyTorch, Whisper
- **Frontend**: HTML, CSS, JavaScript
- **Integração**: Obsidian, Word2Vec
- **Outros**: Rich, TQDM, Requests, Numpy

## Instalação

### Pré-requisitos

- Python 3.8+
- FFmpeg
- Obsidian (opcional, para integração)

### Passos

1. Clone o repositório:
    ```bash
    git clone https://github.com/tatyquebralayout/youtube-transcriber
    cd youtube-transcriber
    ```

2. Crie um ambiente virtual e ative-o:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
    ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4. Configure o caminho do vault do Obsidian no arquivo `main.py`:
    ```python
    VAULT_PATH = r"F:\pasta estudos\Code Brain"
    ```

5. Execute a aplicação:
    ```bash
    python app.py
    ```

## Uso

### Interface Web

1. Acesse `http://localhost:5000` no seu navegador.
2. Insira a URL do vídeo do YouTube e clique em "Transcrever".
3. Acompanhe o progresso e baixe a transcrição quando estiver pronta.

### Interface de Linha de Comando (CLI)

1. Execute o script `main.py`:
    ```bash
    python main.py
    ```
2. Siga as instruções no terminal para inserir a URL do vídeo e selecionar o modelo de transcrição.

## Estrutura do Projeto

- `app.py`: Arquivo principal da aplicação Flask.
- `main.py`: Interface de linha de comando.
- `model_manager.py`: Gerencia o carregamento e inferência do modelo de transcrição.
- `transcriber.py`: Classe principal para transcrição de vídeos.
- `obsidian_integration.py`: Integração com o Obsidian.
- `youtube_audio_downloader.py`: Baixa e converte o áudio dos vídeos do YouTube.
- `messages.py`: Gerencia mensagens de status e progresso.
- `static/`: Arquivos estáticos (CSS, JS).
- `templates/`: Templates HTML.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Agradecimentos

Agradecemos a todos os contribuidores e às bibliotecas de código aberto que tornam este projeto possível.

---

Desenvolvido por [Tatiana Barros](https://github.com/tatyquebralayout)