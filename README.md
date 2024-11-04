# YouTube Transcriber

O **YouTube Transcriber** é uma aplicação que permite transcrever vídeos do YouTube em texto. Utiliza o modelo Whisper da OpenAI para realizar a transcrição e exibe o progresso do processamento em tempo real.

## Estrutura do Projeto

youtube-transcriber/ ├── pycache/ ├── .gitignore ├── app.py ├── downloads/ ├── main.py ├── README.md ├── requirements.txt ├── src/ │ ├── pycache/ │ ├── messages.py │ ├── transcriber.py ├── static/ │ ├── css/ │ │ ├── components/ │ │ │ ├── form.css │ │ │ ├── list.css │ │ │ ├── status.css │ │ ├── main.css ├── templates/ │ ├── index.html ├── transcricoes/ │ ├── [transcrições geradas] ├── venv/ ├── Lib/ ├── pyvenv.cfg ├── Scripts/ ├── share/


## Instalação

1. Clone o repositório:
    ```sh
    git clone https://github.com/seu-usuario/youtube-transcriber.git
    cd youtube-transcriber
    ```

2. Crie e ative um ambiente virtual:
    ```sh
    python -m venv venv
    source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
    ```

3. Instale as dependências:
    ```sh
    pip install -r requirements.txt
    ```

4. Configure o ambiente:
    - Certifique-se de ter o `ffmpeg` instalado e disponível no PATH do sistema.

## Uso

1. Inicie a aplicação:
    ```sh
    python app.py
    ```

2. Acesse a aplicação no navegador:
    ```
    http://127.0.0.1:5000
    ```

3. Insira a URL do vídeo do YouTube que deseja transcrever e acompanhe o progresso.

## Estrutura dos Arquivos

- `app.py`: Contém a lógica principal do servidor Flask.
- `main.py`: Script de exemplo para processamento de vídeo.
- `src/transcriber.py`: Contém funções para baixar vídeos, extrair áudio e transcrever.
- `src/messages.py`: Contém mensagens de progresso e promocionais.
- `templates/index.html`: Interface web para interação com o usuário.
- `static/css/`: Contém os arquivos CSS para estilização da interface.

## Contribuição

1. Faça um fork do projeto.
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`).
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`).
4. Faça push para a branch (`git push origin feature/nova-feature`).
5. Abra um Pull Request.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.