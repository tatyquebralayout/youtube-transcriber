from flask import Flask, render_template, request, jsonify, send_file
from src.transcriber import process_youtube_video
from src.messages import get_random_message  # Importando do módulo messages
import os
from datetime import datetime
import threading
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)

# Define os diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRICOES_DIR = os.path.join(BASE_DIR, 'transcricoes')
DOWNLOADS_DIR = os.path.join(BASE_DIR, 'downloads')

# Dicionário para armazenar o status das transcrições
transcription_status = {}

def get_video_id(url):
    """Extrai o ID do vídeo da URL do YouTube"""
    try:
        parsed_url = urlparse(url)
        if parsed_url.netloc == 'youtu.be':
            return parsed_url.path[1:]
        if parsed_url.netloc in ['www.youtube.com', 'youtube.com']:
            return parse_qs(parsed_url.query)['v'][0]
    except:
        return None

def process_video_task(url, video_id):
    """Processa o vídeo em background e atualiza o status"""
    try:
        # Iniciando (0-10%)
        transcription_status[video_id] = {
            'status': 'starting',
            'message': 'Iniciando processamento...',
            'progress': 5,
            'fun_message': get_random_message(5)
        }

        # Download (11-30%)
        transcription_status[video_id] = {
            'status': 'downloading',
            'message': 'Baixando vídeo do YouTube...',
            'progress': 20,
            'fun_message': get_random_message(20)
        }

        # Processamento (31-50%)
        transcription_status[video_id] = {
            'status': 'processing',
            'message': 'Processando áudio...',
            'progress': 40,
            'fun_message': get_random_message(40)
        }

        # Transcrição (51-80%)
        transcription_status[video_id] = {
            'status': 'transcribing',
            'message': 'Transcrevendo áudio...',
            'progress': 60,
            'fun_message': get_random_message(60)
        }

        success = process_youtube_video(
            url,
            output_dir=TRANSCRICOES_DIR,
            downloads_dir=DOWNLOADS_DIR
        )

        if success:
            # Conclusão (100%)
            transcription_status[video_id] = {
                'status': 'completed',
                'message': 'Transcrição concluída!',
                'progress': 100,
                'fun_message': get_random_message(100)
            }
        else:
            # Erro
            transcription_status[video_id] = {
                'status': 'error',
                'message': 'Erro ao processar o vídeo',
                'progress': -1,
                'fun_message': get_random_message(-1)
            }

    except Exception as e:
        transcription_status[video_id] = {
            'status': 'error',
            'message': f'Erro: {str(e)}',
            'progress': -1,
            'fun_message': get_random_message(-1)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    url = request.json.get('url', '')
    
    if not url:
        return jsonify({'error': 'URL não fornecida'}), 400

    video_id = get_video_id(url)
    if not video_id:
        return jsonify({'error': 'URL do YouTube inválida'}), 400

    thread = threading.Thread(target=process_video_task, args=(url, video_id))
    thread.start()

    return jsonify({
        'status': 'processing',
        'message': 'Processamento iniciado',
        'video_id': video_id
    })

@app.route('/status/<video_id>')
def status(video_id):
    return jsonify(transcription_status.get(video_id, {
        'status': 'not_found',
        'message': 'Transcrição não encontrada',
        'progress': 0,
        'fun_message': 'Transcrição não encontrada 😕'
    }))

@app.route('/transcricoes')
def list_transcricoes():
    try:
        files = []
        for file in os.listdir(TRANSCRICOES_DIR):
            if file.endswith('_transcricao.txt'):
                file_path = os.path.join(TRANSCRICOES_DIR, file)
                files.append({
                    'name': file,
                    'date': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
        return jsonify({'files': sorted(files, key=lambda x: x['date'], reverse=True)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    try:
        return send_file(
            os.path.join(TRANSCRICOES_DIR, filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    os.makedirs(TRANSCRICOES_DIR, exist_ok=True)
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    app.run(debug=True)