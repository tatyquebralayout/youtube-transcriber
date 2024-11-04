import random

# Mensagens promocionais da Tati
PROMO_MESSAGES = [
    
    "Baixando bits e bytes... 💾",
    "Pegando seu vídeo, não demora... bem, talvez demore um pouquinho 😅",
    "Fazendo download na velocidade da tartaruga... 🐢",
    "Download em andamento... hora de checar o Instagram 📱",
    "Convencendo os pixels a cooperarem... 🎨",

]

# Mensagens para início do processo (0-10%)
STARTING_MESSAGES = [
    "Esquentando os motores... 🚀",
    "Preparando a mágica... ✨",
    "Acordando os hamsters que fazem a transcrição... 🐹",
    "Ligando as engrenagens... ⚙️",
    "Calibrando o transcriptômetro... 📊",
]

# Mensagens para download (11-30%)
DOWNLOADING_MESSAGES = [
    "Baixando bits e bytes... 💾",
    "Pegando seu vídeo, não demora... bem, talvez demore um pouquinho 😅",
    "Fazendo download na velocidade da tartaruga... 🐢",
    "Download em andamento... hora de checar o Instagram 📱",
    "Convencendo os pixels a cooperarem... 🎨",
]

# Mensagens para processamento inicial (31-50%)
PROCESSING_MESSAGES = [
    "Processando... pode ser uma boa hora pra fazer aquela pausa 🎮",
    "Nossos robôs estão ouvindo seu vídeo... 🤖",
    "Preparando o texto... momento perfeito para aquele lanche 🍿",
    "Fase intermediária... já voltamos! 🔄",
    "Convertendo vídeo em texto... mágica pura! ✨",
]

# Mensagens para transcrição (51-80%)
TRANSCRIBING_MESSAGES = [
    "Transformando sons em letras... 📝",
    "Nosso robô está digitando rapidinho... ⌨️",
    "Quase lá! Só mais alguns segundos... ou minutos 😅",
    "Convertendo blablablá em texto... 🗣️",
    "A mágica está acontecendo... 🎩✨",
]

# Mensagens para finalização (81-99%)
FINISHING_MESSAGES = [
    "Aplicando os toques finais... 🎨",
    "Quase acabando, prometo! 🤞",
    "Só mais um minutinho... ⏳",
    "Fazendo a revisão final... 📚",
    "Preparando a entrega... 📦",
]

# Mensagens de conclusão (100%)
COMPLETED_MESSAGES = [
    "Ufa! Finalmente terminamos! 🎉",
    "Missão cumprida! 🎯",
    "Seu texto está prontinho! 📝",
    "Trabalho concluído com sucesso! 🌟",
    "Tudo pronto! Hora de conferir o resultado! 🎈",
]

# Mensagens de erro
ERROR_MESSAGES = [
    "Ops! Algo deu errado... 😅",
    "Houston, temos um problema! 🚀",
    "Parece que os hamsters cansaram... 🐹",
    "Falha na matrix! 🤖",
    "Vamos tentar de novo? 🔄",
]

# Contador para alternar entre mensagens normais e promocionais
message_counter = 0

def get_random_message(progress):
    """Retorna uma mensagem aleatória baseada no progresso"""
    global message_counter
    message_counter += 1
    
    # A cada 2 mensagens, mostra uma mensagem promocional
    if message_counter % 2 == 0:
        return random.choice(PROMO_MESSAGES)
    
    # Mensagens normais baseadas no progresso
    if progress < 10:
        messages = STARTING_MESSAGES
    elif progress < 30:
        messages = DOWNLOADING_MESSAGES
    elif progress < 50:
        messages = PROCESSING_MESSAGES
    elif progress < 80:
        messages = TRANSCRIBING_MESSAGES
    elif progress < 100:
        messages = FINISHING_MESSAGES
    elif progress == 100:
        messages = COMPLETED_MESSAGES
    else:
        messages = ERROR_MESSAGES
    
    return random.choice(messages)