import nltk
from typing import List, Optional
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessário do NLTK
nltk.download('punkt')
nltk.download('stopwords')

def get_stop_words(language_code: str = 'pt') -> List[str]:
    """
    Obtém lista de stop words para o idioma especificado.
    
    Args:
        language_code: Código do idioma ('pt', 'en', 'es', etc)
        
    Returns:
        Lista de stop words
    """
    LANGUAGE_MAP = {
        'pt': 'portuguese',
        'en': 'english',
        'es': 'spanish',
        'fr': 'french',
    }
    
    try:
        language = LANGUAGE_MAP.get(language_code, 'portuguese')
        return stopwords.words(language)
    except Exception as e:
        print(f"Erro ao carregar stop words: {e}")
        return []

def extract_keywords(
    text: str,
    language_code: str = 'pt',
    method: str = 'tfidf',
    max_keywords: int = 20,
    word2vec_model: Optional[Word2Vec] = None,
    min_word_length: int = 3,
) -> List[str]:
    """
    Extrai palavras-chave do texto usando diferentes métodos.
    
    Args:
        text: Texto para extrair palavras-chave
        language_code: Código do idioma
        method: Método de extração ('tfidf' ou 'frequency')
        max_keywords: Número máximo de palavras-chave
        word2vec_model: Modelo Word2Vec opcional para filtragem
        min_word_length: Comprimento mínimo das palavras
        
    Returns:
        Lista de palavras-chave
    """
    if not text:
        return []

    try:
        # Obtém stop words
        stop_words = get_stop_words(language_code)
        
        # Tokenização e limpeza básica
        tokens = word_tokenize(text.lower())
        tokens = [
            token for token in tokens 
            if len(token) >= min_word_length
            and token.isalnum()
            and token not in stop_words
        ]
        
        if method == 'tfidf':
            # Extração usando TF-IDF
            vectorizer = TfidfVectorizer(
                stop_words=stop_words,
                max_features=max_keywords
            )
            try:
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                keywords = list(feature_names)
            except Exception:
                keywords = tokens[:max_keywords]
                
        elif method == 'frequency':
            # Extração por frequência
            from collections import Counter
            word_freq = Counter(tokens)
            keywords = [word for word, _ in word_freq.most_common(max_keywords)]
            
        else:
            raise ValueError(f"Método desconhecido: {method}")
        
        # Filtra usando Word2Vec se disponível
        if word2vec_model is not None:
            keywords = [
                word for word in keywords 
                if word in word2vec_model.wv
            ]
        
        return keywords[:max_keywords]
        
    except Exception as e:
        print(f"Erro ao extrair palavras-chave: {e}")
        return []