import contractions
import glob
import nltk
import re, unicodedata

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

def denoise_data(strip_html_tags=True, extract_contractions=True):
    for filepath in glob.glob('TR-mails/TR-processed/*'):
        with open(filepath, 'rb') as infile:
            content = infile.read()

            if strip_html_tags:
                content = BeautifulSoup(content, 'html.parser').text

            if extract_contractions:
                content = contractions.fix(content)

        outfile = open(filepath, 'w+')
        outfile.write(content)

def normalize_data():
    for filepath in glob.glob('TR-mails/TR-processed/*'):
        
        with open(filepath, 'r') as infile:
            content = infile.read()

        # content = content.decode('utf-8')

        content = content.replace('\n', ' ')
        content = content.replace('\r', ' ')

        words = nltk.word_tokenize(content)    

        # Remove long words

        new_words = []

        for word in words:
            if len(word) <= 20:
                new_words.append(word)

        words = new_words

        # Remove Non-ASCII words

        new_words = []

        for word in words:
            new_word = unicodedata.normalize('NFKD',  word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        
        words = new_words

        # Convert to lowercase

        new_words = []

        for word in words:
            new_word = word.lower()
            new_words.append(new_word)

        words = new_words

        # Remove punctuations

        new_words = []

        for word in words:
            new_word = re.sub(r'[^\w\s]', ' ', word)
            if new_word != '' or new_word != ' ' or new_word != '  ':
                new_words.append(new_word.strip())

        words = new_words

        # Remove numbers

        new_words = []

        for word in words:
            new_word = re.sub(r'\d+', '', word)
            if new_word != '':
                new_words.append(new_word)

        words = new_words

        # Remove stopwords

        new_words = []

        for word in words:
            if word not in nltk.corpus.stopwords.words('english'):
                new_words.append(word)

        words = new_words

        # Lemmatize

        new_words = []

        lemmatizer = WordNetLemmatizer()
        lemmas = []

        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)

        new_lemmas = [x.strip() for x in lemmas]

        lemmas = [x for x in new_lemmas if x != '']

        filename = filepath.split('/')[2]
        new_path = 'TR-mails/TR-lemmatized/' + filename

        outfile = open(new_path, 'w+')

        str_lemmas = ' '.join(str(x) for x in lemmas)
        outfile.write(str_lemmas)

denoise_data(strip_html_tags=True, extract_contractions=True)
normalize_data()




