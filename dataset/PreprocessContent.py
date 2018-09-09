import contractions
import glob
import nltk

from bs4 import BeautifulSoup

def denoise_data(strip_html_tags=True, extract_contractions=True):
    for filepath in glob.glob('TR-mails/TR-processed/*'):
        with open(filepath, 'rb') as infile:
            content = infile.read()

            if strip_html_tags:
                content = BeautifulSoup(content, 'html.parser').text

            if extract_contractions:
                content = contractions.fix(content)

        outfile = open(filepath, 'w+')
        outfile.write(str(content))

def normalize_data():
    for filepath in glob.glob('TR-mails/TR-processed/*'):
        string = ''
        
        infile = open(filepath, 'rb')
        for line in infile:
            string += line
            string += ' '

        words = nltk.word_tokenize(string)
        print(words)


denoise_data(strip_html_tags=False, extract_contractions=False)
# normalize_data()




