import contractions
import glob

from bs4 import BeautifulSoup

def clean_data(strip_html_tags=True, extract_contractions=True):
    for filepath in glob.glob('TR-mails/TR-processed/*'):
        new_lines = []

        infile = open(filepath, 'rb')
        for line in infile:
            if strip_html_tags:
                line = BeautifulSoup(line, 'html.parser').text
            new_lines.append(line)

        outfile = open(filepath, 'w+')
        for line in new_lines:
            if line != '\n':
                if extract_contractions:
                    line = contractions.fix(line)
                outfile.write(line)

clean_data()