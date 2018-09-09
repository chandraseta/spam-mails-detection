import glob
import nltk
import os
import re

from bs4 import BeautifulSoup

for filepath in glob.glob('TR-mails/TR-processed/*'):
    new_lines = []

    infile = open(filepath, 'rb')
    for line in infile:
        cleantext = BeautifulSoup(line, 'html.parser').text
        new_lines.append(cleantext)

    outfile = open(filepath, 'w+')
    for line in new_lines:
        if line != '\n':
            outfile.write(line)