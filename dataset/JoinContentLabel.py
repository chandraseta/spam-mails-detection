import glob
import pandas as pd

dataframe = pd.read_csv('spam-mail.tr.label', sep=',')
predictions = dataframe['Prediction'].values

lemmas = []

for idx in range(1, 2501):
    with open('TR-mails/TR-lemmatized/TRAIN_{}.eml'.format(idx)) as emlfile:
        lemma = emlfile.read()

    lemmas.append(lemma)

dataframe = pd.DataFrame({'predictions': predictions, 'lemmas': lemmas})

dataframe.to_csv('preprocessed-result.csv', sep=',', index=False)