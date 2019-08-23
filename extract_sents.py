import codecs
import os

from bs4 import BeautifulSoup


def read_corpus(directory):
    output = codecs.open(directory+'/paraphrase.xml', 'w', 'utf-8')
    for file in os.listdir(directory):
        with codecs.open(os.path.join(directory, file)) as xml:
            soup = BeautifulSoup(xml, 'lxml')
            pairs = soup.find_all('pair')
            for pair in pairs:
                if pair['entailment'] == 'Paraphrase':
                    output.write(str(pair))
    output.close()


def extract_sentences(file):
    output_t = codecs.open(os.path.splitext(file)[0]+'.sent_t', 'w', 'utf-8')
    output_h = codecs.open(os.path.splitext(file)[0]+'.sent_h', 'w', 'utf-8')
    with codecs.open(file, 'r', 'utf-8') as xml:
        soup = BeautifulSoup(xml, 'lxml')
        pairs = soup.find_all('pair')
        for pair in pairs:
            if pair['entailment'] == 'Paraphrase':
                output_t.write(pair.findChildren()[0].text+'\n')
                output_h.write(pair.findChildren()[1].text+'\n')
                # for child in pair.findChildren():
                #     output.write(child.text+'\n')
    output_t.close()
    output_h.close()


if __name__ == '__main__':
    # read_corpus('corpus/dev')
    extract_sentences('corpus/test/assin-ptbr-test.xml')
