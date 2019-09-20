import codecs
import logging

import numpy as np
from gensim import corpora
from gensim.models import KeyedVectors, LsiModel
from nltk import tokenize, RSLPStemmer
from nltk.corpus import stopwords
from scipy import spatial
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class ExtractFeatures:

    def __init__(self, model):
        self.model = KeyedVectors.load_word2vec_format('models/'+model+'.txt')

    def load_files(self, input_h, input_t,):
        self.input_h = self.read_file(input_h)
        self.input_t = self.read_file(input_t)
        self.freq = self.load_frequencies()

    @staticmethod
    def read_file(input_f):
        return codecs.open(input_f, 'r', 'utf-8').readlines()

    @staticmethod
    def load_frequencies():
        freq = {}
        with codecs.open('frequency/pt_br_full.txt', 'r', 'utf-8') as input:
            for line in input.readlines():
                freq[str(line.split()[0])] = int(line.split()[1])
        return freq

    @staticmethod
    def preprocess(snt_h, snt_t):
        tokens_h = tokenize.word_tokenize(snt_h, language='portuguese')
        tokens_t = tokenize.word_tokenize(snt_t, language='portuguese')
        return [t.lower() for t in tokens_h if t not in stopwords.words(u'portuguese')], \
               [t.lower() for t in tokens_t if t not in stopwords.words(u'portuguese')]

    @staticmethod
    def stemmer(snt_h, snt_t):
        stemmer = RSLPStemmer()
        stemmed_h = [stemmer.stem(word) for word in snt_h]
        stemmed_t = [stemmer.stem(word) for word in snt_t]
        return stemmed_h, stemmed_t

    @staticmethod
    def _prepare_corpus(snt_h, snt_t):
        dictionary_h = corpora.Dictionary(snt_h)
        dictionary_t = corpora.Dictionary(snt_t)
        doc_term_matrix_h = [dictionary_h.doc2bow(doc) for doc in snt_h]
        doc_term_matrix_t = [dictionary_t.doc2bow(doc) for doc in snt_t]
        # print(doc_term_matrix_h, doc_term_matrix_t)
        return dictionary_h, dictionary_t, doc_term_matrix_h, doc_term_matrix_t

    def create_gensim_lsa_model(self, snt_h, snt_t, number_of_topics=5):
        dictionary_h, dictionary_t, doc_term_matrix_h, doc_term_matrix_t = self._prepare_corpus(snt_h, snt_t)
        lsa_model_h = LsiModel(doc_term_matrix_h, number_of_topics, id2word=dictionary_h)
        lsa_model_t = LsiModel(doc_term_matrix_t, number_of_topics, id2word=dictionary_t)
        return lsa_model_h, lsa_model_t

    @staticmethod
    def remove_first_pc(x):
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(x)
        pc = svd.components_
        xx = x - x.dot(pc.transpose()) * pc
        return xx

    def get_tokens_in_vocab(self, tokens_h, tokens_t):
        tokens_h = [token for token in tokens_h if token in self.model.vocab]
        tokens_t = [token for token in tokens_t if token in self.model.vocab]
        return tokens_h, tokens_t

    def smooth_inverse_frequency(self, tokens_h, tokens_t):
        a = 0.001
        total_freq = sum(self.freq.values())
        embeddings = []

        weight_h = [a / (a + self.freq.get(token, 0) / total_freq) for token in tokens_h]
        weight_t = [a / (a + self.freq.get(token, 0) / total_freq) for token in tokens_t]
        embedding_h = np.average([self.model[token] for token in tokens_h], axis=0, weights=weight_h)
        embedding_t = np.average([self.model[token] for token in tokens_t], axis=0, weights=weight_t)

        embeddings.append(embedding_h)
        embeddings.append(embedding_t)
        embeddings = self.remove_first_pc(np.array(embeddings))

        sims = [cosine_similarity(embeddings[idx * 2].reshape(1, -1), embeddings[idx * 2 + 1].reshape(1, -1))[0][0]
                for idx in range(int(len(embeddings) / 2))]
        return sims[0]

    def word_move_distance(self, tokens_h, tokens_t):
        return self.model.wmdistance(tokens_h, tokens_t)

    def cosine_distance_embeddings(self, tokens_h, tokens_t):
        vector_h = np.mean([self.model[token] for token in tokens_h], axis=0)
        vector_t = np.mean([self.model[token] for token in tokens_t], axis=0)
        return 1 - spatial.distance.cosine(vector_h, vector_t)

    def extract_features(self, set, div, model):
        with codecs.open('features/'+model+'/'+set+'/features-'+div+'.txt', 'w', 'utf-8') as features:
            for snt_h, snt_t in zip(self.input_h, self.input_t):
                tokens_h, tokens_t = self.preprocess(snt_h, snt_t)
                features.write(str(self.word_move_distance(tokens_h, tokens_t)) + '\t')
                tokens_h, tokens_t = self.get_tokens_in_vocab(tokens_h, tokens_t)
                features.write(str(self.smooth_inverse_frequency(tokens_h, tokens_t)) + '\t')
                features.write(str(self.cosine_distance_embeddings(tokens_h, tokens_t)) + '\t')
                if div == 'enta' or div == 'none':
                    features.write(str(2) + '\n')
                else:
                    features.write(str(1) + '\n')


if __name__ == '__main__':

    # model_skip100 = KeyedVectors.load_word2vec_format('models/skip_s100.txt')
    divs = ['enta', 'none', 'para']
    sets = ['dev', 'test', 'train']
    models = ['fasttext50', 'fasttext100', 'glove50', 'glove100', 'glove300', 'skip50', 'skip100', 'skip300']
    # model = 'models/skip_s300.txt'
    logging.info('Loading sentence and pre-trained model')
    for model in models:
        features = ExtractFeatures(model)
        logging.info('Done!!')
        for s in sets:
            for d in divs:
                input_h = 'corpus/'+s+'/assin-ptpt-'+s+'-'+d+'.sent_h'
                input_t = 'corpus/'+s+'/assin-ptpt-'+s+'-'+d+'.sent_t'
                features.load_files(input_h, input_t)
                logging.info('Extracting features')
                features.extract_features(s, d, model)
                logging.info('Done!!')

# model_glove100 = KeyedVectors.load_word2vec_format('models/glove_s50.txt')

# 1.2787 skip 50    2.5670 glove 50
# 1.4351 skip 100   3.2143 glove 100
# 1.9723 skip 300   4.1235 glove 300
    # snt_none1 = 'André Gomes entra em campo quatro meses depois de uma lesão na perna esquerda o ter afastado dos relvados.'.lower().split()
    # snt_none2 = 'Relembre-se que o atleta estava afastado dos relvados desde maio, altura em que contraiu uma lesão na perna esquerda.'.lower().split()

# 1.0534 skip 50    2.4712 glove 50
# 1.2409 skip 100   3.0036 glove 100
# 1.6262 skip 300   4.0117 glove 300
    # snt_enta1 = 'Deolinda Rodrigues morreu este sábado, ao final do dia, no Hospital de Santa Maria.'.lower().split()
    # snt_enta2 = 'Deolinda Rodrigues estava internada no hospital de Santa Maria e morreu por volta das 19h00 deste sábado.'.lower().split()

# 0.9309 skip 50    2.1061 glve 50
# 1.0662 skip 100   1.6231 glove 100
# 1.411 skip 300    3.6377 glove 300

    # snt_para1 = 'Ver tantas pessoas a dedicar tanto tempo, esforço e amor em fazer música fez-me chegar às lágrimas.'.lower()
    # snt_para2 = 'Ver tantas pessoas juntas, o tempo e esforço que dispensaram para fazer algo do género, o amor em fazer música trouxe-me lágrimas.'.lower()

    # tokenize1 = tokenize.word_tokenize(snt_para1, language='portuguese')
    # tokens1 = [t for t in tokenize1 if t not in stopwords.words(u'portuguese')]
    # tokenize2 = tokenize.word_tokenize(snt_para2, language='portuguese')
    # tokens2 = [t for t in tokenize2 if t not in stopwords.words(u'portuguese')]

    # vector1 = np.mean([model_skip100[word] for word in tokens1], axis=0)
    # print(vector1)
    # vector2 = np.mean([model_skip100[word] for word in tokens2], axis=0)
    # print(vector2)
    # cosine = cosine_similarity(vector1, vector2)
    # result = 1 - spatial.distance.cosine(vector1, vector2)
    # print(result)
    # print(cosine)

# distance_none = model_skip100.wmdistance(snt_none1, snt_none2)
# distance_enta = model_skip100.wmdistance(snt_enta1, snt_enta2)
# distance_para = model_skip100.wmdistance(snt_para1, snt_para2)

# print(distance_none)
# print(distance_enta)
# print(distance_para)

# distance_none = model_glove100.wmdistance(snt_none1, snt_none2)
# distance_enta = model_glove100.wmdistance(snt_enta1, snt_enta2)
# distance_para = model_glove100.wmdistance(snt_para1, snt_para2)

# print(distance_none)
# print(distance_enta)
# print(distance_para)