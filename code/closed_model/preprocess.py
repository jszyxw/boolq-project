import re
import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.vocab import GloVe
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

REPLACE_WITH_SPACE = re.compile(r'[^A-Za-z\s()$]')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(x: list) -> list:
    review = ' '.join(x)
    review = REPLACE_WITH_SPACE.sub(' ', review).lower()
    tokens = review.split()
    tokens = list(map(lambda a: lemmatizer.lemmatize(a, 'v'), tokens))
    tokens = list(filter(lambda a: a not in stop_words, tokens))
    return tokens


def get_train_emb(batch_size=128):
    print('start getting data!')
    print('step 1/5 set up fields')
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=False)
    LABEL = data.Field(sequential=False)

    print('step 2/5 get data sets')
    train, test = data.TabularDataset.splits(path='../data/',
                                             train='closed_train.jsonl',
                                             test='closed_dev.jsonl',
                                             format='json',
                                             fields={'QnP': ('text', TEXT),
                                                     'label': ('label', LABEL)})

    print('step 3/5 preprocess data')
    # i = 0
    for x in train.examples:
        x.text = preprocess(x.text)
        # if i < 5:
        #     print(x.text)
        #     print(x.label)
        #     i += 1
    # i = 0
    for x in test.examples:
        x.text = preprocess(x.text)
        # if i < 5:
        #     print(x.text)
        #     print(x.label)
        #     i += 1

    print('step 4/5 build the vocabulary')
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=100))
    LABEL.build_vocab(train)

    print('step 5/5 make iterators')
    train_iter = data.BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.text), device='cpu')
    test_iter = data.BucketIterator(test, batch_size=batch_size, sort_key=lambda x: len(x.text), device='cpu')
    print('finish getting data!')

    return train_iter, test_iter, TEXT, LABEL
