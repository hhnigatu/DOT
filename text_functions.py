import re
import string
import cv2
import pytesseract 
import tqdm

# !pip install nltk
import nltk
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')

def get_text(image_path):
    text=[]
    for image in tqdm.tqdm(image_path):
        img=cv2.imread(image)
        text.append(pytesseract.image_to_string(img))
    return text

def clean_text(text):
    text=re.sub('[PAD]', '', text)
    text=re.sub('[CLS]', '', text)
    text=re.sub('[SEP]', '', text)
    text=re.sub(r'[^\w\s]', '' ,text)
    return text


def get_cosine_sim(text1, text2, stoplist= stopwords.words('english')):
     
    tf_idf= TfidfVectorizer(stop_words=stoplist, ngram_range=(5,5))
    ngrams=tf_idf.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(ngrams, ngrams)
    return cosine_sim
