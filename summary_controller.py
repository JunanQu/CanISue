import requests
import os
import utils
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy import linalg as LA
import math
from IPython.display import display, Markdown, Latex
# For text summarization
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
import string 

from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Team Rob's Chili"
net_id = "jq77, zs92, ijp9, mlc294, ns739"


# =====REDDIT COSINE======

# title, id, selftext, url, created_utc e60m7

def build_vectorizer(max_features=5000, stop_words="english", max_df=0.8, min_df=10, norm='l2'):
    tfidf_vec = TfidfVectorizer(stop_words=stop_words, norm=norm,
                                max_df=max_df, min_df=min_df, max_features=max_features)
    return tfidf_vec


def get_sim(q_vector, post_vector):
    num = q_vector.dot(post_vector)
    den = np.multiply(np.sqrt(q_vector.dot(q_vector)),
                      np.sqrt(post_vector.dot(post_vector)))
    return num/den


# =====END=======






# =====TEXT SUMMARIZATION METHODS======

"""
Returns the average length of cases in characters
case_json: dict with fields case_name, case_summary, score
"""
def mean_case_length(case_json): # Not used for prototype
    lengths_list = list()
    for case in case_json:
        length_of_text = len(case['case_summary'])
        lengths_list.append(length_of_text)
    return np.mean(lengths_list)

"""
Returns the std dev of length of cases in characters
case_json: dict with fields case_name, case_summary, score
"""
def std_dev_case_length(case_json): # Not used for prototype
    lengths_list = list()
    for case in case_json:
        length_of_text = len(case['case_summary'])
        lengths_list.append(length_of_text)
    return np.std(lengths_list)


"""
Returns true if a string contains a digit character, False otherwise
line: a str
"""
def contains_digit(line):
    return any(char.isdigit() for char in line)
"""
Returns true if a string contains a punctuation character, False otherwise
line: a str
"""
def contains_punctuation(line):
    return any(char in string.punctuation for char in line)
    

"""
Returns a term-frequency dict with (term, frequency) key-value pairs
text_string: a str to create the term-freq dict from
"""
def create_tf_dict(text_string):
    
    # Remove stop words
    text_string = text_string.lower()
    stop_words = set(stopwords.words("english"))
    
    words = word_tokenize(text_string)
    
    # Reduce words to their root form
    stem = PorterStemmer()
    
    # Create dictionary for the word frequency table
    tf_dict = dict()
    for wd in words:
        # Remove puncutation by turning puncutation to ''
        wd = wd.translate(str.maketrans('', '', string.punctuation))
        # Stem
        wd = stem.stem(wd)
        
        if wd in stop_words or wd == '':
            continue
        if wd in tf_dict:
            tf_dict[wd] += 1
        else:
            tf_dict[wd] = 1
    
    return tf_dict

"""
Returns dict with (sentence, score) key-value pairs
sentences: list of sentences
tf_dict: term frequency dict mapping words to num occurrences in document
"""
def create_sentence_scores(sentences, tf_dict, n_chars=10):   
    sentence_weight_dict = dict()

    for sentence in sentences:
        num_words = (len(word_tokenize(sentence)))
        num_words_minus_stop_words = 0
        first_n_chars = sentence[:n_chars]
        
        for word in tf_dict:
        
            if word in sentence.lower():
                num_words_minus_stop_words += 1
                
                if not (contains_digit(first_n_chars) or contains_punctuation(first_n_chars)):

                    if first_n_chars in sentence_weight_dict:
                        sentence_weight_dict[first_n_chars] += tf_dict[word]
                    else:
                        sentence_weight_dict[first_n_chars] = tf_dict[word]
        
        if not (contains_digit(first_n_chars) or contains_punctuation(first_n_chars)):
            # Additive smoothing to avoid divide by 0
            sentence_weight_dict[first_n_chars] = ((sentence_weight_dict[first_n_chars]+1) / (num_words_minus_stop_words+1))
      
    return sentence_weight_dict



"""
Returns average sentence scores in a document
sentence_weight_dict: dict with (sentence, score) key-value pairs
"""
def mean_sentence_score(sentence_weight_dict):
   
    # Calculating the average score for the sentences
    sum_weights = 0
    for sentence in sentence_weight_dict:
        sum_weights += sentence_weight_dict[sentence]

    # Getting sentence average value from source text
    average_score = (sum_weights / len(sentence_weight_dict))

    return average_score


def create_summary(sentences, sentence_weight, threshold, n_chars=10):
    sentence_counter = 0
    article_summary = ''
    num_sentences = len(sentences)
    
    for sentence in sentences:
        
        if sentence[:n_chars] in sentence_weight and sentence_weight[sentence[:n_chars]] >= (threshold):
            article_summary += " " + sentence
        
        sentence_counter += 1

    return article_summary

def case_summary(case_text, multiplier=1.3):
    
    # creating a tf dictionary
    tf_dictionary = create_tf_dict(case_text)

    # tokenize sentences
    sentences = sent_tokenize(case_text)

    # algorithm for scoring a sentence by its words
    sentence_scores = create_sentence_scores(sentences, tf_dictionary)

    #getting the threshold
    threshold = mean_sentence_score(sentence_scores)

    #producing the summary
    case_summary = create_summary(sentences, sentence_scores, multiplier * threshold)

    return case_summary

# =====END TEXT SUMMARIZATION METHODS=======


@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    print(query)
    output_message = ''
    if not query:
        res = []
        output_message = ''
        return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=res)
    else:
        # =====Reddit cos processing START=========
        with open("app/irsystem/controllers/legaladvicesample.json") as f:
            data = json.loads(f.readlines()[0])
        # title, id, selftext, url, created_utc e60m7
        num_posts = len(data)
        index_to_posts_id = {index: post_id for index,
                             post_id in enumerate(data)}
        n_feats = 5000
        doc_by_vocab = np.empty([len(data), n_feats])
        tfidf_vec = build_vectorizer(n_feats)
        doc_by_vocab = tfidf_vec.fit_transform(
            [str(data[d]['selftext'])+data[d]['title'] for d in data]+[query]).toarray()
        sim_posts = []
        for post_index in range(num_posts):
            score = get_sim(doc_by_vocab[post_index], doc_by_vocab[num_posts])
            sim_posts.append((score, post_index))
        sim_posts.sort(key=lambda x: x[0], reverse=True)
        res = []
        for k in range(10):
            res.append(data[index_to_posts_id[sim_posts[k][1]]])
         # =====Reddit cos processing END=========
        output_message_1 = "Your search: " + query
        if(len(res) >= 3):
            output_message_2 = 'Here are the top 3 related cases'
        else:
            output_message_2 = 'Here are the top {n:.0f} related cases'.format(
                n=len(res))
        
        ### Ian Appended Code for Text Summarizing ###
        case_summaries = list()
        for result in res:
            case_summaries.append(case_summary(result['case_summary'])
        ### End ###
                                  
        output_message = output_message_1+' \n '+output_message_2
        return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=case_summaries[:3]) # Changed display results
