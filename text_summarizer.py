import requests
import os
import utils
import json
import numpy as np
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
import string 
import re

def mean_case_length(case_json): 
    """
    Returns the average length of cases (in sentences)
    case_json: dict with fields case_name, case_summary, score
    """
    lengths_list = list()
    for case in case_json:
        length_of_text = len(sent_tokenize(case['case_summary']))
        lengths_list.append(length_of_text)
    return np.mean(lengths_list)


def std_dev_case_length(case_json): 
    """
    Returns the std dev of length of cases (in sentences)
    case_json: dict with fields case_name, case_summary, score
    """
    lengths_list = list()
    for case in case_json:
        length_of_text = len(sent_tokenize(case['case_summary']))
        lengths_list.append(length_of_text)
    return np.std(lengths_list)


# =====TEXT SUMMARIZATION METHODS======

def contains_digit(line):
    """
    Returns true if a string contains a digit character, False otherwise
    
    line: a str
    """
    return any(char.isdigit() for char in line)

def contains_punctuation(line):
    """
    Returns true if a string contains a punctuation character, False otherwise
    
    line: a str
    """
    return any(char in string.punctuation for char in line)
    


def create_tf_dict(text_string):
    """
    Returns a term-frequency dict with (term, frequency) key-value pairs
    
    text_string: a str to create the term-freq dict from
    """
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


def create_sentence_scores(sentences, tf_dict, n_chars=10):  
    """
    Returns dict with (sentence, score) key-value pairs
    
    sentences: list of sentences
    tf_dict: term frequency dict mapping words to num occurrences in document
    """
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




def mean_sentence_score(sentence_weight_dict):
    """
    Returns average sentence scores in a document
    
    sentence_weight_dict: dict with (sentence, score) key-value pairs
    """
    # Calculating the average score for the sentences
    sum_weights = 0
    for sentence in sentence_weight_dict:
        sum_weights += sentence_weight_dict[sentence]

    # Getting sentence average value from source text
    average_score = (sum_weights / len(sentence_weight_dict))

    return average_score


def create_summary(sentences, sentence_weight_dict, threshold, n_chars=10):
    """
    Returns a summary using sentences that have sentence scores above the threshold.
    
    sentences: list of sentences from the case
    sentence_weight_dict: dictionary of (sentence, sentence_score) key-value pairs
    threshold: sentence score threshold that determines which sentences to include in the summary
    n_chars: sentences are kept track of in a dict using the first n_chars of the sentence.
    """
   
    article_summary = ''
    num_sentences = len(sentences)
    
    for sentence in sentences:
        
        if sentence[:n_chars] in sentence_weight_dict and sentence_weight_dict[sentence[:n_chars]] >= (threshold):
            article_summary += " " + sentence

    return article_summary


def sigmoid_func(a,b,c,d):
    return a/(1 + np.exp(-b*c)) + d

def case_summary(case_text, multiplier):
    """
    Returns a summary of case_text
    
    case_text: ranked list of dicts with fields case_name <str>, case_summary <str>, and score <float>
    multiplier: float to threshold sentence_scores which determines what sentences are included in the case summary
    """
    
    try:
        # create a tf dictionary
        tf_dictionary = create_tf_dict(case_text)

        # tokenize sentences
        sentences = sent_tokenize(case_text)

        # algorithm for scoring a sentence by its words
        sentence_scores = create_sentence_scores(sentences, tf_dictionary)

        # get the threshold
        mean_sent_score = mean_sentence_score(sentence_scores)
        
        # produce the summary
        case_summary = create_summary(sentences, sentence_scores, multiplier * mean_sent_score)
        return case_summary
    except Exception as e:
        print(repr(e))
        return None
    
def summarize_cases(results, avg, std_dev):
    """
    Returns a list of dicts with fields case_name <str>, case_summary <str>, and score <float> 
    where the case_summary field is a summarized version of the full text of the court case
    
    results: cases relevant to query from CAP API based on the similarity of the cases's full-text to the query
    """
    for case in results:
        # this is actually the full text of the court case
        case_text = case['case_summary']
        # set the full text to a summarized version
        # thresholds what sentences to include in the summary 
        try:
            z_score = (len(sent_tokenize(case_text)) - avg) / std_dev
        except Exception as e:
            print("error caluculating z-score: " + str(repr(e)))
            z_score = 0
        multiplier = sigmoid_func(2,0.5,z_score,0.25)
        case_text_summary = case_summary(case_text, multiplier)
        # clean up the summary
        replacements = {'\btbe\b' : 'the', '\bTbe\b' : 'The', '\bbouse\b' : 'house', "'\b" : "", \
                        '•' : '', '■' : '', '- ' : ' ', '\]' : '', '\[' : ''}
                        
        if not case_text_summary is None:
            for key in replacements:
                case_text_summary = re.sub(r"%s" % key, replacements[key], case_text_summary)
           
        # replace full-text with cleaned up summary
        case['case_summary'] = case_text_summary
    return results

def wrap_summary(results):
    avg = mean_case_length(results)
    std_dev = std_dev_case_length(results)
    return summarize_cases(results, avg, std_dev)
        

