import json
import re
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string

import sys
sys.path.insert(1, '../../..')
import utils


def tokenize(text:str):
    """
    Tokenizer that removes stemmings from tokens. (currently unused)
    """
    trans_table = {ord(c): None for c in string.punctuation + string.digits}    
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in word_tokenize(text.translate(trans_table)) if len(word) > 1]


def rank_cases(query:str, stem_tokens=False):
    """
    Finds cases relevant to query from CAP API based on the similarity of the
    case summary to the query. Cases are then ranked by tfidf cosine similarity
    between the query and the full case texts. Requires unmetered API key.

    Parameters:
    query: the query
    stem_tokens: if True, removes token stemmings before computing tf-idf matrix
    (False by default because it's a bottleneck and it does not affect results much)

    Returns:
     - ranked list of dicts with fields case_name <str>, case_summary <str>, and score <float>
     - None if API request fails
    """

    ## STEP 1: load cases ##
    try:
        # query data
        response = utils.get_request_caselaw("https://api.case.law/v1/cases/?search='{}'&full_case=TRUE".format(query)).json()
        cases = response['results']
        
        i = 1 # limit to 2 requests (200 cases) because that should be more than enough
        while response['next'] and i < 2: 
            response = utils.get_request_caselaw(response['next']).json()
            cases.extend(response['results'])
            i += 1
    except Exception:
        print("API request failed")
        return None


    ## STEP 2: pre-processing ##

    for case in cases:
        # get rid of non-ok cases
        if case['casebody']['status'] != 'ok':
            cases.remove(case)
            continue

    # enforce case ordering
    case_names = [case['name'] for case in cases]
    case_texts = [case['casebody']['data']['head_matter'].replace("\n", " ") for case in cases]
        
    # case_opinions = []
    # for opinions in [case['casebody']['data']['opinions'] for case in cases]:
    #     for opinion in opinions:
    #         if opinion['type'] == "majority":
    #             case_opinions.append(opinion['text'].replace("\n", " "))
    #             break

    ## STEP 3: assign similarity scores to cases ##

    # compute tf-idf scores
    if stem_tokens:
        stemmer = PorterStemmer()
        vec = TfidfVectorizer(tokenizer=tokenize, 
                        min_df=.01, 
                        max_df=0.8, 
                        max_features=5000, 
                        stop_words=[stemmer.stem(item) for item in ENGLISH_STOP_WORDS], 
                        norm='l2')
    else:
        vec = TfidfVectorizer(min_df=.01, 
                        max_df=0.8, 
                        max_features=5000, 
                        stop_words='english', 
                        norm='l2')

    tfidf_matrix = vec.fit_transform(case_texts + [query]).toarray()

    # compute cosine similarity of cases to search query
    query_vec = tfidf_matrix[-1]
    scores = [cosine_similarity(query_vec.reshape(1,-1), doc_vec.reshape(1,-1))[0][0] for doc_vec in tfidf_matrix[:-1]]



    ## STEP 4: sort and return cases ##

    # TODO: currently just returns full case, need to find a way to summarize case
    results = pd.DataFrame(list(zip(case_names, case_texts, scores)), columns=['case_name', 'case_summary', 'score'])
    results = results.sort_values('score', ascending=False).reset_index(drop=True)
    
    return results.to_dict('records')


if __name__ == "__main__":
    with open('output.json', 'w') as f:
        json.dump(rank_cases("employer fired me for age discrimination"), f)