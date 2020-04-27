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
import requests

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


def get_court_jurisdictions():
    """
    Uses CourtListener's court API to generate a mapping of courts to
    their corresponding jurisdictions. This takes a few seconds to run
    and does not depend on the query, so the output should be pre-loaded.

    Returns:
    dict where keys are federal / state court names <str> and values are
    either "trial" or "appeal" <str>
    """
    # fetch response
    response = requests.get("https://www.courtlistener.com/api/rest/v3/courts/?format=json").json()
    courts = response['results']

    while response['next']: 
        response = requests.get(response['next']).json()
        courts.extend(response['results'])

    # create mapping
    jurisdictions = {
        "F"   : "appeal",
        "FD"  : "trial",
        "FB"  : "trial",
        "FBP" : "appeal",
        "FS"  : "appeal", # unsure
        "S"   : "appeal",
        "SA"  : "appeal",
        "ST"  : "trial",
        "SS"  : "trial",  # unsure
        "SAG" : "trial",  # unsure
        "C"   : "trial",  # unsure
        "I"   : "trial",  # unsure
        "T"   : "trial"   # unsure
    }
    # vast majority of cases are not in 'unsure' categories so its no big deal

    return {court['full_name']: jurisdictions[court['jurisdiction']] for court in courts}


def rank_cases(query:str, stem_tokens=False, jurisdiction='', earlydate = ''):
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

    ## ==== <STEP 1: load cases> ==== ##

    debug_message = ''
    try:
        # query data
        url = "https://api.case.law/v1/cases/?search='{}'&full_case=TRUE".format(query)
        if jurisdiction == 'all':
            jurisdiction = ''
        else:
            url = url + '&jurisdiction=' + str(jurisdiction)
        if earlydate and len(earlydate) > 0:
            url = url + '&decision_date_min=' + str(earlydate)
        response = utils.get_request_caselaw(url).json()
        cases = response['results']
        
        i = 1 # limit to 5 requests (500 cases) because that should be more than enough
        while response['next'] and i < 5: 
            response = utils.get_request_caselaw(response['next']).json()
            cases.extend(response['results'])
            i += 1
    except Exception:
        print("API request failed")
        debug_message = 'The Case Law API is currently down. Sorry for the inconvenience!'
        return (None, debug_message)
    
    ## ==== <STEP 2: pre-processing> ==== ##

    for case in cases:
        # get rid of non-ok cases
        if case['casebody']['status'] != 'ok':
            cases.remove(case)
            continue

    case_names = [case['name'] for case in cases]
    case_texts = [case['casebody']['data']['head_matter'].replace("\n", " ") for case in cases]
    case_urls = [case['frontend_url'] for case in cases]

    case_opinions = []
    for opinions in [case['casebody']['data']['opinions'] for case in cases]:
        for opinion in opinions:
            if opinion['type'] == "majority":
                case_opinions.append(opinion['text'].replace("\n", " "))
                break

    case_jurisdictions = []
    # load static info on court type
    with open('data/court_types.json', 'r') as f:
        court_types = json.load(f)
    courts_included = set(court_types.keys())

    for court1 in [case['court']['name'] for case in cases]:
        # find closest court name from CourtListener court api, then obtain its jurisdiction (trial/appeal/etc)
        closest_court = ""

        # look for pure match first
        if court1 in courts_included:
            closest_court = court1
        else:
            # otherwise, use jaccard
            max_score = 0
            court1_tokens = set(court1.split())
            for court2 in courts_included:
                court2_tokens = set(court2.split())
                score = float(len(court1_tokens & court2_tokens)) / len(court1_tokens | court2_tokens)
                if score > max_score:
                    max_score = score
                    closest_court = court2
    
        case_jurisdictions.append(court_types[closest_court])    

    ## ==== <STEP 3: determine opinion of cases> ==== ##
    # TODO: check time to run step 3 -- if it's too slow, only compute opinions for top cases

    case_outcomes = []

    trial_regex = re.compile("(verdict|judgment)[ \w]* (plaintiff|defendant)", re.IGNORECASE)

    for case_index in range(len(cases)):
        match_not_found = False
        if case_jurisdictions[case_index] == "trial":
            result_text = re.search(trial_regex, case_texts[case_index])
            # try first regex on trial cases
            if result_text:
                match = result_text.group()
                if "against" in match:
                    case_outcomes.append("plaintiff" if match[-9:] == "defendant" else "defendant")
                else:
                    case_outcomes.append(match[-9:])
            else:
                result_opinion = re.search(trial_regex, case_opinions[case_index])
                if result_opinion:
                    match = result_opinion.group()
                    if "against" in match:
                        case_outcomes.append("plaintiff" if match[-9:] == "defendant" else "defendant")
                    else:
                        case_outcomes.append(match[-9:])
                else:
                    # opinions mentioning 'dismissed' near the end are often pro-defendant
                    idx = case_opinions[case_index].rfind("dismissed")
                    if idx != -1 and (len(case_opinions[case_index]) - idx) / len(case_opinions[case_index]) < 0.2:
                        # threshold at 0.2 to verify 'dismissed' is said near the end
                        case_outcomes.append("defendant")
                    else:
                        # try appellate regex logic in case of misclassification
                        match_not_found = True
        
        if case_jurisdictions[case_index] == "appeal" or match_not_found:
            # TODO: find original verdict (using above logic) and then look for appeal keywords
            case_outcomes.append("unknown")

    ## ==== <STEP 4: rank cases by similarity to query> ==== ##

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
    try:
        # compute cosine similarity of cases to search query
        tfidf_matrix = vec.fit_transform(case_texts + [query]).toarray()
        query_vec = tfidf_matrix[-1]
        scores = [cosine_similarity(query_vec.reshape(1,-1), doc_vec.reshape(1,-1))[0][0] for doc_vec in tfidf_matrix[:-1]]

        ## ==== <STEP 5: sort and return cases> ==== ##

        results = pd.DataFrame(list(zip(case_names, case_texts, case_urls, scores)), columns=['case_name', 'case_summary', 'case_url', 'score'])
        results = results.sort_values('score', ascending=False).reset_index(drop=True)
        
        return (results.to_dict('records'), debug_message)
    except:
        debug_message = 'No cases found. Please enter a new query or try a wider date range!'
        return (None, debug_message)

    

if __name__ == "__main__":
    with open('output.json', 'w') as f:
        json.dump(rank_cases("fence built on my property"), f)