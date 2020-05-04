import json
import re
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
import nltk.tokenize
from nltk.stem.porter import PorterStemmer
import string
import requests

import sys
sys.path.insert(1, '../../..')
import utils

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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


def rank_cases(query:str, stem_tokens=False, jurisdiction='', earlydate = '', ncases=10):
    """
    Finds cases relevant to query from CAP API based on the similarity of the
    case summary to the query. Cases are then ranked by tfidf cosine similarity
    between the query and the full case texts. Requires unmetered API key.

    Parameters:
     - query: the query
     - stem_tokens: if True, removes token stemmings before computing tf-idf matrix
       (False by default because it's a bottleneck and it does not affect results much)
     - jurisdiction: if specified, query cases only under certain jurisdiction
     - earlydate: if specified, query cases only after certain date
     - ncases: top number of matches to retrieve from similarity model


    Returns: <tuple> of 
     - ranked list of dicts with relevant case fields (<None> if API fails)
     - debug message indicating error
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
        
        i = 1 # limit to 2 requests (200 cases) because that should be more than enough
        while response['next'] and i < 2: 
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
    
    case_opinions = []
    for opinions in [case['casebody']['data']['opinions'] for case in cases]:
        op_num = 0
        if not opinions:
            # no opinions (never happened yet)
            case_opinions.append(" ")
        for opinion in opinions:
            op_num += 1
            if opinion['type'] == "majority":
                case_opinions.append(opinion['text'].replace("\n", " "))
                break
            if op_num == len(opinions):
                # majority opinion not found (very rare)
                case_opinions.append(opinions[0]['text'].replace("\n", " "))
                break

    case_urls = [case['frontend_url'] for case in cases]

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

    case_outcomes = []

    trial_regex = re.compile("(verdict|judgment)[ \w]* (plaintiff|defendant)", re.IGNORECASE)
    appeal_regex = re.compile("(plaintiff|defendant)s* (in|brings) error", re.IGNORECASE)
    positive_keywords = ['affirmed']
    negative_keywords = ['reversed', 'overruled', 'remanded']
    appeal_outcomes = [""]*len(cases)

    for case_index in range(len(cases)):
        appeal_regex_failed = False
        case_juris = case_jurisdictions[case_index]

        # Part 1: if appeal case, find appeal outcome

        if case_juris == "appeal":
            opinion = case_opinions[case_index].lower()
            if any(word in opinion for word in negative_keywords):
                appeal_outcomes[case_index] = "negative"
            if any(word in opinion for word in positive_keywords):
                if appeal_outcomes[case_index] == "negative":
                    # both postive and negative keywords
                    # try again, but only check last few words
                    opinion = opinion[-200:]
                    appeal_outcomes[case_index] = ""
                    if any(word in opinion for word in negative_keywords):
                        appeal_outcomes[case_index] = "negative"
                    if any(word in opinion for word in positive_keywords):
                        if appeal_outcomes[case_index] == "negative":
                            # found both keywords, unable to classify
                            case_juris = "trial" # evaluate as trial case
                        else:
                            appeal_outcomes[case_index] = "positive"
                else:
                    appeal_outcomes[case_index] = "positive"
            if appeal_outcomes[case_index] == "":
                # found no keywords, unable to classify
                case_juris = "trial" # evaluate as trial case

        # Part 2: find trial case outcome / original appeal case outcome

        # find cases with phrase "verdict/judgement ... plaintiff/defendant"
        result_text = re.search(trial_regex, case_texts[case_index])
        if result_text:
            match = result_text.group()
            if "against" in match:
                case_outcomes.append("plaintiff" if match[-9:].lower() == "defendant" else "defendant")
            else:
                case_outcomes.append(match[-9:].lower())
        else:
            result_opinion = re.search(trial_regex, case_opinions[case_index])
            if result_opinion:
                match = result_opinion.group()
                if "against" in match:
                    case_outcomes.append("plaintiff" if match[-9:].lower() == "defendant" else "defendant")
                else:
                    case_outcomes.append(match[-9:].lower())
            else:
                # find cases with phrase "plaintiff/defendant in/brings error" **for appeal cases only**
                if case_juris == "appeal":
                    result_text = re.search(appeal_regex, case_texts[case_index])
                    if result_text:
                        match = result_text.group()
                        # caveat: making assumption that the appellant was the losing party
                        case_outcomes.append("plaintiff" if match[:9].lower() == "defendant" else "defendant")
                    else:
                        result_opinion = re.search(appeal_regex, case_opinions[case_index])
                        if result_opinion:
                            match = result_opinion.group()
                            # caveat: making assumption that the appellant was the losing party
                            case_outcomes.append("plaintiff" if match[:9].lower() == "defendant" else "defendant")
                        else:
                            appeal_regex_failed = True
                if case_juris == "trial" or appeal_regex_failed:
                    # opinions mentioning 'dismissed' near the end are often pro-defendant
                    idx = case_opinions[case_index].rfind("dismissed")
                    if idx != -1 and (case_juris == 'appeal' or (len(case_opinions[case_index]) - idx) / len(case_opinions[case_index]) < 0.2):
                        # threshold at 0.2 **for trial cases** to verify 'dismissed' is said near the end (last 20% of opinion)
                        case_outcomes.append("defendant")
                    else:
                        case_outcomes.append("unclear")
        
        # Part 3: For appeal cases, adjust original decision based on appeal outcome

        if case_juris == "appeal" and case_outcomes[-1] != "unclear":
            if appeal_outcomes[-1] == "negative":
                case_outcomes[-1] = ("plaintiff" if case_outcomes[-1].lower() == "defendant" else "defendant")

    ## ==== <STEP 4: rank cases by similarity to query> ==== ##

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(case_texts)]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    updated_query = model.infer_vector([query])
    sims = model.docvecs.most_similar([updated_query], topn=ncases)

    ## ==== <STEP 5: summarize top cases> ==== ##

    case_summaries = [""]*len(cases)
    for idx,_ in sims:
        case_sents = {(k[:-1] if k[-1] == "." else k): False 
                       for k in nltk.tokenize.sent_tokenize(case_texts[idx]) + nltk.tokenize.sent_tokenize(case_opinions[idx])}
        important_lines = cases[idx]['preview']
    
        for line in important_lines:
            line = line.replace("<em class='search_highlight'>", "").replace("</em>", "").replace(".", "")
            for sent,_ in case_sents.items():
                if line in sent:
                    case_sents[sent] = True
                    break

        case_summaries[idx] = ". ".join([sent for sent,valid in case_sents.items() if valid])

    ## ==== <STEP 6: return> ==== ##

    out_cases = []
    for i in sims:
        case_dict = dict()
        case_dict.update({
            'case_name'    : case_names[i[0]],
            'case_summary' : case_summaries[i[0]],
            'case_url'     : case_urls[i[0]],
            'case_outcome' : case_outcomes[i[0]],
            'score'        : (i[1] + 1) / 2
        })
        out_cases.append(case_dict)
        
    return (out_cases, debug_message)
        

if __name__ == "__main__":
    rank_cases("fence built on my property")
