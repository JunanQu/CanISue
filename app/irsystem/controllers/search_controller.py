from text_summarizer import wrap_summary
from app.irsystem.controllers.case_ranking import rank_cases
import math
import json
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import os
from flask import Flask, render_template, Response, jsonify
import time
from flask import current_app
# ASYNC and LOADING STUFF
from rq import Queue
from rq.job import Job
from worker import conn
from app import app
import scipy.spatial.distance


q = Queue(connection=conn)
# END
print(os.getcwd())

project_name = "Can I Sue?"
net_id = "Junan Qu (jq77), Zachary Shine (zs92), Ian Paul (ijp9), Max Chen (mlc294), Nikhil Saggi (ns739)"

#r = requests.get(
#    "https://storage.googleapis.com/can_i_sue_reddit/reddit_data.json")
#data = r.json()
data=None
tfidf_vec=None
doc_by_vocab=None
with app.app_context():
    data = current_app.data
    tfidf_vec = current_app.tfidf_vectorizer
    doc_by_vocab = current_app.tfidf_matrix
#doc_by_vocab = []
#doc_by_vocab_flag = False
#tfidf_vec = TfidfVectorizer(min_df=.01,
#                            max_df=0.8,
#                            max_features=5000,
#                            stop_words='english',
#                            norm='l2')
print("loaded reddit info ")
status = 0


def wrap_fun(query, minimum_date, jurisdiction, suing="yes"):
    global status
    global doc_by_vocab
    #global doc_by_vocab_flag
    global tfidf_vec

    output_message = ''
    if not query:
        res = []
        output_message = ''
        print('no query')
        return project_name, net_id, output_message, res

    else:
        # =====Reddit cos processing START=========
        # title, id, selftext, url, created_utc e60m7
        num_posts = len(data)
        index_to_posts_id = {index: post_id for index,
                             post_id in enumerate(data)}

        #if doc_by_vocab_flag == False:
            # d_array = [str(data[d]['selftext'])+str(data[d]['title']) for d in data]
        #    d_array = []
        #    for d in data:
        #        s = str(data[d]['selftext'])+str(data[d]['title'])
        #        d_array.append(s)
        #    doc_by_vocab = tfidf_vec.fit_transform(d_array).toarray()
        #    doc_by_vocab_flag = True

        post_vector = tfidf_vec.transform([query]).toarray()[0]
        start = time.time()
        sims = scipy.spatial.distance.cdist(doc_by_vocab, [post_vector], 'cosine').reshape(-1)
        end = time.time()
        print('Reddit cosine Time elapsed: ', str(end - start))
        # quit()
        sim_posts = []
        for i in range(len(sims)):
            score = sims[i]
            if np.isnan(score):
                score = 0.0
            else:
                score = round(score, 3)
            sim_posts.append((score, 3), i))
        
        print('calculated similarities')
        sim_posts.sort(key=lambda x: x[0], reverse=True)
        print('sorted similarities')

        status = 50

        res = []
        for k in range(10):
            e = data[index_to_posts_id[sim_posts[k][1]]]
            e.update({"score": round(sim_posts[k][0], 3)})
            res.append(e)
        print('added results')
        # =====Reddit cos processing END=========
        print('retrieved reddit cases')
        # =====CaseLaw Retrieval=====
        print('begin caselaw retrieval')

        status = 60
        start = time.time()
        caselaw, debug_msg = rank_cases(
            query, jurisdiction=jurisdiction, earlydate=minimum_date)
        end = time.time()
        print('Case Law Time elapsed: ', str(end - start))
        error = False
        if not caselaw:
            judgment_rec = ""
            # API call to CAP failed
            caseresults = [-1]
            error = True
        else:
            caseresults = caselaw[0:5]
            # Score to keep to 3 decimals
            for case in caseresults:
                case['score'] = round(case['score'], 3)
                case['fulltext'] = case['case_summary']
            # caseresults = wrap_summary(caseresults)
            for case in caseresults:
                if not case['case_summary']:  # if case has no summary
                    case['case_summary'] = "No case summary found"
                    # case['case_summary'] = case['fulltext']
                    continue
                case['case_summary'] = case['case_summary'][0:min(
                    1000, len(case['case_summary']))]
                if len(case['case_summary']) == 1000:
                    case['case_summary'] = case['case_summary'] + '...'

            # calculate judgment score
            judgment_score = 0
            judgment_rec = ""
            score_limit = 0
            confidence = 0
            for case in caselaw:
                score_limit += case['score']
                if case['case_outcome'] == "plaintiff":
                    judgment_score += case['score']
                    confidence += 1
                elif case['case_outcome'] == "defendant":
                    judgment_score -= case['score']
                    confidence += 1
            confidence *= 100/len(caselaw)
            if suing == "no":
                judgment_score *= -1

            if judgment_score >= -score_limit and judgment_score < -score_limit/4:
                judgment_rec = "Verdict: Likely to lose! ({}% confident)".format(
                    confidence)
            elif judgment_score >= -score_limit/4 and judgment_score <= score_limit/4:
                judgment_rec = "Verdict: Could go either way ({}% confident)".format(
                    confidence)
            elif judgment_score > score_limit/4 and judgment_score <= score_limit:
                judgment_rec = "Verdict: Likely to win! ({}% confident)".format(
                    confidence)
            else:
                judgment_rec = ""

            for case in caseresults:
                case['case_outcome'] = case['case_outcome'][0].capitalize() + \
                    case['case_outcome'][1:]

        # =====Processing results================
        print('completed caselaw retrieval')

        status = 70

        for i in range(5):
            post = res[i]
            if (post['selftext'] is not None) and (len(post['selftext'])) > 500:
                post['selftext'] = post['selftext'][0:500] + '...'

        caselaw_message = "Historical precedences:"

        status = 80

        output_message = "Past discussions:"
        print('rendering template..')

        status = 100
        # ============================

        return project_name, net_id, output_message, res[:5], caseresults, caselaw_message, query, debug_msg, judgment_rec, error


@irsystem.route('/about.html')
def go_to_about():
    return render_template('about.html')


@irsystem.route('/', methods=['GET'])
def search():
    return render_template('search.html')


@irsystem.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@irsystem.route("/results/<job_key>", methods=['GET'])
def get_results(job_key):
    job = Job.fetch(job_key, connection=conn)
    if job.is_finished:
        return jsonify(job.result), 200
    else:
        return "Nay!", 202


@irsystem.route('/start', methods=['POST'])
def get_counts():
    data = json.loads(request.data.decode())
    data = data['data']
    print(data)
    query = data[0]
    min_date = data[1]
    state = data[2]
    suing = data[3]

    if min_date is None:
        min_date = ''
    if (state is None) or (state == ""):
        state = 'all'

    job = q.enqueue_call(
        func=wrap_fun, args=(query, min_date,
                             state, suing), result_ttl=5000
    )
    return job.get_id()
