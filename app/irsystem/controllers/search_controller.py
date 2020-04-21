import math
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import os
# print(os.getcwd())
from app.irsystem.controllers.case_ranking import rank_cases

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


@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    print(query)
    output_message = ''
    if not query:
        res = []
        output_message = ''
        print('no query')
        return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=res)
    else:
        # =====Reddit cos processing START=========
        with open("app/irsystem/controllers/legaladvicesample.json") as f:
            data = json.loads(f.readlines()[0])
        # title, id, selftext, url, created_utc e60m7
        print('loaded reddit data')
        num_posts = len(data)
        index_to_posts_id = {index: post_id for index,
                             post_id in enumerate(data)}
        print('created index')
        n_feats = 5000
        # doc_by_vocab = np.empty([len(data)+1, n_feats])
        print('initialize numpy array')
        tfidf_vec = build_vectorizer(n_feats)
        print("initialize vectorizer")

        # d_array = [str(data[d]['selftext'])+str(data[d]['title']) for d in data]
        d_array = []
        for d in data:
            s = str(data[d]['selftext'])+str(data[d]['title'])
            d_array.append(s)

        print("built d_array")
        d_array.append(query)
        print("concatenated text and query")
        fit_vec = tfidf_vec.fit_transform(d_array)
        print('fit_transform')
        print(fit_vec)
        doc_by_vocab = np.array(fit_vec.todense())
        print('to array')
        sim_posts = []
        for post_index in range(num_posts):
            score = get_sim(doc_by_vocab[post_index], doc_by_vocab[num_posts])
            sim_posts.append((score, post_index))
        print('calculated similarities')
        sim_posts.sort(key=lambda x: x[0], reverse=True)
        print('sorted similarities')
        res = []
        for k in range(10):
            res.append(data[index_to_posts_id[sim_posts[k][1]]])
        print('added results')
        # =====Reddit cos processing END=========
        print('retrieved reddit cases')
        # =====CaseLaw Retrieval=====
        print('begin caselaw retrieval')
        caselaw = rank_cases(query)
        caseresults = caselaw[0:5]
        print(len(caselaw))
        # =====Processing results================
        print('completed caselaw retrieval')
        for i in range(3):
            post = res[i]
            if len(post['selftext']) > 500:
                post['selftext'] = post['selftext'][0:500] + '...'
        # output_message_1 = "Your search: " + query
        # output_message_2 = "Here's what other people have experienced:"
        # if(len(res) >= 3):
        #     output_message_2 = 'Here are the top 3 related cases'
        # else:
        #     output_message_2 = 'Here are the top {n:.0f} related cases'.format(
        #         n=len(res))

        # output_message = output_message_1+' \n '+output_message_2
        caselaw_message = "Historical precedences on '" + query + "':"
        output_message = "Past discussions on '" + query + "':"
        print('rendering template..')
        return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=res[:3], casedata=caseresults, caselaw_message=caselaw_message)
