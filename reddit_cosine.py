from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import math

query = "Basically my nephew fell out of his chair at a Barnes N' Noble and my dad helped him up, turned around and his Nook was gone. \n\nHe is going to file a police report today but do you think they will actually do anything. Here's what we have:\n\nHe ordered 3 UFC books before my dad was able to get home and take his credit card off/report it as stolen to Barnes N' Noble\n\nThe eBay auction was created 1 day after the Nook was taken; is the only Nook for sale on eBay located within my towns vicinity -- and has my the exact \n\nI have the kids (not really a kid -- he's 22) myspace and facebook profiles that are littered with UFC stuff.\n\nMy dad still has all"

with open("../subreddit_op_scrape.json") as f:
    data = json.loads(f.readlines()[0])

# title, id, selftext, url, created_utc e60m7

num_posts = len(data)
index_to_posts_id = {index:post_id for index, post_id in enumerate(data)}


def build_vectorizer(max_features=5000, stop_words="english", max_df=0.8, min_df=10, norm='l2'):
    tfidf_vec = TfidfVectorizer(stop_words=stop_words, norm=norm, max_df=max_df, min_df=min_df, max_features=max_features)
    return tfidf_vec

n_feats = 5000
doc_by_vocab = np.empty([len(data), n_feats])
tfidf_vec = build_vectorizer(n_feats)

doc_by_vocab = tfidf_vec.fit_transform([str(data[d]['selftext'])+data[d]['title'] for d in data]+[query]).toarray()

def get_sim(q_vector, post_vector):    
    num = q_vector.dot(post_vector)
    den = np.multiply(np.sqrt(q_vector.dot(q_vector)), np.sqrt(post_vector.dot(post_vector))) 
    return num/den

sim_posts = []
for post_index in range(num_posts):
    score = get_sim(doc_by_vocab[post_index], doc_by_vocab[num_posts])
    sim_posts.append((score, post_index))

sim_posts.sort(key=lambda x: x[0], reverse=True)

res = []
for k in range(10):
    res.append(data[index_to_posts_id[sim_posts[k][1]]])

print(res)

