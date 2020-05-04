import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

# data = []
with open('data/reddit_data_large.json', 'r') as f:
  data = json.loads(f.read())

num_posts = len(data)
index_to_posts_id = {index: post_id for index,
                      post_id in enumerate(data)}
n_feats = 5000
# doc_by_vocab = np.empty([len(data)+1, n_feats])
tfidf_vec = TfidfVectorizer(min_df=.01,
                            max_df=0.9,
                            max_features=n_feats,
                            stop_words='english',
                            norm='l2')

# d_array = [str(data[d]['selftext'])+str(data[d]['title']) for d in data]
d_array = []
for d in data:
    s = str(data[d]['selftext'])+str(data[d]['title'])
    d_array.append(s)

# print("built d_array")
# d_array.append(query)
doc_by_vocab = tfidf_vec.fit_transform(d_array).toarray()
pickle.dump(tfidf_vec, open("models/reddit_tfidf_vectorizer.pickle", "wb"))
np.save("models/tfidf_matrix_reddit", doc_by_vocab)

# tokenizer = tfidf_vec.build_tokenizer()
# print(tfidf_vec.transform(['fence built on my property']).toarray())
# print('to array')
# sim_posts = []
# for post_index in range(num_posts):
#     # score = get_sim(doc_by_vocab[post_index], doc_by_vocab[num_posts])
#     q_vector = doc_by_vocab[post_index]
#     post_vector = doc_by_vocab[num_posts]
#     num = q_vector.dot(post_vector)
#     den = np.multiply(np.sqrt(q_vector.dot(q_vector)),
#                       np.sqrt(post_vector.dot(post_vector)))
#     score = num/den
#     sim_posts.append((score, post_index))
# print('calculated similarities')
# sim_posts.sort(key=lambda x: x[0], reverse=True)
# print('sorted similarities')
# res = []
# for k in range(10):
#     e = data[index_to_posts_id[sim_posts[k][1]]]
#     e.update({"score": round(sim_posts[k][0], 3)})
#     res.append(e)

# num_posts = len(data)
# index_to_posts_id = {index: post_id for index, post_id in enumerate(data)}
# print('created index')
# n_feats = 5000
# # doc_by_vocab = np.empty([len(data)+1, n_feats])
# posts = []
# for d in data:
#   s = str(data[d]['selftext'])+str(data[d]['title'])
#   posts.append(s)
# documents = [TaggedDocument(doc, [index_to_posts_id.get(i)]) for i, doc in enumerate(posts)]
# model = Doc2Vec(documents, vector_size=3, window=2, min_count=1, workers=4)
# model.save('models/reddit_doc2vec_vector3')