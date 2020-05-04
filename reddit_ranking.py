from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

# def rank_reddit(query):
#   model = Doc2Vec.load('models/reddit_doc2vec_vector3')
#   updated_query = model.infer_vector([query])
#   sims = model.docvecs.most_similar([updated_query], topn=25)
#   indices = list()
#   scores = list()
#   for pair in sims:
#     indices.append(pair[0])
#     scores.append((round(pair[1],3) + 1) / 2)
#   return (indices, scores)
# #print(rank_reddit("fence built on my property"))

def rank_reddit(query):
  vectorizer = pickle.load(open("models/reddit_tfidf_vectorizer.pickle", "rb"))
  tfidf = np.load("models/tfidf_matrix_reddit.npy")
  np.savetxt("models/tfidf_matrix_reddit.csv", tfidf, delimiter=",")
  q = vectorizer.transform([query]).toarray()[0]

  sim_posts = []
  for post_index in range(tfidf.shape[0]):
      post_vector = tfidf[post_index]
      num = post_vector.dot(q)
      den = np.multiply(np.sqrt(post_vector.dot(post_vector)),
                        np.sqrt(q.dot(q)))
      try:
        score = num/den
      except:
        score = 0
      sim_posts.append((score, post_index))
  print('calculated similarities')
  sim_posts.sort(key=lambda x: x[0], reverse=True)
  print('sorted similarities')
  res = []
  for k in range(10):
      e = data[index_to_posts_id[sim_posts[k][1]]]
      e.update({"score": round(sim_posts[k][0], 3)})
      res.append(e)
  return res


# import json
# with open('data/reddit_data_large.json', 'r') as f:
#   data = json.loads(f.read())

# rank_reddit('fence built on my property')

# num_posts = len(data)
# index_to_posts_id = {index: post_id for index, post_id in enumerate(data)}

# idx, sim = rank_reddit("neighbor built fence on my property")

# for i in idx:
#   print(i)