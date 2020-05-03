import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# data = []
with open('data/reddit_data_large.json', 'r') as f:
  data = json.loads(f.read())

num_posts = len(data)
index_to_posts_id = {index: post_id for index, post_id in enumerate(data)}
print('created index')
n_feats = 5000
# doc_by_vocab = np.empty([len(data)+1, n_feats])
posts = []
for d in data:
  s = str(data[d]['selftext'])+str(data[d]['title'])
  posts.append(s)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(posts)]
model = Doc2Vec(documents, vector_size=10, window=2, min_count=1, workers=4)
model.save('models/reddit_doc2vec')