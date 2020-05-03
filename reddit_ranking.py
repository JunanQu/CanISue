from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def rank_reddit(query):
  model = Doc2Vec.load('models/reddit_doc2vec')
  updated_query = model.infer_vector([query])
  sims = model.docvecs.most_similar([updated_query], topn=25)
  indices = list()
  scores = list()
  for pair in sims:
    indices.append(pair[0])
    scores.append(round(pair[1],3))
  return (indices, scores)
print(rank_reddit("fence built on my property"))