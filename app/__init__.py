# Gevent needed for sockets
from gevent import monkey
monkey.patch_all()

# Imports
import json
import os
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
# Configure app
socketio = SocketIO()
app = Flask(__name__)
app.data = None
#with open('data/reddit_data_big.json', 'r') as f:
#    app.data = json.loads(f.read())
app.data = requests.get('https://storage.googleapis.com/can_i_sue_reddit/reddit_data.json').json()
tfidf_vec = TfidfVectorizer(min_df=.01,
                            max_df=0.8,
                            max_features=5000,
                            stop_words='english',
                            norm='l2')
print("loaded data and initialized vectorizer")
d_array = []
for d in app.data:
    s = str(app.data[d]['selftext'])+str(app.data[d]['title'])
    d_array.append(s)
print("created d_array")
doc_by_vocab = tfidf_vec.fit_transform(d_array).toarray()
print("fit transformed")
app.tfidf_matrix = doc_by_vocab
app.tfidf_vectorizer = tfidf_vec
print("added environment variables")
APP_SETTINGS="config.DevelopmentConfig"
app.config.from_object(os.environ["APP_SETTINGS"])

# DB
db = SQLAlchemy(app)

# Import + Register Blueprints
from app.accounts import accounts as accounts
app.register_blueprint(accounts)
from app.irsystem import irsystem as irsystem
app.register_blueprint(irsystem)

# Initialize app w/SocketIO
socketio.init_app(app)

# HTTP error handling
@app.errorhandler(404)
def not_found(error):
  return render_template("404.html"), 404
