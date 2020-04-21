# import praw
import time
import requests
import json
from datetime import datetime
from math import ceil

#Step 1: Get all posts
def get_all_subreddit_posts(subreddit='legaladvice'):
  ops = dict()
  newurl = 'https://api.pushshift.io/reddit/search/submission/?subreddit=' +\
  subreddit + '&sort=asc&size=1000'
  data = requests.get(url = newurl).json().get('data')
  now = (datetime.utcnow() - datetime(1970, 1, 1)).total_seconds()
  while len(data) > 0 and data != None and data[-1].get('created_utc') < now:
    for post in data:
      ops.update({
        post.get('id') : {
          'title': post.get('title'),
          'id': post.get('id'),
          'selftext': post.get('selftext'),
          'url': post.get('url'),
          'created_utc': post.get('created_utc')
        }
      })
    last_timestamp = data[-1].get('created_utc')
    newurl = 'https://api.pushshift.io/reddit/search/submission/?subreddit=' +\
    subreddit + '&sort=asc&after=' + \
    str(last_timestamp) + '&before=' + str(ceil(now)) + '&size=1000'
    data = requests.get(url = newurl).json().get('data')
    # print(newurl)
    # print('timestamp_scraped = ' + str(last_timestamp))
    # print('stopping point = ' + str(now))
    # print('posts = ' + str(len(ops)))
    # with open('subreddit_op_scrape.json', 'w') as f:
    #   f.write(json.dumps(ops))

  with open('subreddit_op_scrape.json', 'w') as f:
      f.write(json.dumps(ops))

def split_data(big_json_path, folder = 'data/', partitions = 20):
  data = None
  with open('data/subreddit_op_scrape.json', 'r') as f:
    data = json.loads(f.read())

  target_size = len(data) // partitions

  split_idx = 0
  while len(data) >= target_size:
    with open('data/legaladvice' + str(split_idx) + '.json', 'w') as f:
      f.write(json.dumps(dict(list(data.items())[:target_size])))
      data = dict(list(data.items())[target_size:])
    print(split_idx)
    split_idx += 1

# split_data('data/subreddit_op_scrape.json', partitions = 15)
# get_all_subreddit_posts()