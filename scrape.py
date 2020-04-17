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

get_all_subreddit_posts()