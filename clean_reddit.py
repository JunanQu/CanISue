import json
with open("../subreddit_op_scrape.json") as f:
    data = json.loads(f.readlines()[0])

fset = set()

for d in data:
    if data[d]['selftext']=='[deleted]':
        fset.add(data[d]['title'])
    
ops = dict()

for d in data:
    if data[d]['title'] not in fset:
        ops.update({
        data[d]['id'] : {
          'title': data[d]['title'],
          'id': data[d]['id'],
          'selftext': data[d]['selftext'],
          'url': data[d]['url'],
          'created_utc': data[d]['created_utc']
        }
      })

with open('../cleaned_reddit_data.json', 'w') as f:
      f.write(json.dumps(ops))