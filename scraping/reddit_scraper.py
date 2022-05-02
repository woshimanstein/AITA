import os
import json
import praw
import prawcore
import tqdm
import argparse
from psaw import PushshiftAPI
import pickle
import datetime as dt

def make_tree(comment):
    tree = {'body': comment.body, 'author': str(comment.author), 'ups': comment.ups}

    if len(comment._replies) > 0:
        tree['replies'] = []
        for reply in comment._replies:
            tree['replies'].append(make_tree(reply))

    return tree


def replace_space_and_illegal_char(s):
    return s.replace(' ', '_') \
        .replace('<', '[less]') \
        .replace('>', '[greater]') \
        .replace(':', '[colon]') \
        .replace('"', '[quote]') \
        .replace('/', '[forward_slash]') \
        .replace('\\', '[backslash]') \
        .replace('|', '[pipe]') \
        .replace('?', '[qmark]') \
        .replace('*', '[asterisk]')


'''
General setup
'''
# setup args
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    '-s', '--subreddit',
    type=str,
    default='AmItheAsshole',
    help='Specify the subreddit you want to scrape'
)

args = arg_parser.parse_args()
os.chdir('../')

'''
Find all posts under a user specified subreddit
'''
api = PushshiftAPI()

after = int(dt.datetime(2012,1,1,0,0).timestamp())
before = int(dt.datetime(2021,12,31,0,0).timestamp())
gen = api.search_submissions(subreddit=args.subreddit, after=after, before=before)

counter = 0
if not os.path.exists(os.path.join('scraping', 'posts')):
    posts = []
    for result in gen:
        if counter < 1000 and counter % 100 == 0:
            print(counter)
        if counter % 1000 == 0:
            print(counter)
        try:
            if result.author != '[deleted]':
                try:
                    posts.append((result.title, result.author, result.selftext, result.full_link))
                except Exception as e:
                    print(e)
                    continue
                counter += 1
        except Exception as e:
            print(e)
            continue
    with open(os.path.join('scraping', 'posts'), 'wb') as f:
        pickle.dump(posts, f)
else:
    with open(os.path.join('scraping', 'posts'), 'rb') as f:
        posts = pickle.load(f)
        counter = len(posts) 

print(f'There are {counter} submissions in total under r/{args.subreddit}. Start scraping individual posts.')

'''
Scraping individual posts
'''
try:
    with open(os.path.join('scraping', 'reddit_credentials.txt')) as r:
        lines = r.readlines()
        client_id, client_secret, user_agent = [line.strip() for line in lines]
except FileNotFoundError:
    print('Please add the credential file under scraping folder.')
    print('See https://praw.readthedocs.io/en/stable/getting_started/quick_start.html#prerequisites for details.')
    print('While creating this file, put client ID, client secret and user agent in 3 separate lines (in this order).')
    exit(0)

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
)

os.makedirs(os.path.join('/', 'data', 'ziyuan', args.subreddit), exist_ok=True)
skipped_links = set()

donwloaded = [int(name.split('_')[0]) for name in os.listdir(os.path.join('/', 'data', 'ziyuan', args.subreddit))]
current_index = max(donwloaded)
for counter in tqdm.tqdm(range(len(posts))):
    if counter > current_index:
        title, author, self_text, link = posts[counter]
        submission = reddit.submission(url=link)

        submissionTree = []

        try:
            if len(submission.comments) == 0:
                continue
        except prawcore.exceptions.NotFound or prawcore.exceptions.ServerError:
            continue

        for top_level_comment in submission.comments:
            try:
                submissionTree.append(make_tree(top_level_comment))
            except AttributeError:
                skipped_links.add(link)
                continue

        if len(submissionTree) > 0:
            js = json.dumps(submissionTree)

            file_title = f'{counter}_{replace_space_and_illegal_char(title)}'
            file_title = file_title if len(file_title) <= 150 else file_title[:150]

            with open(os.path.join('/', 'data', 'ziyuan', args.subreddit, file_title.encode('latin-1', 'replace').decode('latin-1')), 'w', encoding="utf-8") as w:
                w.write(f'{title}\n')
                w.write(f'{author}\n')
                w.write(f'{self_text}\n')
                w.write(f'{js}\n')
                w.write(f'{link}\n')

# record files with deleted responses
with open(os.path.join('/', 'data', 'ziyuan', args.subreddit, 'skipped_posts.txt'), 'w') as skip_w:
    for skipped_link in skipped_links:
        skip_w.write(f'{skipped_link}\n')
print(f'Scraped {counter} posts in total.')
