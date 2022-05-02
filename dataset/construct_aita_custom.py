import os
import json
import pandas as pd
import tqdm
import argparse


from label import Label, BinarizedLabel

os.chdir('../')

def clean_scrape(df):
    print("Before cleaning, there are " +  str(len(df)) + " posts.")
    # Remove any edits that may give away the answer [ie, "edit: okay you're right I'm the asshole" ]
    df['body'] = df['body'].str.replace("(edit|update).*?(YTA|a-|ass|\\sta\\s)(.*)","",case=False)
    # Remove any deleted or removed posts
    gone_list = ["[deleted]","[removed]",""]
    df = df[df['body'].isin(gone_list)==False]
    print("After removing deleted posts, there are " +  str(len(df)) + " posts left.")
    # Make a grand binary variable conslidating "no assholes here" and "everyone sucks" into the dominant classes

    return df


'''
extract posts and comments; filter out invalid data points
(1) selftext has content
(2) >= 10 comments
(3) agreement > 90%
'''
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--num_comments',
    type=int,
    default=10
)
arg_parser.add_argument(
    '--agreement',
    type=float,
    default=0.9
)
args = arg_parser.parse_args()

LEAST_NUM_COMMENTS = args.num_comments
AGREEMENT = args.agreement

raw_posts_comments = []
for filename in tqdm.tqdm(list(os.scandir(os.path.join('/', 'data', 'ziyuan', 'AmItheAsshole')))):
    if filename.is_file():
        with open(filename.path) as f:
            try:
                lines = f.readlines()
                title = lines[0].strip()
                selftext = ' '.join(lines[2:-2]).strip()
                comment_tree_list = json.loads(lines[-2].strip())

                # extract verdict
                num_valid_comment = 0
                verdict_count = {label: 0 for label in Label}
                comment_label_ups = []
                for index in range(len(comment_tree_list)):
                    comment = comment_tree_list[index]['body']
                    label = Label.extract_from_text(comment)
                    if label:
                        num_valid_comment += 1
                        verdict_count[label] += 1
                        comment_label_ups.append((comment, label, comment_tree_list[index]['ups']))

                
                # agreement
                if num_valid_comment < LEAST_NUM_COMMENTS:
                    agreement = 0
                else:
                    agreement = max(verdict_count[Label.AUTHOR] / sum(verdict_count.values()), verdict_count[Label.OTHER] / sum(verdict_count.values()))
                    if agreement >= AGREEMENT:
                        if verdict_count[Label.AUTHOR] > verdict_count[Label.OTHER]:
                            is_asshole = 1
                            top_level_comment_ups = {}
                            for comment, label, ups in comment_label_ups:
                                if label == Label.AUTHOR:
                                    top_level_comment_ups[comment] = ups
                            comment_most_up = [k for k, v in sorted(top_level_comment_ups.items(), key=lambda item: item[1])][-1]
                        else:
                            
                            is_asshole = 0
                            top_level_comment_ups = {}
                            for comment, label, ups in comment_label_ups:
                                if label == Label.OTHER:
                                    top_level_comment_ups[comment] = ups
                            comment_most_up = [k for k, v in sorted(top_level_comment_ups.items(), key=lambda item: item[1])][-1]


                # check validity
                if selftext != '[removed]' and num_valid_comment >= LEAST_NUM_COMMENTS and agreement >= AGREEMENT:
                    raw_posts_comments.append([title, selftext, comment_most_up, is_asshole])
            except Exception as e:
                print(e)

posts_comments_df = pd.DataFrame(data=raw_posts_comments, columns=['title', 'body', 'comment', 'is_asshole'])
posts_comments_df = clean_scrape(posts_comments_df)
posts_comments_df.to_csv(os.path.join('data', 'aita', f'aita_custom_agr_{AGREEMENT}_comment_{LEAST_NUM_COMMENTS}.csv'), header=True, index=False)

print(len(raw_posts_comments))