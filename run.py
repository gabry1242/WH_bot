import re
import spacy
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher, Matcher
import pandas as pd


#sentence-transformers/all-MiniLM-L6-v2 
#Possibly look into this to have also different phrase embeddings

df = pd.read_csv("processed_dataset.csv")
df_list = list(df["string"])

nlp = English()

#distinguish between words and phrases
single = [k for k in df_list if " " not in k]
multi  = [k for k in df_list if " " in k]

#phrase case exact finding
matcher_phrase = PhraseMatcher(nlp.vocab, attr="LOWER")
multi_patterns = [nlp.make_doc(p) for p in multi if p]
if multi_patterns:
    matcher_phrase.add("KW_PHRASE", multi_patterns)

#word case
token_matcher = Matcher(nlp.vocab)

#done to match word that can be similar
#e.g word.lower convert the word in lowercase, re.escape escapes special characters so su+re becomes su\+\re
# rf is need to raw format, raw removes the backslash and format allows to use {}
# ^ means that the words starts 
# \w* means zero or more characters
# $ means end of sequence 
def make_regex(word: str) -> str:
    return rf"^{re.escape(word.lower())}\w*$"


for w in single:
    token_matcher.add(f"KW_{w}", [[{"LOWER": {"REGEX": make_regex(w)}}]])


def find_keywords(text: str):
    doc = nlp(text)
    hits = []

    for _, start, end in matcher_phrase(doc):
        hits.append(doc[start:end].text)

    for match_id, start, end in token_matcher(doc):
        hits.append(doc[start:end].text)

    return (1 if hits else 0), sorted(set(hits))

flag, hits = find_keywords("Brother I am really IshowSPEEDING if you know what I mean.")
print(flag, hits)
