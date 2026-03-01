import re
import spacy
from spacy.matcher import PhraseMatcher, Matcher

nlp = spacy.load("en_core_web_sm")

phrase_keywords = [
    "battery life",
    "customer service",
    "refund policy",
]

phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
phrase_patterns = [nlp.make_doc(p) for p in phrase_keywords]
phrase_matcher.add("KW_PHRASE", phrase_patterns)

print(phrase_patterns)