import spacy
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """Basic text preprocessing: lowercasing and removing extra spaces."""
    return " ".join(text.lower().split())

def extract_syntax_features(doc):
    """Extracts normalized counts of POS tags, dependency relations, tree shape metrics."""
    pos_counts = Counter()
    dep_counts = Counter()
    sentence_lengths = []
    tree_depths = []

    for sent in doc.sents:
        sentence_lengths.append(len(sent))
        tree_depths.append(compute_tree_depth(sent.root))

        for token in sent:
            pos_counts[token.pos_] += 1
            dep_counts[token.dep_] += 1

    features = {}
    N, M = 0, 0
    for pos in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]:
        features[f"pos_count_{pos}"] = pos_counts.get(pos, 0)
        N += features[f"pos_count_{pos}"]

    for dep in ["nsubj", "dobj", "amod", "advmod", "prep", "ROOT"]:
        features[f"dep_count_{dep}"] = dep_counts.get(dep, 0)
        M += features[f"dep_count_{dep}"]
    # Normalize POS counts by total number of tokens
    for pos in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]:
        features[f"pos_count_{pos}_norm"] = features[f"pos_count_{pos}"] / N if N > 0 else 0   
    # Normalize dependency counts by total number of dependencies
    for dep in ["nsubj", "dobj", "amod", "advmod", "prep", "ROOT"]:
        features[f"dep_count_{dep}_norm"] = features[f"dep_count_{dep}"] / M if M > 0 else 0
    # Sentence-level metrics
    features["avg_sent_len"] = np.mean(sentence_lengths) if sentence_lengths else 0
    features["max_tree_depth"] = max(tree_depths) if tree_depths else 0
    features["avg_tree_depth"] = np.mean(tree_depths) if tree_depths else 0

    return features

def compute_tree_depth(token):
    """Recursively computes depth of dependency tree rooted at token."""
    if not list(token.children):
        return 1
    return 1 + max(compute_tree_depth(child) for child in token.children)


class RelationTfidfExtractor:
    def __init__(self):
        self.vectorizers = {}  

    def fit(self, inputs):
        """Fit TF-IDF vectorizers per relation type on the training data."""
        relation_texts = defaultdict(list)
        texts = [input[0] for input in inputs]  # Extract texts from inputs
        for text in texts:
            doc = nlp(preprocess(text))
            for token in doc:
                if not token.is_stop and not token.is_punct and token.text != "'s" and token.text != "'" and token.text != "’s" and len(token.text) > 1:
                    relation_texts[token.dep_].append(token.text)

        for rel, words in relation_texts.items():
            if not words:
                continue
            vec = TfidfVectorizer(max_features=1000, stop_words='english')
            vec.fit([" ".join(words)])
            self.vectorizers[rel] = vec
    ## A revoir, je sais pas si c'est pertinent
    def transform(self, text):
        """Extract relation-wise TF-IDF features for a single text."""
        doc = nlp(preprocess(text))
        rel_words = defaultdict(list)

        for token in doc:
            rel_words[token.dep_].append(token.text)

        features = {}
        for rel, vec in self.vectorizers.items():
            if rel not in rel_words or not rel_words[rel]:
                continue
            words = " ".join(rel_words.get(rel, []))
            tfidf_vec = vec.transform([words]).toarray().flatten()
            for i, val in enumerate(tfidf_vec):
                features[f"tfidf_{rel}_{i}"] = val

        return features
        

def extract_features(inputs, rel_extractor):
    text = np.array([input[0] for input in inputs])
    years = np.array([input[1] for input in inputs])
    all_features = []
    for t in text:
        doc = nlp(preprocess(t))
        feats = {}
        feats.update(extract_syntax_features(doc))
        feats.update(rel_extractor.transform(t))
        all_features.append(feats)
    
    # Ajouter l'année comme feature directement dans chaque dictionnaire
    for i, feat in enumerate(all_features):
        feat['year'] = years[i]
    
    return all_features

