import numpy as np
import heapq as pq
from math import ceil
from collections import defaultdict, namedtuple
from contextlib import closing

import sys
sys.path.append('../coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# -----------------------------------------------------------------------------
# Pre-processing
# -----------------------------------------------------------------------------

def preprocess(x):
  """Pre-process text: remove punctuation, lower case"""
  return x.lower().replace('.',' ').replace(',',' ')

def words(words):
  if isinstance(words,str):
    return words.split()
  else:
    return words

def one_grams(words):
  if isinstance(words,str):
    words = words.split()
  count = defaultdict(int)
  for word in words:
    count[word] += 1
  return count

def n_grams(words,n):
  if isinstance(words,str):
    words = words.split()
  count = defaultdict(int)
  for i in range(len(words)-n+1):
    ngram = tuple(words[i:i+n])
    count[ngram] += 1
  return count

# -----------------------------------------------------------------------------
# Calculating metrics
# -----------------------------------------------------------------------------

def bleu_scores(trues, pred, n=4):
  """Compute BLEU scores for a fixed prediction, with pycocoevalcap"""
  trues = dict([(i,[r])    for i,r in enumerate(trues)])
  preds = dict([(i,[pred]) for i,_ in enumerate(trues)])
  score, all_score = Bleu(n).compute_score(trues,preds)
  return score

def all_scores(trues, pred, n=4):
  """
  Compute scores using (modified) pycocoevalcap.
  Either for a single predicted report used for all images, or for a list of the same size as the true reports.
  Returns: BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE, CIDEr, CIDEr-D
  """
  trues = dict([(i,[r]) for i,r in enumerate(trues)])
  if isinstance(pred,str):
    preds = dict([(i,[pred]) for i,_ in enumerate(trues)])
  else:
    preds = dict([(i,[r]) for i,r in enumerate(pred)])
  bleus, _ = Bleu(n).compute_score(trues,preds)
  #with closing(Meteor()) as meteor_runtime:
  #meteor, _ = Meteor().compute_score(trues,preds)
  meteor=0.0
  rouge, _ = Rouge().compute_score(trues,preds)
  ciders, _ = Cider().compute_score(trues,preds)
  return bleus,meteor, rouge, ciders

def cider_scores(trues, pred, n=4):
  """
  Compute CIDEr and CIDEr-D for a fixed prediction, with pycocoevalcap
  """
  trues = dict([(i,[r])    for i,r in enumerate(trues)])
  preds = dict([(i,[pred]) for i,_ in enumerate(trues)])
  ciders, _ = Cider().compute_score(trues,preds)
  return ciders

# -----------------------------------------------------------------------------
# Calculating BLEU
# -----------------------------------------------------------------------------

def brevity_penalty(n_true, n_pred):
  if n_pred == 0:
    return 0
  elif n_pred < n_true:
    return np.exp(1 - n_true/n_pred)
  else:
    return 1

def precision_to_bleu_score(n_correct_ngram, n_predicted_ngram, n_true_word, n_predicted_word):
  return brevity_penalty(n_true_word, n_predicted_word) * n_correct_ngram / max(1e-10,n_predicted_ngram)

def counts_to_bleu(trues, pred):
  """
  Bleu score for a single predicted report against a list of true reports
  Input is given as dictionary of n-gram counts
  """
  npr = sum(pred.values()) * len(trues)
  ntr = sum(sum(t.values()) for t in trues)
  prec = np.sum([min(i,t[w]) for t in trues for w,i in pred.items()])
  return precision_to_bleu_score(prec, npr, ntr, npr)

def table_to_bleu_precision(true_table, num_trues, pred):
  """
  Compute BLEU score (without brevity penalty) for a fixed report, given a ngram count table for the dataset.
  For a bunch of reports:
    true_table = build_word_count_table(reports)
    num_trues  = len(reports)
  """
  total = sum(pred.values()) * num_trues
  correct = 0
  for w,count in pred.items():
    for i in range(count):
      correct += true_table[(w,i)]
  return correct / max(1e-10,total)

# -----------------------------------------------------------------------------
# Random baseline
# -----------------------------------------------------------------------------

def all_scores_random_baseline(reports, train_reports):
  """
  Random baseline, as per https://ml4health.github.io/2019/pdf/175_ml4h_preprint.pdf
  """
  preds = np.random.choice(train_reports, len(reports))
  return all_scores(reports,preds)

# -----------------------------------------------------------------------------
# Removing rare words
# -----------------------------------------------------------------------------

def sum_histograms(hists):
  out = defaultdict(int)
  for hist in hists:
    for key,value in hist.items():
      out[key] += value
  return out

def word_histogram(reports):
  return sum_histograms([one_grams(r) for r in reports])

def remove_rare_words(report, histogram, threshold=5):
  return ' '.join([word for word in report.split() if histogram[word]>=threshold])

# -----------------------------------------------------------------------------
# Optimizing BLEU-1
# -----------------------------------------------------------------------------

def build_count_table(reports_ngrams):
  """
  Like build_word_count_table, but each report is represented as a dict of ngrams
  """
  table = defaultdict(int)
  for r in reports_ngrams:
    for w,count in r.items():
      for i in range(count):
        table[(w,i)] += 1
  return table

def build_word_count_table(reports):
  """
  Collect counts of all words in all reports
  Produces a table  (word,index) -> count
  The second instance of a word in a report is represented as (word,1), etc.
  """
  return build_count_table([one_grams(r) for r in reports])

def optimal_words_bleu1(reports, max_length=100):
  # Make a heap from a word count table
  # Note: heapq is a min-heap, so negate counts
  table = build_word_count_table(reports)
  heap = []
  for w,count in table.items():
    pq.heappush(heap,(-count,w))
  # Remove most frequent word from heap
  words = []
  for i in range(max_length):
    word = pq.heappop(heap)
    words.append(word[1][0])
  return words

def slow_optimal_report_bleu1(reports, max_length=100):
  words = optimal_words_bleu1(reports, max_length)
  reports_one_grams = [one_grams(r) for r in reports]
  prev_score = 0
  for i in range(1,max_length+1):
    score = counts_to_bleu(reports_one_grams, one_grams(words[:i]))
    if score > prev_score:
      prev_score = score
    else:
      return words[:i]
  return words

def optimal_report_bleu1(reports, max_length=100):
  # Make a heap from a word count table
  table = build_word_count_table(reports)
  heap = []
  for w,count in table.items():
    pq.heappush(heap,(-count,w))
  # Repeatedly remove most frequent word from heap
  words = []
  best_score = 0
  n_true = sum(table.values())
  n_correct = 0
  for i in range(max_length):
    neg_count,(word,_) = pq.heappop(heap)
    words.append(word)
    # compute bleu-1
    n_pred = len(words) * len(reports)
    n_correct += -neg_count
    score = precision_to_bleu_score(n_correct, n_pred, n_true, n_pred)
    # does this improve the score?
    if score > best_score:
      best_score = score
    else:
      return words[:-1]
  return words

# Note: optimal length is likely just the average length of reports in the dataset, such that there is no brevity penalty

# -----------------------------------------------------------------------------
# Optimizing BLEU-2
# -----------------------------------------------------------------------------

# Algorithm ideas:
# * restrict to ~100 best words as per BLEU-1
#   * maybe with a lower bound on BLEU-1 score
# * then we want to find the a highest weight path between them
# * try all possible starting words
#   * could do greedy: pick best possible next word, and repeat
# * make matrix: start*end
#   * for each n(a,b) in table of 2-grams, ordered by frequency
#     * update m[u,b]=max(m[u,b], m[u,a]+n[a,b])  if count_of[a][u,b]<max_1gram[a]
#   * can be done for a single start, tracking all possible current ends

def optimal_report_bleu2(reports, max_length=100):
  # set of words to use
  words = optimal_report_bleu1(reports, max_length)
  return optimal_report_bleu2_given_words(reports, words)

def optimal_report_bleu2_given_words(reports, words):
  # table of 2-grams
  table = build_count_table([n_grams(r,2) for r in reports])
  # add to heap, but only 2-grams that use relevant words
  heap = []
  for ((a,b),_),count in table.items():
    if a in words and b in words:
      pq.heappush(heap,(-count,a,b))
  # start from disjoint words, join them together if possible
  fragments = [[w] for w in words]
  while len(heap) > 0 and len(fragments) > 1:
    _,a,b = pq.heappop(heap)
    # find fragment ending with a, and one starting with b
    fragment_a = [f for f in fragments if f[-1] == a]
    if len(fragment_a) == 0:
      continue
    fragment_a = fragment_a[0]
    fragment_b = [f for f in fragments if f[0] == b and f != fragment_a]
    if len(fragment_b) == 0:
      continue
    fragment_b = fragment_b[0]
    # join them
    fragments.remove(fragment_a)
    fragments.remove(fragment_b)
    fragments.append(fragment_a + fragment_b)
  # join
  words = [w for f in fragments for w in f]
  return ' '.join(words)

# -----------------------------------------------------------------------------
# Optimizing BLEU-2: beam search version
# -----------------------------------------------------------------------------

def heappush_with_limit(heap, item, limit):
  """
  Add an item to a heap, limiting the size.
  The smallest item is removed if the heap would become too large
  """
  if len(heap) < limit:
    pq.heappush(heap, item)
  else:
    pq.heappushpop(heap, item)

def optimal_report_bleu_beam(reports, max_n=4, max_words=200, max_length=None, beam_width=100):
  tables = [build_count_table([n_grams(r,i) for r in reports]) for i in range(1,max_n+1)]
  num_true = len(reports)
  num_true_words = sum(tables[0].values())
  if max_length is None:
    max_length = ceil(num_true_words / num_true)
  # collect most common words
  words = []
  for ((word,),_),count in tables[0].items():
    heappush_with_limit(words, (count,word), max_words)
  words = list(set([word for _,word in words]))
  # beam search: add one word at a time
  beam = [optimal_bleu_item_empty(max_n)] # best beam items for previous length
  best = beam[0]
  for l in range(max_length):
    next_beam = []
    for _,item in beam:
      for word in words:
        next_item = optimal_bleu_item_add_word(item, word, tables)
        #print(next_item)
        heappush_with_limit(next_beam, next_item, beam_width)
    beam = next_beam
    #bp = brevity_penalty(num_true_words, len(item.words)*len(reports))
  # find best
  while len(beam) > 1:
    pq.heappop(beam)
  return beam[0]

BeamItem = namedtuple('BeamItem', ['words', 'ngram_counts', 'used'])

def optimal_bleu_item_score(item):
  # score is mean precision.
  # actual bleu score uses geometric mean, but that doesn't work if some things are zero
  num_ngrams = np.maximum(1, np.arange(len(item.words), len(item.words)-len(item.ngram_counts), -1))
  precision = item.ngram_counts / num_ngrams
  #return np.mean(precision * np.array([0.01,0.5,1,1])[:len(item.ngram_counts)])
  return np.mean(precision)

def optimal_bleu_item_empty(max_n):
  item = BeamItem([], np.zeros(max_n), defaultdict(int))
  score = 0
  return (score,item)

def optimal_bleu_item_add_word(item, word, tables):
  words = item.words + [word]
  ngram_counts = item.ngram_counts.copy()
  used = item.used.copy()
  new_item = BeamItem(words, ngram_counts, used)
  max_n = len(tables)
  for n in range(max_n):
    ngram = tuple(words[-(n+1):])
    used[ngram] += 1
    ngram_counts[n] += tables[n][(ngram,used[ngram])]
  new_score = optimal_bleu_item_score(new_item)
  return (new_score, new_item)


# -----------------------------------------------------------------------------
# Find best single report: BLEU
# -----------------------------------------------------------------------------

def pairwise_bleu_scores(reports, pred_reports, max_n=4):
  """
  Compute bleu-1..max_n for the dataset of reports, when taking one of the pred_reports as predictions
  """
  out = np.zeros((len(pred_reports),max_n))
  # start with brevity penalty
  n_true_words = sum([len(words(r)) for r in reports])
  brevity = np.array([brevity_penalty(n_true_words, len(words(r))*len(reports)) for r in pred_reports])
  # product of precision_[0:i]
  prods = 1
  for n in range(1,max_n+1):
    # precision of n-grams
    ngrams = [n_grams(r,n) for r in reports]
    table = build_count_table(ngrams)
    precisions = [table_to_bleu_precision(table, len(reports), n_grams(pred,n)) for pred in pred_reports]
    prods *= np.array(precisions)
    # geometric mean
    out[:,n-1] = brevity * (prods ** (1/n))
  return out

def slow_pairwise_bleu_scores(reports, pred_reports, n=4):
  return np.array([bleu_scores(reports, pred, n) for pred in pred_reports])

def find_best_report_bleu(reports, pred_reports, n=4):
  """
  Find the single report among `pred_reports`, that has the highest BLEU-`n` on `reports`.
  """
  bleus = pairwise_bleu_scores(reports, pred_reports, n)
  best = np.argmax(bleus[:,n-1])

  return pred_reports[best],best,bleus[best,:]

def upper_bound_bleu_score(reports, length=None, max_n=4):
  """ Upper bound on BLEU score for a single report """
  # length and brevity penalty
  n_true_words = sum([len(words(r)) for r in reports])
  if length == None:
    length = ceil(n_true_words / len(reports))
  brevity = brevity_penalty(n_true_words, length*len(reports))
  # bleu precisions
  bleus = np.zeros((max_n))
  for n in range(1,max_n+1):
    # precision of n-grams
    ngrams = [n_grams(r,n) for r in reports]
    table = build_count_table(ngrams)
    # best score when picking top length+1-n ngrams
    num_ngrams = (length+1-n)
    correct = sum(sorted(table.values())[-num_ngrams:])
    precision = correct / (num_ngrams * len(reports))
    bleus[n-1] = precision
  return brevity * np.cumprod(bleus) ** (1/np.arange(1,n+1))

def upper_bound_bleu_score_top_words(reports, num_words=100, length=None, max_n=4):
  """ Upper bound on BLEU score for a single report, using only the top num_words words
  """
  # length and brevity penalty
  n_true_words = sum([len(words(r)) for r in reports])
  if length == None:
    length = ceil(n_true_words / len(reports))
  brevity = brevity_penalty(n_true_words, length*len(reports))
  # top words
  table = build_word_count_table(reports);
  heap = []
  for (word,_),count in table.items():
    pq.heappush(heap, (-count,word))
  top_words = set()
  while len(top_words) < num_words and len(heap):
    _,word = pq.heappop(heap)
    top_words.add(word)
  # bleu precisions
  bleus = np.zeros((max_n))
  for n in range(1,max_n+1):
    # precision of n-grams
    ngrams = [n_grams(r,n) for r in reports]
    table = build_count_table(ngrams)
    # best score when picking top length+1-n ngrams
    num_ngrams = (length+1-n)
    # limit table to top words
    heap = []
    for (ngram,_),count in table.items():
      if all(word in top_words for word in ngram):
        heappush_with_limit(heap,count,num_ngrams)
    correct = sum(heap)
    precision = correct / (num_ngrams * len(reports))
    bleus[n-1] = precision
  return brevity * np.cumprod(bleus) ** (1/np.arange(1,n+1))

# -----------------------------------------------------------------------------
# Find best single report: CIDEr
# -----------------------------------------------------------------------------

def document_frequencies(reports):
  """
  Given a list of ngram dictionaries,
  Count the fraction of documents in which each term appears at least once
  """
  counts = defaultdict(float)
  for report in reports:
    for ngram in report:
      counts[ngram] += 1 / len(reports)
  return defaultdict(lambda: 1/len(reports), counts)

def to_tfidf(ngrams, doc_freq):
  vec = defaultdict(float)
  for word,count in ngrams.items():
    idf = -np.log(doc_freq[word])
    vec[word] = idf * count
    # note: term frequency is proportional to count, and we normalize anyway
  norm = np.sqrt(sum([x*x for x in vec.values()]))
  if norm == 0:
    return {}
  return {word:x/norm for word,x in vec.items()}

def inner_product(vec,vec2):
  total = 0.
  for key,value in vec.items():
    total += value * vec2[key]
  return total

def pairwise_cider_scores(reports, pred_reports, max_n=4):
  """
  Compute cider (not cider-d) for the dataset of reports, when taking one of the pred_reports as predictions
  """
  # cider is average of correlations of tf-idf vectors
  cider_n = np.zeros((len(pred_reports),max_n))
  for n in range(1,max_n+1):
    # precision of n-grams
    ngrams = [n_grams(r,n) for r in reports]
    doc_freq = document_frequencies(ngrams)
    true_vecs = [to_tfidf(r,doc_freq) for r in ngrams]
    true_vec  = sum_histograms(true_vecs)
    pred_vecs = [to_tfidf(n_grams(r,n),doc_freq) for r in pred_reports]
    cider_n[:,n-1] = [inner_product(pred,true_vec) / len(reports) for pred in pred_vecs]
  cider = np.mean(cider_n,1) * 10
  # Note: code seems to multiply by 10, even though paper does not
  return cider

def slow_pairwise_cider_scores(reports, pred_reports):
  return np.array([cider_scores(reports, pred) for pred in pred_reports])

def find_best_report_cider(reports, pred_reports):
  """
  Find the single report among `pred_reports`, that has the highest CIDEr (not CIDEr-D) on `reports`.
  """
  ciders = pairwise_cider_scores(reports, pred_reports)
  best = np.argmax(ciders)
  return pred_reports[best]

def upper_bound_cider_score(reports, max_n=4):
  """ Upper bound on cider score for a single report """
  cider_n = np.zeros((max_n))
  for n in range(1,max_n+1):
    # precision of n-grams
    ngrams = [n_grams(r,n) for r in reports]
    doc_freq = document_frequencies(ngrams)
    true_vecs = [to_tfidf(r,doc_freq) for r in ngrams]
    true_vec  = sum_histograms(true_vecs)
    # cider is maximized if pred_vec = true_vec/norm(true_vec),
    # then inner product = norm(true_vec)
    norm = np.sqrt(sum([x*x for x in true_vec.values()]))
    bound = norm / len(reports)
    cider_n[n-1] = bound
  cider = np.mean(cider_n) * 10
  # Note: code seems to multiply by 10, even though paper does not
  return cider, cider_n

# -----------------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------------

def train_test_split(n, train_fraction=0.8, seed=1):
  if not isinstance(n,int):
    n = len(n)
  rng = np.random.RandomState(seed)
  perm = rng.permutation(n)
  num_train = int(n * train_fraction)
  train = perm[0:num_train]
  test  = perm[num_train:]
  return train,test


def cross_validate(data, method, seed=1, folds=10):
  rng = np.random.RandomState(seed)
  n = len(data)
  perm = rng.permutation(n)
  results = []
  for i in range(folds):
    test  = perm[(i*n)//folds:((i+1)*n)//folds]
    train = [*perm[0:(i*n)//folds], *perm[((i+1)*n)//folds:]]
    results.append(method(data[train],data[test]))
  return results

def cross_validate_scores(data, method, **args):
  scores = cross_validate(data, lambda train,test: all_scores(test, method(train)), **args)
  return np.array(scores)

def cross_validate_with_idx(data, method, seed=1, folds=10):
  rng = np.random.RandomState(seed)
  n = len(data)
  perm = rng.permutation(n)
  results = []
  for i in range(folds):
    test  = perm[(i*n)//folds:((i+1)*n)//folds]
    train = [*perm[0:(i*n)//folds], *perm[((i+1)*n)//folds:]]
    results.append(method(data,train,test))
  return results

