#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import gensim
import os
import codecs
import pandas as pd
from sklearn import linear_model
import numpy as np
import StringIO

model_en = gensim.models.KeyedVectors.load_word2vec_format(os.path.expanduser("~/wiki.en.vec"))
model_ta = gensim.models.KeyedVectors.load_word2vec_format(os.path.expanduser("~/wiki.ta.vec"))

with open("top_en_ta.csv", "rb") as myfile:
    decoded = codecs.decode(myfile.read(), "utf-8", "ignore")
    encoded = codecs.encode(decoded, "utf-8", "ignore")
    fakefile = StringIO.StringIO(encoded)

df = pd.read_csv(fakefile, encoding='utf-8')
for n in range(len(df)):
    if df['tamil'][n] not in model_ta.vocab or df['english'][n] not in model_en.vocab:
        df = df.drop(n)
df = df.reset_index(drop=True)

df['vector_ta'] = [model_ta[df['tamil'][n]] for n in range(len(df))]
df['vector_en'] = [model_en[df['english'][n]] for n in range(len(df))]
ma_train_ta = pd.DataFrame(df['vector_ta'][:6000].tolist()).values
ma_train_en = pd.DataFrame(df['vector_en'][:6000].tolist()).values


def most_similar_vect(self, vect_enter, top_n=10):
    self.init_sims()
    vect_unit = gensim.matutils.unitvec(vect_enter)
    dists = np.dot(np.squeeze(np.asarray(self.syn0norm)), np.squeeze(np.asarray(vect_unit)))
    if not top_n:
        return dists
    best = np.argsort(dists)[::-1][:top_n]
    # ignore (don't return) words from the input
    result = [(self.index2word[sim], float(dists[sim])) for sim in best]
    return result[:top_n]


"""
ta to en
"""
clf_ta_en = linear_model.LinearRegression()
clf_ta_en.fit(ma_train_ta, ma_train_en)


def traducwithsco_ta_to_en(w, numb=10):
    return most_similar_vect(model_en, clf_ta_en.predict(model_ta[w]), numb)


def traduclist_ta_to_en(w, numb=10):
    return [traducwithsco_ta_to_en(w, numb)[k][0] for k in range(numb)]


for n in range(6000, 6010):
    print(df['english'][n])
    print(df['tamil'][n])
    print(traduclist_ta_to_en(df['tamil'][n], 5))

"""
clearing
தீர்வு
[u'solve', u'solution', u'solving', u'resolving', u'solutions']

mess
குழப்பம்
[u'confusion', u'misunderstandable', u'misinterpration', u'confusing', u'misinterpretting']

reward
வெகுமதி
[u'exorbitantly', u'exorbitant', u'profiteering', u'enticements', u'disincentivise']

diet
உணவில்
[u'nutritionally', u'antinutritional', u'uncooked', u'nutritive', u'food']

rounds
சுற்று
[u'circuit', u'circuit,', u'circuit\u2014the', u'circuits', u'circuitz']

frustration
ஏமாற்றம்
[u'disappointed', u'faltering', u'embarrassment', u'desperation', u'embarrasment']

treaty
ஒப்பந்தம்
[u'renegotiated', u'renegotiation', u'agreement', u'renegotiations', u'renegotiate']

planted
நடப்படுகிறது
[u'happens', u'fluctuates', u'takes', u'occurs', u'puts']

degradation
சீரழிவு
[u'ameliorating', u'deterioration', u'ameliorative', u'amelioration', u'ameliorations']
"""

traductest_ta_to_en = [df['english'][n] in traduclist_ta_to_en(df['tamil'][n], 1) for n in range(6000, 6500)]
scorefinal_ta_to_en = sum(traductest_ta_to_en) / len(traductest_ta_to_en)
print(scorefinal_ta_to_en)

"""
en to ta
"""
clf_en_ta = linear_model.LinearRegression()
clf_en_ta.fit(ma_train_en, ma_train_ta)


def traducwithsco_en_to_ta(w, numb=10):
    return most_similar_vect(model_ta, clf_en_ta.predict(model_en[w]), numb)


def traduclist_en_to_ta(w, numb=10):
    return [traducwithsco_en_to_ta(w, numb)[k][0] for k in range(numb)]


for n in range(7000, 7010):
    print(df['tamil'][n])
    print(df['english'][n])
    for i in traduclist_en_to_ta(df['english'][n], 5):
        print(i)

"""
கலைப்பு
liquidation
நஷ்டஈடு
கடனீட்டு
கடனாளி
கடனளிப்பு
ஏஐஜி

கவரும்
lure
முயலுகையில்
விரட்ட
மீன்பிடிக்க
துரத்திக்கொண்டு
காப்பாற்றிக்கொள்ள

விசைப்பலகைகள்
keyboards
pcகள்
pdaகள்
cpuகள்
ஏபிஐகள்
lcdகள்

தீர்க்கரேகை
meridian
ஐகன்சு–பிரனெல்
அட்சரேகை
அதிபரவளைவின்
காம்பர்ட்சேவ்

தூக்க
sleeping
குளியலறையில்
வயிறார
கழிப்பறையில்
தூங்கும்
தூங்காத

தக்கவைத்து
retaining
தக்கவைத்திருக்கும்
hpfs
தக்கவைத்துக்கொள்வதற்காக
தக்கவைக்கும்
மாற்றியமைப்பதற்கான

இடைவெளி
interval
μl
−∞
v/r
λm
a,bஇல்

சுதந்திரம்
liberty
எட்ஜ்பாஸ்டன்
கம்னியூஸ்ட்
ரீகன்புக்ஸ்
கம்னியூஸ்டு
லேக்ஹவுஸ்

பற்கள்
teeth
நகங்கள்
விரலெலும்புகள்
இடுப்பெலும்புகள்
களிம்புகள்
முலைக்காம்புகள்

வாட்கின்ஸ்
watkins
ஹோர்டன்
டேவ்சன்
ஜெஃப்சன்
டாம்லின்சன்
ரொபின்சன்
"""
