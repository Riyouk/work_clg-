import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
import re
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

sentence = "!Lorem ipsum dolor, sit amet consectetur adipisicing elit. Minus dicta nemo, labore consequatur dolorum odit maxime sequi neque at nesciunt sunt repellendus perspiciatis cupiditate, totam sed provident et explicabo quaerat corrupti laudantium fugit dolores vitae! Eum tempora iste magnam excepturi? Animi reprehenderit libero aut molestias adipisci voluptas atque asperiores laboriosam debitis deserunt aspernatur beatae est itaque, hic numquam fugiat fugit cupiditate, veniam nam error quae voluptatibus facilis deleniti eius. Iusto laboriosam harum inventore laborum, id odit deserunt excepturi."

#retain only words
text = re.sub('[^a-zA-Z]',' ',sentence)
print(text)

# transform one form
text = text.lower()
print("\n",text)

#tokenize 

words = text.split()
print("\n",words)


print("remove stopwords")
ps = PorterStemmer()
text = [ps.stem(word) for word in words if word not in STOPWORDS]
print("\n the base words",text)
# def clean_text(text):

lemematizer = WordNetLemmatizer()
print("\n lemmatization")
text = [lemematizer.lemmatize(word) for word in words if word not in STOPWORDS]


# words_ap = []
# for i in text :
#     words_ap.append(i)
# print("\n the base words",text)

# tokens = nltk.word_tokenize(text)
# print("tokens",tokens)
words_ap = []
pos_t = []
pos_descriptions = {
        # Noun tags
        'NN': 'Noun, singular',
        'NNS': 'Noun, plural',
        'NNP': 'Proper noun, singular',
        'NNPS': 'Proper noun, plural',
        
        # Verb tags
        'VB': 'Verb, base form',
        'VBD': 'Verb, past tense',
        'VBG': 'Verb, gerund or present participle',
        'VBN': 'Verb, past participle',
        'VBP': 'Verb, non-3rd person singular present',
        'VBZ': 'Verb, 3rd person singular present',
        
        # Adjective tags
        'JJ': 'Adjective',
        'JJR': 'Adjective, comparative',
        'JJS': 'Adjective, superlative',
        
        # Adverb tags
        'RB': 'Adverb',
        'RBR': 'Adverb, comparative',
        'RBS': 'Adverb, superlative',
        
        # Pronoun tags
        'PRP': 'Personal pronoun',
        'PRP$': 'Possessive pronoun',
        'WP': 'Wh-pronoun',
        'WP$': 'Possessive wh-pronoun',
        
        # Determiner tags
        'DT': 'Determiner',
        'WDT': 'Wh-determiner',
        
        # Preposition and conjunction tags
        'IN': 'Preposition or subordinating conjunction',
        'CC': 'Coordinating conjunction',
        
        # Other tags
        'CD': 'Cardinal number',
        'EX': 'Existential there',
        'FW': 'Foreign word',
        'LS': 'List item marker',
        'MD': 'Modal',
        'PDT': 'Predeterminer',
        'POS': 'Possessive ending',
        'RP': 'Particle',
        'SYM': 'Symbol',
        'TO': 'to',
        'UH': 'Interjection',
        'WRB': 'Wh-adverb'
    }
# pos_descriptions_list = [get_pos_description.get(tag,"")]
pos_descriptions_list = []


pos_tags = nltk.pos_tag(text)
print(pos_tags)


for words,pos in pos_tags:
    words_ap.append(words)
    pos_t.append(pos)
    
# key = pos_descriptions.keys()
# values = pos_descriptions.values()

for key,val in pos_descriptions.items():
    print(key)
    print(val)
    if key in pos_t:
        pos_descriptions_list.append(val)


print("\n words",words_ap)
print("\n pos",pos_t)
print("\n description : ",pos_descriptions_list)


from nltk import ngrams 

#generate n-grams 
for i in text:
    ugram = list(ngrams(i,1))
    # print("\n ugram",ugram)

    bigram = list(ngrams(i,2))
    # print("\n bigram",bigram)

    trigram = list(ngrams(i,3))
    # print("\n trigram",trigram)



with open("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/NLP/nltp/preprocessing/manual/datafram.csv", "w") as f:
    # write header
    f.write("words,post_key,pos_description\n")
    
    # write rows
    for w, t, d in zip(words_ap, pos_t, pos_descriptions_list):
        f.write(f"{w},{t},{d}\n")
