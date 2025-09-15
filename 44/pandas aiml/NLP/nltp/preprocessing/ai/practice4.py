import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

# Download required NLTK resources
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

# Function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(word):
    """Convert NLTK POS tag to WordNet POS tag"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)  # Default to NOUN if tag not found

# Initialize stemmers and lemmatizers
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Compare lemmatization with and without POS
print("Manual POS specification:")
print("lemmatization", lemmatizer.lemmatize('running', pos='v'))
print("Stemming", ps.stem('running'))
print("lemmatization", lemmatizer.lemmatize('better', pos='a'))
print("Stemming", ps.stem('better'))

print("\nAutomatic POS detection:")
print("lemmatization", lemmatizer.lemmatize('running', pos=get_wordnet_pos('running')))
print("lemmatization", lemmatizer.lemmatize('better', pos=get_wordnet_pos('better')))

# Process a sample text
text = "The striped bats are hanging on their feet for best"
print("\nProcessing text with automatic POS detection:")

for word in text.split():
    # Get the appropriate POS tag automatically
    pos = get_wordnet_pos(word)
    
    # Compare lemmatization with automatic POS vs stemming
    print(f"{word} (POS: {pos}):")
    print(f"  - Lemmatized: {lemmatizer.lemmatize(word, pos=pos)}")
    print(f"  - Stemmed: {ps.stem(word)}")

# Function to process text with automatic POS detection
def process_text_with_auto_pos(text):
    """Process text with automatic POS detection for lemmatization"""
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    results = []
    for word, tag in pos_tags:
        # Convert NLTK tag to WordNet tag
        tag_first_letter = tag[0].upper()
        wordnet_pos = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }.get(tag_first_letter, wordnet.NOUN)
        
        # Lemmatize with the detected POS
        lemma = lemmatizer.lemmatize(word, pos=wordnet_pos)
        stem = ps.stem(word)
        
        results.append({
            'word': word,
            'pos_tag': tag,
            'wordnet_pos': wordnet_pos,
            'lemma': lemma,
            'stem': stem
        })
    
    return results

# Example usage of the advanced function
print("\nAdvanced text processing with automatic POS detection:")
sample_text = "The running dogs are better than walking cats"
processed = process_text_with_auto_pos(sample_text)

# Display results in a readable format
for item in processed:
    print(f"{item['word']} ({item['pos_tag']}):\n  - Lemma: {item['lemma']}\n  - Stem: {item['stem']}")

# Original code for reference (commented out)
'''
sentence = "Lorem ipsum dolor, sit amet consectetur adipisicing elit. Minus dicta nemo, labore consequatur dolorum odit maxime sequi neque at nesciunt sunt repellendus perspiciatis cupiditate, totam sed provident et explicabo quaerat corrupti laudantium fugit dolores vitae! Eum tempora iste magnam excepturi? Animi reprehenderit libero aut molestias adipisci voluptas atque asperiores laboriosam debitis deserunt aspernatur beatae est itaque, hic numquam fugiat fugit cupiditate, veniam nam error quae voluptatibus facilis deleniti eius. Iusto laboriosam harum inventore laborum, id odit deserunt excepturi. Molestiae ipsum eligendi debitis sunt ratione modi. Accusamus assumenda enim qui dolorem asperiores rerum adipisci dolor exercitationem expedita pariatur dolores deserunt nemo nisi quaerat ipsum sapiente ut, quasi fugit sunt at eius dignissimos repudiandae nesciunt debitis. Ipsa ad doloribus aliquam soluta, pariatur blanditiis eius repellendus debitis doloremque quos illum cupiditate maxime esse ipsam beatae ducimus, culpa accusamus iste hic consequuntur, eaque odio enim iusto! Nesciunt ea nostrum ipsam molestias ab. Error aspernatur optio aliquid minima magni nemo architecto, nostrum vitae placeat obcaecati debitis voluptatem! Veritatis ipsa, inventore tempora fuga facilis perferendis iusto. Praesentium atque obcaecati, magni ut illum voluptates soluta facere iusto enim commodi modi quidem placeat distinctio minima est explicabo incidunt dolor eveniet et, at ducimus, eos assumenda in. Deleniti atque reprehenderit maxime voluptates mollitia, id nulla deserunt, ea consequatur incidunt laborum alias ratione suscipit explicabo eos similique, doloribus consectetur inventore distinctio sequi aspernatur tempora quas nemo cumque. Fugiat non illum molestiae commodi qui itaque accusamus molestias corrupti illo? Fugiat aut officia porro, libero omnis nesciunt necessitatibus deleniti harum est vero nisi dolorem totam cumque laboriosam, assumenda optio officiis dolor eos eum non praesentium facere sunt tempore? Odio ipsum, magni cupiditate voluptatem possimus animi ab provident placeat nesciunt vel aut assumenda pariatur asperiores neque aperiam impedit enim omnis inventore eaque dolor deleniti illo! Repellendus exercitationem, eaque magnam sed repudiandae quam, commodi beatae nihil minus sapiente quibusdam ipsum. Animi dolorem voluptatibus voluptas commodi corporis enim distinctio totam? Libero amet natus odit eligendi vel perspiciatis ducimus quia eaque id corrupti. Ipsam aut dicta deleniti, impedit, alias ea hic recusandae quia minus, beatae consequatur! Corrupti nulla dicta similique quo magni, nisi cupiditate sit aperiam molestias minima labore eligendi enim in quod perferendis temporibus officiis molestiae possimus asperiores accusantium voluptatum! Excepturi, porro officiis. Eaque, laborum. Odio minima, recusandae sequi, enim accusamus placeat vero laborum blanditiis, at dicta quaerat. Veniam vero ullam sequi laboriosam tenetur blanditiis, impedit ipsum inventore assumenda, quibusdam suscipit enim nulla provident sit vel rerum accusantium aliquid at labore quos itaque hic, ex aspernatur. Architecto tenetur aut, recusandae assumenda molestias ipsum labore incidunt. Distinctio, tenetur eligendi perspiciatis neque a, possimus quos fugit veritatis ex cumque eum in asperiores numquam quidem quia omnis velit provident delectus alias! Voluptatum, animi fugit nesciunt eum ipsa fuga recusandae debitis sint at ducimus excepturi expedita doloremque? Assumenda error molestias exercitationem perferendis. Perspiciatis atque aut perferendis minima accusantium enim facere tempora veniam est doloremque pariatur iste unde, quaerat voluptatem deserunt culpa libero voluptate inventore quis eligendi dolores, voluptatibus optio? Iusto alias, nemo nesciunt explicabo rem distinctio dicta reprehenderit ad!!"
# print(sentence)
splt = sentence.split(" ")
# print(splt)

punc = [',','!',' ','"','@','$','?']
punct = []
words = []

for i in (splt):
    if i[-1] in punc:
        punct.append(i[:-1])

    
# for i in words :
#     if i in punc :
#         punct.append(i)



print(punct)
# print(words)

print(len(punct))


def clean_file():
    opt = input("1. file \n 2. type")

    if opt == 1:
        file_add = input("enter the address of the file !")
        file = open(file_add,'r')
'''