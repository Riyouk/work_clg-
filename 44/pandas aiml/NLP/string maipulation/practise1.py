sentence = "Lorem ipsum dolor, sit amet consectetur adipisicing elit. Minus dicta nemo, labore consequatur dolorum odit maxime sequi neque at nesciunt sunt repellendus perspiciatis cupiditate, totam sed provident et explicabo quaerat corrupti laudantium fugit dolores vitae! Eum tempora iste magnam excepturi? Animi reprehenderit libero aut molestias adipisci voluptas atque asperiores laboriosam debitis deserunt aspernatur beatae est itaque, hic numquam fugiat fugit cupiditate, veniam nam error quae voluptatibus facilis deleniti eius. Iusto laboriosam harum inventore laborum, id odit deserunt excepturi. Molestiae ipsum eligendi debitis sunt ratione modi. Accusamus assumenda enim qui dolorem asperiores rerum adipisci dolor exercitationem expedita pariatur dolores deserunt nemo nisi quaerat ipsum sapiente ut, quasi fugit sunt at eius dignissimos repudiandae nesciunt debitis. Ipsa ad doloribus aliquam soluta, pariatur blanditiis eius repellendus debitis doloremque quos illum cupiditate maxime esse ipsam beatae ducimus, culpa accusamus iste hic consequuntur, eaque odio enim iusto! Nesciunt ea nostrum ipsam molestias ab. Error aspernatur optio aliquid minima magni nemo architecto, nostrum vitae placeat obcaecati debitis voluptatem! Veritatis ipsa, inventore tempora fuga facilis perferendis iusto. Praesentium atque obcaecati, magni ut illum voluptates soluta facere iusto enim commodi modi quidem placeat distinctio minima est explicabo incidunt dolor eveniet et, at ducimus, eos assumenda in. Deleniti atque reprehenderit maxime voluptates mollitia, id nulla deserunt, ea consequatur incidunt laborum alias ratione suscipit explicabo eos similique, doloribus consectetur inventore distinctio sequi aspernatur tempora quas nemo cumque. Fugiat non illum molestiae commodi qui itaque accusamus molestias corrupti illo? Fugiat aut officia porro, libero omnis nesciunt necessitatibus deleniti harum est vero nisi dolorem totam cumque laboriosam, assumenda optio officiis dolor eos eum non praesentium facere sunt tempore? Odio ipsum, magni cupiditate voluptatem possimus animi ab provident placeat nesciunt vel aut assumenda pariatur asperiores neque aperiam impedit enim omnis inventore eaque dolor deleniti illo! Repellendus exercitationem, eaque magnam sed repudiandae quam, commodi beatae nihil minus sapiente quibusdam ipsum. Animi dolorem voluptatibus voluptas commodi corporis enim distinctio totam? Libero amet natus odit eligendi vel perspiciatis ducimus quia eaque id corrupti. Ipsam aut dicta deleniti, impedit, alias ea hic recusandae quia minus, beatae consequatur! Corrupti nulla dicta similique quo magni, nisi cupiditate sit aperiam molestias minima labore eligendi enim in quod perferendis temporibus officiis molestiae possimus asperiores accusantium voluptatum! Excepturi, porro officiis. Eaque, laborum. Odio minima, recusandae sequi, enim accusamus placeat vero laborum blanditiis, at dicta quaerat. Veniam vero ullam sequi laboriosam tenetur blanditiis, impedit ipsum inventore assumenda, quibusdam suscipit enim nulla provident sit vel rerum accusantium aliquid at labore quos itaque hic, ex aspernatur. Architecto tenetur aut, recusandae assumenda molestias ipsum labore incidunt. Distinctio, tenetur eligendi perspiciatis neque a, possimus quos fugit veritatis ex cumque eum in asperiores numquam quidem quia omnis velit provident delectus alias! Voluptatum, animi fugit nesciunt eum ipsa fuga recusandae debitis sint at ducimus excepturi expedita doloremque? Assumenda error molestias exercitationem perferendis. Perspiciatis atque aut perferendis minima accusantium enim facere tempora veniam est doloremque pariatur iste unde, quaerat voluptatem deserunt culpa libero voluptate inventore quis eligendi dolores, voluptatibus optio? Iusto alias, nemo nesciunt explicabo rem distinctio dicta reprehenderit ad!!"
# print(sentence)
splt = sentence.split(" ")
# print(splt)

# Define punctuation marks to check
punc = [',', '!', ' ', '"', '@', '$', '?', '.', ':', ';', '(', ')', '[', ']', '{', '}', '-', '_', '/']
punct = []
words = []

# Process each word in the split sentence
for i in splt:
    if i and i[-1] in punc:  # Check if word ends with punctuation
        punct.append(i[:-1])  # Add word without punctuation to punct list
        words.append(i[:-1])  # Add word without punctuation to words list
    else:
        words.append(i)  # Add word as is to words list

# Print results
print("Words with punctuation removed:")
print(punct)
print("All words:")
print(words)
print(f"Number of words with punctuation: {len(punct)}")
print(f"Total number of words: {len(words)}")

# Function to count word frequency
def count_word_frequency(word_list):
    word_freq = {}
    for word in word_list:
        word = word.lower()  # Convert to lowercase for case-insensitive counting
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    return word_freq

# Function to clean text from a file or user input
def clean_file():
    opt = input("Choose an option:\n1. Process from file\n2. Process from typed text\nEnter choice (1 or 2): ")

    text = ""
    if opt == "1":
        file_add = input("Enter the path to the file: ")
        try:
            with open(file_add, 'r', encoding='utf-8') as file:
                text = file.read()
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")
            return
        except Exception as e:
            print(f"An error occurred: {e}")
            return
    elif opt == "2":
        text = input("Enter the text you want to process: ")
    else:
        print("Invalid option. Please choose 1 or 2.")
        return
    
    # Process the text
    process_text(text)

# Function to process text
def process_text(text):
    # Split text into words
    words = text.split()
    
    # Remove punctuation
    clean_words = []
    for word in words:
        # Remove punctuation at the beginning and end of words
        clean_word = word
        while clean_word and clean_word[0] in punc:
            clean_word = clean_word[1:]
        while clean_word and clean_word[-1] in punc:
            clean_word = clean_word[:-1]
        
        if clean_word:  # Only add non-empty words
            clean_words.append(clean_word)
    
    # Count word frequency
    word_freq = count_word_frequency(clean_words)
    
    # Sort words by frequency (most frequent first)
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Print results
    print("\nText Analysis Results:")
    print(f"Total words: {len(clean_words)}")
    print(f"Unique words: {len(word_freq)}")
    
    print("\nTop 10 most frequent words:")
    for word, count in sorted_words[:10]:
        print(f"{word}: {count} times")
    
    # Calculate average word length
    avg_length = sum(len(word) for word in clean_words) / len(clean_words) if clean_words else 0
    print(f"\nAverage word length: {avg_length:.2f} characters")

# Function to find specific patterns in text
def find_pattern(text, pattern):
    import re
    matches = re.findall(pattern, text)
    return matches

# Main execution
if __name__ == "__main__":
    # Process the predefined sentence
    print("\nAnalysis of predefined sentence:")
    process_text(sentence)
    
    # Ask if user wants to process another text
    user_choice = input("\nDo you want to process another text? (y/n): ")
    if user_choice.lower() == 'y':
        clean_file()
        