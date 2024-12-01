Web Mining and Applied NLP (44-620)

# Final Project: Article Summarizer
Student Name: Albert Kabore
Link to GitHub: https://github.com/albertokabore/Albert_Final_Module_7_article-summarizer

Complete the tasks in the Python Notebook in this repository.
Make sure to add and push the pkl or text file of your scraped html (this is specified in the notebook)

## Rubric

* (Question 1) Article html stored in separate file that is committed and pushed: 1 pt
* (Question 2) Polarity score printed with an appropriate label: 1 pt
* (Question 2) Number of sentences printed: 1 pt
* (Question 3) Correct (or equivalent in the case of multiple tokens with same frequency) tokens printed: 1 pt
* (Question 4) Correct (or equivalent in the case of multiple lemmas with same frequency) lemmas printed: 1 pt
* (Question 5) Histogram shown with appropriate labelling: 1 pt
* (Question 6) Histogram shown with appropriate labelling: 1 pt
* (Question 7) Cutoff score seems appropriate given histograms: 2 pts (1/score)
* (Question 8) Summary contains a shortened version of the article (less than half the number of sentences): 1 pt
* (Question 8) Summary sentences are in the same order as they appeared in the original article: 1 pt
* (Question 9) Polarity score printed with an appropriate label: 1 pt
* (Question 9) Number of sentences printed: 1 pt
* (Question 10) Summary contains a shortened version of the article (less than half the number of sentences): 1 pt
* (Question 10) Summary sentences are in the same order as they appeared in the original article: 1 pt
* (Question 11) Polarity score printed with an appropriate label: 1 pt
* (Question 11) Number of sentences printed: 1 pt
* (Question 12) Thoughtful answer based on reported polarity scores: 1 pt
* (Question 13) Thoughtful answer based on summaries: 1 pt



Perform the tasks described in the Markdown cells below. When you have completed the assignment make sure your code cells have all been run (and have output beneath them) and ensure you have committed and pushed ALL of your changes to your assignment repository.

You should bring in code from previous assignments to help you answer the questions below.

Every question that requires you to write code will have a code cell underneath it; you may either write your entire solution in that cell or write it in a python file (.py), then import and run the appropriate code to answer the question.


Clone your new repository down to your machine into your Documents folder.

git clone: https://github.com/albertokabore/Albert_Final_Module_7_article-summarizer

### Install SpaCy

```powershell
python -m venv .env
.env\Scripts\activate
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```


```powershell
git add .
git commit -m "initial commit"                         
git push origin main
```

# Import and test necessary packages

```python
from collections import Counter
import pickle
import requests
import spacy
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from spacytextblob.spacytextblob import SpacyTextBlob


!pip list

print('All prereqs installed.')
```

### Question 1. Find on the internet an article or blog post about a topic that interests you and you are able to get the text for using the technologies we have applied in the course. Get the html for the article and store it in a file (which you must submit with your project)

Retrieve and Save the HTML Content: Utilize Python to fetch the article's HTML content and save it to a file. Here's how:

```python
# URL of the NIH article on Type 2 Diabetes
url = 'https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes/type-2-diabetes'

# Send a GET request to fetch the HTML content
response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful

# Save the HTML content to a file
with open('type_2_diabetes_article.html', 'w', encoding='utf-8') as file:
    file.write(response.text)

print("HTML content saved to 'type_2_diabetes_article.html'")
```

Parse and Extract Text from the HTML: After saving the HTML, use BeautifulSoup to parse and extract the main text content:

```python
# Load the saved HTML file
with open('type_2_diabetes_article.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Extract text from specific HTML elements (e.g., paragraphs)
article_text = ''
for paragraph in soup.find_all('p'):
    article_text += paragraph.get_text() + '\n'

# Save the extracted text to a new file
with open('type_2_diabetes_article.txt', 'w', encoding='utf-8') as file:
    file.write(article_text)

print("Extracted text saved to 'type_2_diabetes_article.txt'")
```

### Question 2. Read in your article's html source from the file you created in question 1 and do sentiment analysis on the article/post's text (use .get_text()). Print the polarity score with an appropriate label. Additionally print the number of sentences in the original article (with an appropriate label)

```python
# Fetch the HTML content from the URL
url = 'https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes/type-2-diabetes'
response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful
html_content = response.text

# Save the HTML content to a file
file_path = 'type_2_diabetes_article.html'
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(html_content)

print(f"HTML content saved to '{file_path}'")

#  Parse the HTML and extract the text
soup = BeautifulSoup(html_content, 'html.parser')
article_text = soup.get_text()

# Print the extracted article content
print("\nExtracted Article Text:\n")
print(article_text)

# Load SpaCy and add SpaCyTextBlob
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")

# Perform sentiment analysis on the text
doc = nlp(article_text)
polarity = doc._.blob.polarity

# Determine the sentiment label
if polarity > 0:
    sentiment_label = "Positive"
elif polarity < 0:
    sentiment_label = "Negative"
else:
    sentiment_label = "Neutral"

# Print the polarity score with an appropriate label
print(f"\nPolarity score: {polarity:.2f} ({sentiment_label})")

# Count the number of sentences in the text
num_sentences = len(list(doc.sents))
print(f"Number of sentences in the article: {num_sentences}")
```

### Question 3. Load the article text into a trained spaCy pipeline, and determine the 5 most frequent tokens (converted to lower case). Print the common tokens with an appropriate label. Additionally, print the tokens their frequencies (with appropriate labels)

```python
# Process the article text
doc = nlp(article_text)

# Tokenize, normalize to lowercase, and count frequencies
tokens = [token.text.lower() for token in doc if token.is_alpha]  # Only consider alphabetic tokens
token_frequencies = Counter(tokens)

# Get the 5 most common tokens
most_common_tokens = token_frequencies.most_common(5)

# Print the results
print("5 Most Frequent Tokens:")
for token, frequency in most_common_tokens:
    print(f"Token: {token}, Frequency: {frequency}")
```

### Question 4. Load the article text into a trained spaCy pipeline, and determine the 5 most frequent lemmas (converted to lower case). Print the common lemmas with an appropriate label. Additionally, print the lemmas with their frequencies (with appropriate labels).

```python
# Process the article text
doc = nlp(article_text)

# Extract lemmas, normalize to lowercase, and count frequencies
lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]  # Only consider alphabetic tokens
lemma_frequencies = Counter(lemmas)

# Get the 5 most common lemmas
most_common_lemmas = lemma_frequencies.most_common(5)

# Print the results
print("5 Most Frequent Lemmas:")
for lemma, frequency in most_common_lemmas:
    print(f"Lemma: {lemma}, Frequency: {frequency}")
```

### Question 5. Make a list containing the scores (using tokens) of every sentence in the article, and plot a histogram with appropriate titles and axis labels of the scores. From your histogram, what seems to be the most common range of scores (put the answer in a comment after your code)?

```python
# Calculate sentiment scores for each sentence
sentence_scores = [sent._.blob.polarity for sent in doc.sents]

# Plot a histogram of the scores
plt.figure(figsize=(10, 6))
plt.hist(sentence_scores, bins=10, color='blue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Sentiment Scores by Sentence', fontsize=16)
plt.xlabel('Sentiment Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

### Question 6. Make a list containing the scores (using lemmas) of every sentence in the article, and plot a histogram with appropriate titles and axis labels of the scores. From your histogram, what seems to be the most common range of scores (put the answer in a comment after your code)?

```python
#  Plot a histogram of the scores
plt.figure(figsize=(10, 6))
plt.hist(sentence_scores, bins=10, color='green', edgecolor='black', alpha=0.7)
plt.title('Histogram of Sentiment Scores by Sentence (Using Lemmas)', fontsize=16)
plt.xlabel('Sentiment Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

### Question 7. Using the histograms from questions 5 and 6, decide a "cutoff" score for tokens and lemmas such that fewer than half the sentences would have a score greater than the cutoff score. Record the scores in this Markdown cell
Cutoff Score (tokens):
Cutoff Score (lemmas):
Feel free to change these scores as you generate your summaries. Ideally, we're shooting for at least 6 sentences for our summary, but don't want more than 10 (these numbers are rough estimates; they depend on the length of your article).


```python
# Decide cutoff scores based on histograms
# Calculate cutoff such that fewer than half the sentences exceed the cutoff
cutoff_tokens = np.percentile(token_scores, 50)  # Median score for tokens
cutoff_lemmas = np.percentile(lemma_scores, 50)  # Median score for lemmas

# Print the cutoff scores
print(f"Cutoff Score (tokens): {cutoff_tokens:.2f}")
print(f"Cutoff Score (lemmas): {cutoff_lemmas:.2f}")

#  Select sentences based on cutoff scores
# Token-based summary
token_summary = [sent.text for sent in doc.sents if sent._.blob.polarity > cutoff_tokens]

# Lemma-based summary
lemma_summary = [
    sent.text for sent in doc.sents
    if len([token for token in sent if token.is_alpha]) > 0 and
    (sum(token._.blob.polarity for token in sent if token.is_alpha) / len([token for token in sent if token.is_alpha])) > cutoff_lemmas
]

# Print the summaries
print("\nToken-Based Summary:")
for sentence in token_summary[:10]:  # Limit to 10 sentences
    print(f"- {sentence}")

print("\nLemma-Based Summary:")
for sentence in lemma_summary[:10]:  # Limit to 10 sentences
    print(f"- {sentence}")

# Output the number of selected sentences
print(f"\nNumber of sentences in Token-Based Summary: {len(token_summary)}")
print(f"Number of sentences in Lemma-Based Summary: {len(lemma_summary)}")
```

```python
# Calculate sentiment scores for tokens and lemmas
# Token-based scores
token_scores = [sent._.blob.polarity for sent in doc.sents]

# Lemma-based scores (ensure division by zero does not occur)
lemma_scores = [
    (sum(token._.blob.polarity for token in sent if token.is_alpha) / len([token for token in sent if token.is_alpha]))
    if len([token for token in sent if token.is_alpha]) > 0 else 0
    for sent in doc.sents
]

# Calculate cutoff scores
cutoff_tokens = np.percentile(token_scores, 50)  # Median score for tokens
cutoff_lemmas = np.percentile(lemma_scores, 50)  # Median score for lemmas

# Plot histograms
plt.figure(figsize=(15, 6))

# Token-based histogram
plt.subplot(1, 2, 1)
plt.hist(token_scores, bins=10, color='blue', edgecolor='black', alpha=0.7)
plt.axvline(cutoff_tokens, color='red', linestyle='--', label=f"Cutoff: {cutoff_tokens:.2f}")
plt.title("Histogram of Token Sentiment Scores", fontsize=16)
plt.xlabel("Sentiment Score (tokens)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()

# Lemma-based histogram
plt.subplot(1, 2, 2)
plt.hist(lemma_scores, bins=10, color='green', edgecolor='black', alpha=0.7)
plt.axvline(cutoff_lemmas, color='red', linestyle='--', label=f"Cutoff: {cutoff_lemmas:.2f}")
plt.title("Histogram of Lemma Sentiment Scores", fontsize=16)
plt.xlabel("Sentiment Score (lemmas)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()

# Print the cutoff scores
print(f"Cutoff Score (tokens): {cutoff_tokens:.2f}")
print(f"Cutoff Score (lemmas): {cutoff_lemmas:.2f}")
```

### Question 8. Create a summary of the article by going through every sentence in the article and adding it to an (initially) empty list if its score (based on tokens) is greater than the cutoff score you identified in question 8. If your loop variable is named sent, you may find it easier to add sent.text.strip() to your list of sentences. Print the summary (I would cleanly generate the summary text by joining the strings in your list together with a space (' '.join(sentence_list)).

```python
# Calculate sentiment scores for tokens
token_scores = [sent._.blob.polarity for sent in doc.sents]

# Determine the cutoff score (median value)
token_cutoff = np.percentile(token_scores, 50)  # Median value

# Generate a summary based on the cutoff score
summary_sentences = [sent.text.strip() for sent, score in zip(doc.sents, token_scores) if score > token_cutoff]

# Combine sentences into a single summary text
summary_text = ' '.join(summary_sentences)

# Print the summary
print("Article Summary:")
print(summary_text)
```

### Question 9. Print the polarity score of your summary you generated with the token scores (with an appropriate label). Additionally, print the number of sentences in the summarized article.

```python
#  Calculate the polarity score of the summary
summary_doc = nlp(summary_text)
summary_polarity = summary_doc._.blob.polarity

# Count the number of sentences in the summary
num_summary_sentences = len(list(summary_doc.sents))

#  Print the results
print(f"Polarity Score of Summary: {summary_polarity:.2f}")
print(f"Number of Sentences in the Summary: {num_summary_sentences}")
```

### Question 10. Create a summary of the article by going through every sentence in the article and adding it to an (initially) empty list if its score (based on lemmas) is greater than the cutoff score you identified in question 8. If your loop variable is named sent, you may find it easier to add sent.text.strip() to your list of sentences. Print the summary (I would cleanly generate the summary text by joining the strings in your list together with a space (' '.join(sentence_list)).

```python
#  Generate a summary based on lemma scores and the cutoff score
lemma_summary_sentences = [
    sent.text.strip()
    for sent, score in zip(doc.sents, lemma_scores)
    if score > lemma_cutoff
]

# Combine sentences into a single summary text
lemma_summary_text = ' '.join(lemma_summary_sentences)

# Print the summary
print("Article Summary (Based on Lemmas):")
print(lemma_summary_text)
```

### Question 11. Print the polarity score of your summary you generated with the lemma scores (with an appropriate label). Additionally, print the number of sentences in the summarized article.

```python
#  Calculate the polarity score of the lemma-based summary
lemma_summary_doc = nlp(lemma_summary_text)
lemma_summary_polarity = lemma_summary_doc._.blob.polarity

# Count the number of sentences in the lemma-based summary
num_lemma_summary_sentences = len(list(lemma_summary_doc.sents))

# Print the results
print(f"Polarity Score of Lemma-Based Summary: {lemma_summary_polarity:.2f}")
print(f"Number of Sentences in the Lemma-Based Summary: {num_lemma_summary_sentences}")
```

### Question 12. Compare your polarity scores of your summaries to the polarity scores of the initial article. Is there a difference? Why do you think that may or may not be?. Answer in this Markdown cell.

### Question 13. Based on your reading of the original article, which summary do you think is better (if there's a difference). Why do you think this might be?

```powershell
git add .
git commit -m "final commit"                         
git push origin main
```

!jupyter nbconvert --to html article-summarizer.ipynb