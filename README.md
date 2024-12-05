Web Mining and Applied NLP (44-620)

# Final Project: Article Summarizer

### Student Name: Albert Kabore

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
# Import and test necessary packages

from collections import Counter
import pickle
import requests
import spacy
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud
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
# Step 4: Text cleaning and tokenization
tokens = [
    token.text.lower().strip()
    for token in doc
    if not token.is_stop  # Remove stop words
    and not token.is_punct  # Remove punctuation
    and not token.is_digit  # Remove numbers
    and token.text.strip()  # Remove empty strings or whitespace
]

# Step 5: Count token frequencies
token_counts = Counter(tokens)

# Get the 5 most common tokens
most_common_tokens = token_counts.most_common(5)

# Print the results
print("5 Most Frequent Tokens:")
for token, freq in most_common_tokens:
    print(f"Token: '{token}' - Frequency: {freq}")
 ```

### Question 4. Load the article text into a trained spaCy pipeline, and determine the 5 most frequent lemmas (converted to lower case). Print the common lemmas with an appropriate label. Additionally, print the lemmas with their frequencies (with appropriate labels).

```python
# Extract and clean lemmas
lemmas = [
    token.lemma_.lower().strip()
    for token in doc
    if not token.is_stop  # Remove stop words
    and not token.is_punct  # Remove punctuation
    and not token.is_digit  # Remove numbers
    and token.text.strip()  # Remove empty strings or whitespace
]

# Count lemma frequencies
lemma_counts = Counter(lemmas)

# Get the 5 most common lemmas
most_common_lemmas = lemma_counts.most_common(5)

# Print the results
print("5 Most Frequent Lemmas:")
for lemma, freq in most_common_lemmas:
```

### Question 5. Make a list containing the scores (using tokens) of every sentence in the article, and plot a histogram with appropriate titles and axis labels of the scores. From your histogram, what seems to be the most common range of scores (put the answer in a comment after your code)?

```python
# Create a list of sentiment scores for each sentence
sentence_scores = [sent._.blob.polarity for sent in doc.sents]

# Plot a histogram of the sentiment scores
plt.figure(figsize=(10, 6))
plt.hist(sentence_scores, bins=10, edgecolor='black', alpha=0.7)
plt.title('Histogram of Sentence Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Print the sentiment scores (optional)
print("Sentence Sentiment Scores:")
print(sentence_scores)

# Comment on the most common range of scores
# Based on the histogram, the most common range of scores appears to be around 0 (neutral),
# indicating the article's tone is largely factual and unbiased.
```

### Question 6. Make a list containing the scores (using lemmas) of every sentence in the article, and plot a histogram with appropriate titles and axis labels of the scores. From your histogram, what seems to be the most common range of scores (put the answer in a comment after your code)?

```python
# Create a list of sentiment scores using lemmas for each sentence
sentence_lemma_scores = []
for sent in doc.sents:
    lemmas = [token.lemma_.lower() for token in sent if not token.is_stop and not token.is_punct]
    lemma_doc = nlp(" ".join(lemmas))
    sentence_lemma_scores.append(lemma_doc._.blob.polarity)

# Plot a histogram of the sentiment scores using lemmas
plt.figure(figsize=(10, 6))
plt.hist(sentence_lemma_scores, bins=10, edgecolor='black', alpha=0.7)
plt.title('Histogram of Sentence Sentiment Scores (Using Lemmas)')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Print the sentiment scores (optional)
print("Sentence Sentiment Scores (Using Lemmas):")
print(sentence_lemma_scores)

# Comment on the most common range of scores
# Based on the histogram, the most common range of scores appears to be around 0 (neutral),
# indicating the article's tone is largely factual and unbiased.
```

### Question 7. Using the histograms from questions 5 and 6, decide a "cutoff" score for tokens and lemmas such that fewer than half the sentences would have a score greater than the cutoff score. Record the scores in this Markdown cell
Cutoff Score (tokens):
Cutoff Score (lemmas):
Feel free to change these scores as you generate your summaries. Ideally, we're shooting for at least 6 sentences for our summary, but don't want more than 10 (these numbers are rough estimates; they depend on the length of your article).


```python
# Create sentiment scores for tokens and lemmas
sentence_token_scores = [sent._.blob.polarity for sent in doc.sents]
sentence_lemma_scores = []
for sent in doc.sents:
    lemmas = [token.lemma_.lower() for token in sent if not token.is_stop and not token.is_punct]
    lemma_doc = nlp(" ".join(lemmas))
    sentence_lemma_scores.append(lemma_doc._.blob.polarity)

# Calculate cutoff scores dynamically
token_cutoff = np.percentile(sentence_token_scores, 50)  # Median score (50th percentile)
lemma_cutoff = np.percentile(sentence_lemma_scores, 50)  # Median score (50th percentile)

# Filter sentences based on cutoff scores
selected_sentences_tokens = [sent.text for sent, score in zip(doc.sents, sentence_token_scores) if score > token_cutoff]
selected_sentences_lemmas = [sent.text for sent, score in zip(doc.sents, sentence_lemma_scores) if score > lemma_cutoff]

# Print selected sentences
print("Selected Sentences (Tokens):")
for sentence in selected_sentences_tokens:
    print(sentence)

print("\nSelected Sentences (Lemmas):")
for sentence in selected_sentences_lemmas:
    print(sentence)

# Plot histograms with cutoff lines
plt.figure(figsize=(12, 6))

# Tokens histogram
plt.subplot(1, 2, 1)
plt.hist(sentence_token_scores, bins=10, edgecolor='black', alpha=0.7)
plt.axvline(x=token_cutoff, color='red', linestyle='--', label=f"Cutoff = {token_cutoff:.2f}")
plt.title('Histogram of Sentence Sentiment Scores (Tokens)')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Lemmas histogram
plt.subplot(1, 2, 2)
plt.hist(sentence_lemma_scores, bins=10, edgecolor='black', alpha=0.7)
plt.axvline(x=lemma_cutoff, color='red', linestyle='--', label=f"Cutoff = {lemma_cutoff:.2f}")
plt.title('Histogram of Sentence Sentiment Scores (Lemmas)')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Record cutoffs
print(f"\nCutoff Score (tokens): {token_cutoff:.2f}")
print(f"Cutoff Score (lemmas): {lemma_cutoff:.2f}")
```

### Question 8. Create a summary of the article by going through every sentence in the article and adding it to an (initially) empty list if its score (based on tokens) is greater than the cutoff score you identified in question 8. If your loop variable is named sent, you may find it easier to add sent.text.strip() to your list of sentences. Print the summary (I would cleanly generate the summary text by joining the strings in your list together with a space (' '.join(sentence_list)).

```python
# Process the article text with SpaCy
doc = nlp(article_text)

# Create sentiment scores for tokens
sentence_token_scores = [sent._.blob.polarity for sent in doc.sents]

# Define cutoff score dynamically (50th percentile)
token_cutoff = np.percentile(sentence_token_scores, 50)  # Median score

# Create a summary list for sentences with scores greater than the cutoff
summary_sentences = [sent.text.strip() for sent, score in zip(doc.sents, sentence_token_scores) if score > token_cutoff]

# Join the summary sentences into a single text
summary_text = ' '.join(summary_sentences)

# Print the summary
print("Summary of the Article:")
print(summary_text)

# Print the cutoff score for reference
print(f"\nCutoff Score (tokens): {token_cutoff:.2f}")
```


### Question 9. Print the polarity score of your summary you generated with the token scores (with an appropriate label). Additionally, print the number of sentences in the summarized article.

```python
# Calculate the polarity score of the summary
summary_doc = nlp(summary_text)
summary_polarity = summary_doc._.blob.polarity

# Count the number of sentences in the summary
summary_sentence_count = len(list(summary_doc.sents))

# Print the results
print(f"Polarity Score of the Summary: {summary_polarity:.2f}")
print(f"Number of Sentences in the Summary: {summary_sentence_count}")
```

### Question 10. Create a summary of the article by going through every sentence in the article and adding it to an (initially) empty list if its score (based on lemmas) is greater than the cutoff score you identified in question 8. If your loop variable is named sent, you may find it easier to add sent.text.strip() to your list of sentences. Print the summary (I would cleanly generate the summary text by joining the strings in your list together with a space (' '.join(sentence_list)).

```python
# Create a summary list for sentences with lemma-based scores greater than the cutoff
summary_sentences_lemmas = [
    sent.text.strip() 
    for sent, score in zip(doc.sents, sentence_lemma_scores) 
    if score > lemma_cutoff
]

# Join the summary sentences into a single text
summary_text_lemmas = ' '.join(summary_sentences_lemmas)

# Print the summary
print("Summary of the Article (Based on Lemmas):")
print(summary_text_lemmas)

# Print the cutoff score for reference
print(f"\nCutoff Score (lemmas): {lemma_cutoff:.2f}")
```

### Question 11. Print the polarity score of your summary you generated with the lemma scores (with an appropriate label). Additionally, print the number of sentences in the summarized article.

```python
# Calculate the polarity score of the lemma-based summary
summary_doc_lemmas = nlp(summary_text_lemmas)
summary_polarity_lemmas = summary_doc_lemmas._.blob.polarity

# Count the number of sentences in the lemma-based summary
summary_sentence_count_lemmas = len(list(summary_doc_lemmas.sents))

# Print the results
print(f"Polarity Score of the Lemma-Based Summary: {summary_polarity_lemmas:.2f}")
print(f"Number of Sentences in the Lemma-Based Summary: {summary_sentence_count_lemmas}")
```

### Question 12. Compare your polarity scores of your summaries to the polarity scores of the initial article. Is there a difference? Why do you think that may or may not be?. Answer in this Markdown cell.

### Question 13. Based on your reading of the original article, which summary do you think is better (if there's a difference). Why do you think this might be?

```powershell
git add .
git commit -m "final commit"                         
git push origin main
```

!jupyter nbconvert --to html article-summarizer.ipynb