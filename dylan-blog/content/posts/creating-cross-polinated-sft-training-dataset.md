+++
title = 'Creating Cross-Pollinated SFT Training Dataset for Novel Knowledge Recombination'
date = 2025-03-13T03:41:39-07:00
draft = false
+++

In the realm of large language models (LLMs), the quality and diversity of training data significantly impact a model's ability to generate creative, insightful responses. While traditional training approaches often treat different knowledge domains as separate silos, there's a compelling opportunity to create more versatile models by deliberately cross-pollinating knowledge across domains.

This blog post explores a methodology for creating a specialized Supervised Fine-Tuning (SFT) dataset that deliberately bridges diverse knowledge domains—specifically, how to extract, align, and combine content from textbooks of vastly different genres such as mathematics and history. The goal is to create embedding links in the LLM's weights that enable it to recombine knowledge in novel ways, essentially teaching the model the epistemology of how knowledge forms and interconnects.

## Why Cross-Pollinate Knowledge Domains?

Research has consistently shown that increased diversity in training data improves cross-domain knowledge and downstream generalization in large language models. For example, The Pile dataset (825 GB from 22 diverse sources) yielded models with stronger broad knowledge than those trained on single-source data.

However, simply including diverse texts isn't enough. By deliberately aligning and connecting concepts across domains, we can:

1. **Teach analogical reasoning**: Help models understand how concepts in one domain might map to another
2. **Encourage novel insights**: Create neural pathways that facilitate unexpected but valuable connections
3. **Develop epistemological understanding**: Help models grasp how knowledge is structured and interconnected across fields
4. **Reduce domain isolation**: Prevent the model from treating knowledge areas as completely separate entities

## The Cross-Pollination Process

Let's break down the process of creating this specialized dataset:

### 1. Extracting Textual Data from Textbooks

The first step is obtaining raw text from source textbooks. Depending on the format, you have several options:

#### For Digital Textbooks (PDF, EPUB)

```python
import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# Extract text from math and history textbooks
math_text = extract_text_from_pdf("math_textbook.pdf")
history_text = extract_text_from_pdf("history_textbook.pdf")

# Clean and preprocess the text
def clean_text(text):
    # Remove page numbers
    text = re.sub(r'\n\d+\n', '\n', text)
    # Remove headers/footers (customize based on your textbooks)
    text = re.sub(r'Chapter \d+.*\n', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

math_text = clean_text(math_text)
history_text = clean_text(history_text)
```

#### For Scanned Books (Images)

```python
from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Process multiple pages
history_text = ""
for i in range(1, 100):  # Adjust range based on number of pages
    page_text = extract_text_from_image(f"history_page{i}.jpg")
    history_text += page_text
```

#### Segmenting into Manageable Units

After extraction, segment the text into logical units for easier processing:

```python
def segment_text(text):
    # Split by paragraphs (double newlines)
    sections = text.split("\n\n")
    # Filter out very short sections (likely headers, page numbers, etc.)
    sections = [s for s in sections if len(s.split()) > 15]
    return sections

math_sections = segment_text(math_text)
history_sections = segment_text(history_text)
```

### 2. Aligning and Cross-Pollinating Content

This is the core of our approach. We need to find meaningful connections between content in different domains.

#### Method 1: Entity-Based Alignment

Find sections that mention the same entities (people, places, concepts) across domains:

```python
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_lg")

def extract_key_entities(sections):
    entities = {}
    for i, section in enumerate(sections):
        doc = nlp(section)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART"]:
                if ent.text not in entities:
                    entities[ent.text] = []
                entities[ent.text].append(i)
    return entities

# Extract entities from both domains
math_entities = extract_key_entities(math_sections)
history_entities = extract_key_entities(history_sections)

# Find overlapping entities
common_entities = set(math_entities.keys()) & set(history_entities.keys())

# Create paired sections based on common entities
entity_based_pairs = []
for entity in common_entities:
    for math_idx in math_entities[entity]:
        for history_idx in history_entities[entity]:
            entity_based_pairs.append({
                "math_section": math_sections[math_idx],
                "history_section": history_sections[history_idx],
                "linking_entity": entity
            })
```

#### Method 2: Semantic Similarity Matching

Even when specific entities don't match, we can find conceptually similar passages:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Create TF-IDF vectors for all sections
all_sections = math_sections + history_sections
vectorizer = TfidfVectorizer(max_df=0.8, stop_words='english')
tfidf = vectorizer.fit_transform(all_sections)

# Split vectors by domain
math_vecs = tfidf[:len(math_sections)]
history_vecs = tfidf[len(math_sections):]

# Find similar sections across domains
similarity_based_pairs = []
similarity_threshold = 0.1  # Adjust based on your needs

for i, math_vec in enumerate(math_vecs):
    # Calculate similarity between this math section and all history sections
    similarities = (history_vecs * math_vec.T).toarray().flatten()
    # Find top matches
    top_indices = np.argsort(similarities)[-3:]  # Get top 3 matches
    
    for idx in top_indices:
        sim_score = similarities[idx]
        if sim_score >= similarity_threshold:
            similarity_based_pairs.append({
                "math_section": math_sections[i],
                "history_section": history_sections[idx],
                "similarity_score": sim_score
            })
```

#### Method 3: Using Advanced Embeddings

For more sophisticated semantic matching, use transformer-based embeddings:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
math_embeddings = model.encode(math_sections)
history_embeddings = model.encode(history_sections)

# Find similar sections
transformer_based_pairs = []
for i, math_emb in enumerate(math_embeddings):
    # Calculate similarities
    similarities = cosine_similarity([math_emb], history_embeddings)[0]
    # Find top matches
    top_indices = np.argsort(similarities)[-5:]  # Top 5 matches
    
    for idx in reversed(top_indices):
        sim_score = similarities[idx]
        if sim_score >= 0.5:  # Higher threshold for better quality
            transformer_based_pairs.append({
                "math_section": math_sections[i],
                "history_section": history_sections[idx],
                "similarity_score": sim_score
            })
```

### 3. Constructing the Cross-Pollinated Dataset

Now that we have paired sections, we need to format them for training:

#### Format 1: Combined Expository Text

```python
def create_combined_text(math_section, history_section, linking_term=None):
    if linking_term:
        connector = f"The concept of {linking_term} appears in both mathematics and history. "
    else:
        connector = "This concept has interesting parallels in mathematics and history. "
    
    combined = f"In mathematics: {math_section}\n\n{connector}\n\nIn history: {history_section}"
    return combined

# Create combined texts from our pairs
dataset_entries = []

# From entity-based pairs
for pair in entity_based_pairs:
    combined = create_combined_text(
        pair["math_section"], 
        pair["history_section"],
        pair["linking_entity"]
    )
    dataset_entries.append(combined)

# From similarity-based pairs
for pair in similarity_based_pairs:
    combined = create_combined_text(
        pair["math_section"], 
        pair["history_section"]
    )
    dataset_entries.append(combined)
```

#### Format 2: Question-Answer Pairs

```python
def create_qa_pairs(math_section, history_section, linking_term=None):
    if linking_term:
        question = f"Explain the significance of '{linking_term}' in both mathematics and history."
    else:
        question = "How might these concepts from different domains relate to each other?"
    
    answer = f"In mathematics: {math_section}\n\nIn history: {history_sections}\n\nThese concepts relate through their shared patterns of {linking_term or 'structure and development'}."
    
    return {"instruction": question, "response": answer}

# Create QA pairs
qa_dataset = []
for pair in entity_based_pairs[:100]:  # Limit to first 100 for example
    qa_pair = create_qa_pairs(
        pair["math_section"],
        pair["history_section"],
        pair["linking_entity"]
    )
    qa_dataset.append(qa_pair)
```

#### Format 3: Simulated Dialogues

```python
def create_dialogue(math_section, history_section, linking_term=None):
    if linking_term:
        intro = f"A mathematician and historian discuss the concept of {linking_term}."
    else:
        intro = "A mathematician and historian discuss connections between their fields."
    
    dialogue = f"{intro}\n\nMathematician: {math_section}\n\nHistorian: Interestingly, we see similar patterns in history. {history_section}\n\nMathematician: That's fascinating! The parallel between these concepts shows how knowledge transcends disciplinary boundaries."
    
    return dialogue

# Create dialogues
dialogue_dataset = []
for pair in transformer_based_pairs[:50]:  # Limit to first 50 for example
    dialogue = create_dialogue(
        pair["math_section"],
        pair["history_section"]
    )
    dialogue_dataset.append(dialogue)
```

### 4. Balancing and Finalizing the Dataset

To ensure a well-rounded dataset:

```python
# Combine all formats
all_entries = dataset_entries + [item["instruction"] + "\n\n" + item["response"] for item in qa_dataset] + dialogue_dataset

# Add some standalone domain-specific entries for balance
all_entries.extend(math_sections[:100])  # Add 100 pure math sections
all_entries.extend(history_sections[:100])  # Add 100 pure history sections

# Shuffle the dataset
import random
random.shuffle(all_entries)

# Save to Hugging Face dataset format
from datasets import Dataset
dataset = Dataset.from_dict({"text": all_entries})

# Preview a few examples
print(dataset[:3]["text"])

# Save the dataset
dataset.save_to_disk("cross_pollinated_dataset")

# Optionally push to Hugging Face Hub
# dataset.push_to_hub("username/cross-pollinated-sft-dataset")
```

## Best Practices for Effective Cross-Pollination

When building your cross-pollinated dataset, keep these guidelines in mind:

1. **Maintain context clarity**: Provide clear signals when switching between domains to avoid confusing the model.

2. **Quality over quantity**: Focus on meaningful connections rather than forcing tenuous links.

3. **Balance domain representation**: Ensure roughly equal representation of all domains in your final dataset.

4. **Preserve factual accuracy**: Be careful not to distort facts when creating analogies or connections.

5. **Include epistemological content**: Add meta-content about how knowledge is formed in different fields.

6. **Use diverse formats**: Mix standalone domain content, cross-domain pairs, QA formats, and dialogues.

7. **Intermix domains during training**: Don't segregate domains; shuffle examples to prevent the model from partitioning knowledge.

## Evaluating Cross-Domain Understanding

After fine-tuning, test your model with prompts that require cross-domain reasoning:

- "Draw an analogy between calculus and the Industrial Revolution."
- "How might Euler's identity relate to Renaissance art?"
- "What mathematical principles could help understand the rise and fall of ancient civilizations?"

A model trained on well-structured cross-pollinated data should produce insightful, linked answers that demonstrate it has learned to connect knowledge across domains.

## Conclusion

By deliberately cross-pollinating content from diverse textbooks, we can create SFT datasets that teach LLMs not just to memorize facts, but to understand the interconnected nature of knowledge. This approach encourages models to develop a more holistic understanding of information, enabling them to make novel connections and generate more insightful responses.

The code provided in this post offers a starting point for creating your own cross-pollinated dataset. The specific domains can be expanded beyond mathematics and history to include science, literature, philosophy, or any other fields you wish to connect. The key is to create meaningful bridges between domains that encourage the model to develop a unified understanding of knowledge.

By training models to see connections across traditionally separate domains, we move closer to AI systems that can reason more like humans do—drawing from diverse knowledge sources to generate novel insights and solve complex problems.

## References

1. Gao et al., "The Pile: An 800GB Dataset of Diverse Text for Language Modeling." (2020)
2. SciPhi Project, "Textbooks are All You Need – A Library of Alexandria for LLMs." (2023)
3. Li et al., "CulturePark: Boosting Cross-cultural Understanding in LLMs." (2024)
4. Yuan et al., "ANALOGYKB: Unlocking Analogical Reasoning of LMs with a Million-scale Knowledge Base." (2024)
