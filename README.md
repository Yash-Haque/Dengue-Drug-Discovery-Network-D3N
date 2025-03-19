# Dengue Interaction Knowledge Graph (DIKG)

This project constructs a **Dengue Interaction Knowledge Graph (DIKG)** using research papers from PubMed. The goal is to identify important biological entities and their relationships to help researchers understand interactions related to the Dengue virus.

## **Project Overview**
The project has three main steps:

1. **Identifying Key Terms (Named Entity Recognition - NER)**
2. **Finding Relationships Between Terms (Relation Extraction)**
3. **Building a Knowledge Graph for Visualization**

## **Data Used**
We collected **10,000 research abstracts** from PubMed related to Dengue research. The data is structured as:

- **unsegmented_unfiltered.csv**: Metadata including titles, abstracts, and publication details.
- **segmented_unfiltered.csv**: A structured dataset where abstracts are split into sentences for better analysis.

## **How It Works**
### **1. Identifying Key Terms (NER)**
We use an encoder Transformers (BioBERT) model to detect scientific terms like **Proteins, Genes, DNA, RNA, Cell Types, and Cell Lines** in research papers.

- **Model Used**: BioBERT, a language model trained for biomedical research.
- **Process**:
  - The AI scans each sentence and finds important terms.
  - It assigns a category to each detected term (e.g., "Protein" or "Gene").

### **2. Finding Relationships Between Terms**
Once key terms are identified, we determine how they are related. For example, does a specific protein interact with a gene?

- **Method 1: Direct Classification**
  - AI checks the words around two entities to determine their relationship.
  - It labels interactions as **Enzymatic, Structural, or No Relation**.

- **Method 2: Using a Large Language Model (LLM)**
  - Instead of direct classification, AI is given structured prompts to predict the relationship.
  - This method improves accuracy and provides better context.

### **3. Building the Knowledge Graph**
With extracted terms and relationships, we construct a **Knowledge Graph** where:

- **Nodes (Points)**: Represent Proteins, Genes, Diseases, etc.
- **Edges (Lines)**: Show the relationships between them.
- **Visualization**: We use **Neo4j**, a graph database, to explore the network and see important patterns.

## **How to Use**
### **Requirements**
- Python 3.8 or later
- Install required libraries:
  ```bash
  pip install pandas numpy scikit-learn torch transformers nltk neo4j networkx
  ```

### **Steps to Run**
1. **Find Key Terms (NER)**
   ```bash
   python ner_extraction.py --input data/pubmed_abstracts.json
   ```
2. **Identify Relationships**
   ```bash
   python relation_extraction.py --input extracted_entities.json
   ```
3. **Build the Knowledge Graph**
   ```bash
   python build_kg.py --entities extracted_entities.json --relations extracted_relations.json
   ```

<!-- ## **Results**
- **NER Performance:** BioBERT accurately detects biomedical terms.
- **Relation Extraction:** AI effectively classifies relationships between terms.
- **Knowledge Graph Analysis:** The graph provides a clear and structured view of Dengue-related interactions. -->

## **Contributing**
We welcome contributions! Follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request.

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.

---

## **Acknowledgments**
- PubMed for research papers
- Hugging Face for AI models
- Neo4j for Knowledge Graph visualization

