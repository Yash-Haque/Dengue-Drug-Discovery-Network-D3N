# Dengue-Drug-Discovery-Network-D3N
A repository aimed to construct a deep learning network using pytorch, in hopes to find a suitable candidate for Dengue Vaccine.

## Purpose of the Data-Mining branch
The following branch of the project houses one of the most important components of the network. That is, the webscraper used to extract unstructured text from the pubmed database.

The following branch is mainly created to separately store and record the progress of the network's Webscraper.

The branch comprises of the main folder "Webscraper", which contains the following python files: "**main.py**", "**html-parser.py**", "**extractor.py**", "**abstract_extractor.py**", "**file_merger.py**".

Further cleaning of the folder needs to be done in order for appropriate presentation. For now, for demonstration of the project's progress, the files have been uploaded in their raw state.

*In order to clean up the branch, the following features must be added/changed in this branch:*
- [ ] Create a suitable Extractor that combines the strengths of all the functions within the shared python files.
- [ ] Create a separate dedicated folder within Webscraper to store all the relevant files relevant to data extraction
- [ ] Find a better way to extract/mine data from PubMed. (i.e. using PyMed)
