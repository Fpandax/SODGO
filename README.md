# SODGO: Sub-ontology driven disentangled gene ontology embedding model

### Fig.1 GO Graph
![GO Graph](./figure/Fig.1%20GO%20Graph.png)

### Fig.2 SODGO
![SODGO](./figure/Fig.2%20SODGO.png)

### Dataset
Please download it and put it in the data folder of the corresponding directory

| Dataset    | Links |--- |
|------------| ------- | -- | 
| basic      | https://drive.google.com/drive/folders/1ijrEUiMy4StYU0wBmYB8MlIgo9Kut8yt?usp=drive_link | Model datasets | 
| experiment | https://drive.google.com/drive/folders/147QG-36i6yXofoSzXMPXorcfPudGLLfd?usp=drive_link | Embed the analysis dataset | 
| experiment | https://drive.google.com/drive/folders/1FIMibwHAQh4JqNlonB4oafRUskNPsPEm?usp=drive_link | Downstream task datasets | 

### Folder structure
- The `data_load` folder process go.obo generates triples and dataset partitions.
- The `basic` folder contains the main model SODGO, as described in Section 2 of the paper.
- The `experiment` folder is for embedding analysis, corresponding to Sections 3.1 and 3.2 of the paper.
- The `downstream` folder contains the downstream tasks, as described in Section 3.3 of the paper.

### Run

#### Obtaining embeddings
To obtain the embeddings, run the `SODGOrun.py` file in the `basic` folder.
```bash
python SODGOrun.py
```


#### Embedding analysis
- The `attention_analyse` and `distance_analyse` folder, specifically, is used for **Section 3.1: Weight Distribution and Embedding Information Variation of the SODGO Model**, and contains the `analyze.ipynb` notebook.
- The `embed_analyse` folder is used for **Section 3.2: Performance Evaluation of GO Representation Learning**, and contains an `embedding` subfolder with an `analyze.ipynb` file for evaluating GO representation learning performance.
