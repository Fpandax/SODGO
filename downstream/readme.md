#  Downstream task
Replace the data to the corresponding location of the original author model input GO embedding

## SODGO+
1) obtain the gene embeddings and construct the GGI
 
![SODGO](./SODGO+/Fig.7%20SODGO+.png)
```shell
python run.py
```
2) run the downstream task(GEARS)


## PA(Protein annotation prediction) 
1) Please download the PO2GO model
2) Please download it and put it in the data folder of the corresponding directory

## DTA(Drug-target interaction prediction)
1) Please download SISDTA model
2) Please download it and put it in the data folder of the corresponding directory
3) draw_explain.ipynb corresponds to Subsection 3.4 of the paper 

