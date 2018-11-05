# PPIS
This README is for national user,we will describe our algorithm to predict the interaction sites of protein and protein.
[中文版说明](https://github.com/LionsChen/PPIS/blob/master/中文说明.md)
## data set
  We download about 20000 complex protein from PDB in PDB format, and the complex protein id listed in the pdblist.txt, you can use the Biopython API to download all the protein in pdblist.txt.All of the complex protein are selected if its number of entities are more than 2, and the entity we select protein. You can also use PDB RESTful search service to build your own dataset to train.
