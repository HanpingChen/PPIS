# PPIS
This README is for national user,we will describe our algorithm to predict the interaction sites of protein and protein.
[中文版说明](https://github.com/LionsChen/PPIS/blob/master/中文说明.md)
We will provied web server for the prediction of PPIS soon...
## dependencies
- [numpy](http://www.numpy.org)
- [tensorflow](https://tensorflow.google.cn)
- [Biopython](https://biopython.org)
- [tflearn](http://tflearn.org)
- [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/)
- [...]()
## data set
  We download about 20000 complex protein from PDB in PDB format, and the complex protein id listed in the pdblist.txt, you can use the Biopython API to download all the protein in ```pdblist.txt```. All of the complex protein are selected if its number of entities are more than 2, and the entity we select protein. You can also use PDB RESTful search service to build your own dataset to train.
  
we need to use [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/) to calculate the interaction site of a complex protein, if you want to train your own model, you must download DSSP program and comopile it. We use Biopython to call DSSP to calculate the sites and then we write the sites in json format.If you do not want to compile it yourself,you can use the training data we provied in json format.
