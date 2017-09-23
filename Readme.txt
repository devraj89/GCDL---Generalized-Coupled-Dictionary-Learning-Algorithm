This code shows the GCDL implementation

The code is setup to give results for the CUHK experiments 
running it you are expected to get aroung 98% accuracy.

Please use the option=2 and option=4 to get the results for the different 
versions of GCDL implementation i.e., GCDL1 and GCDL2.

Please download the cuhk dataset from the link provided here
http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html
For the CUHK dataset the features used is the intensity profile

For running the code also it is necessary to re-compile the SPAMS toolbox
found at http://spams-devel.gforge.inria.fr/
The code uses the mexTrainDL and mexLasso functions in the GCDL implementation