This code shows the GCDL implementation.
The article is available here http://www.ee.iisc.ac.in/people/faculty/soma.biswas/pdf/mandal_gcdl_tip2016.pdf

The code is setup to give results for the CUHK experiments 
running it you are expected to get aroung 98% accuracy.

Please use the option=2 and option=4 to get the results for the different 
versions of GCDL implementation i.e., GCDL1 and GCDL2.
*****************************************************************************
You may select option=1 and option=3 also. These were some modifications that
I had made on the intra covariance matrices Cxx and Cyy but did not use it on
my TIP work.
*****************************************************************************

Please download the cuhk dataset from the link provided here
http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html
For the CUHK dataset the features used is the intensity profile

For running the code also it is necessary to re-compile the SPAMS toolbox
found at http://spams-devel.gforge.inria.fr/
The code uses the mexTrainDL and mexLasso functions in the GCDL implementation

*****************************************************************************
For implementing this method on other datasets kindly please follow the paper.
You need to set the parameters appropriately to get the best results.
*****************************************************************************
