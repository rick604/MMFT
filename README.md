# Multi-source Manifold Feature Transfer Learning with Domain Selection for Brain-Computer Interfaces
In this paper, we propose a multi-source manifold feature transfer learning  (MMFT) framework to classify multi-source EEG signals. Firstly, the tangent space feature is extracted from a symmetric positive definite (SPD) manifold to utilize  the covariance matrices of EEG trials. Taking the advantage of geometric properties of Grassmann manifold, the marginal probability distribution shift is minimized. Then the manifold features of different source domains are transferred to the target domain via a voting mechanism. The classification model is trained by summarizing the structural risk minimization (SRM) over source domains and conditional alignment. Furthermore, a weighted MMFT (w-MMFT) algorithm is proposed to cope with class imbalance situations. To overcome negative transfer for a large number of source domains, a domain selection approach called label similarity analysis (LSA) is proposed for MMFT, termed as LSA-MMFT, which is helpful to score the transferability between different source and target domains when it is integrated with MMFT. Experimental results on five datasets demonstrate that MMFT has achieved superior performance in classification accuracy and computational efficiency compared to state-of-the-art methods, with high classification accuracy achieved by the LSA-MMFT from selected source domains.

# Running the code
Due to the limitation of upload file size and total capacity, we only upload MI2 and RSVP datasets.   
The rest datasets used in our experiment can be downloaded   
Here    : https://pan.baidu.com/s/1FG1pDzFlW8b596MUIdXP3g.  Extraction code：MMFT .    
Or here : https://drive.google.com/file/d/1brbkeLITArv_cBx6W2RquQVAl2Qq4PXo/view?usp=sharing . 

The implementation of MMFT approach:  
MMFT and MMFT_RSVP  

The implementation of LSA-MMFT approach:  
LSA_MMFT and LSA_MMFT_RSVP  

Run these demo files in MATLAB could show the performance similar(should be consistent) to the results in our paper.   
demo_MI    :  Obtain the classification results of MI1-MI4 datasets.  
demo_RSVP  :  Obtain the classification results of RSVP dataset.  
demo_ERN   :  Obtain the classification results of ERN dataset.  
(Using MMFT and MMFT_RSVP)

Run these files in MATLAB could show the performance similar(should be consistent) to the results in our paper.   
Domain_selection_MI   ： Obtain the domian selection results of MI1-MI4 datasets.  
Domain_selection_MI   ： Obtain the domian selection results of RSVP dataset.  
Domain_selection_MI   ： Obtain the domian selection results of ERN dataset.  
(Using LSA_MMFT and LSA_MMFT_RSVP)
