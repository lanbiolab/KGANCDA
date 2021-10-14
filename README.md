# KGANCDA
KGANCDA is effective to predict associations between circRNA and cancer, which is based on knowledge graph attention network. The detail of the paper will be released in this page after the paper accepted.

Author: Wei Lan. Yi Dong. Qingfeng Chen. Jin Liu. Yi Pan. Yi-Ping Phoebe Chen.

# Environment Requirement
+ tensorflow == 1.12.0
+ numpy == 1.15.4
+ scipy == 1.1.0
+ sklearn == 0.20.0

# Dataset
## Dataset 1 (Cancer)
including those data files:
+ cancer_list.csv record cancer name and id.
+ circRNA_list.csv record circRNA name and id.
+ lncRNA_list.csv record lncRNA name and id.
+ miRNA_list.csv record miRNA name and id.
+ circRNA-cancer.csv record associations between circRNAs and cancers.
+ circRNA-miRNA.csv record associations between circRNAs and miRNAs.
+ lncRNA-cancer.txt record associations between lncRNAs and cancers.
+ lncRNA-miRNA.txt record associations between lncRNAs and miRNAs.
+ miRNA-cancer.csv record associations between miRNAs and cancers.

## Dataset 2 (Non-cancer)
including those data files:
+ circ_list.csv record circRNA name and id.
+ dis_list.csv record disease name and id.
+ lnc_list.csv record lncRNA name and id.
+ mir_list.csv record miRNA name and id.
+ circrna-disease.txt record associations between circRNAs and diseases.
+ circrna-mirna.txt record associations between circRNAs and miRNAs.
+ lncrna-disease.txt record associations between lncRNAs and diseases.
+ lncrna-mirna.txt record associations between lncRNAs and miRNAs.
+ mirna-disease.txt record associations between miRNAs and diseases.

# Model
+ KGANCDA.py: the core model proposed in the paper.
+ main.py: the main program in the project. Run main.py to generate the embeddings of all nodes in the network
+ ExtractFeature.py: After obtaining the embedding of all nodes, run this file to validate the effect of the model in 5-fold cross validation.

# Compare_models
+ There are 6 the state-of-the-art models including: CD-LNLP, DMFCDA, DWNN-RLS, GCNCDA, KATZHCDA, RWR, which are compared under the same experiment settings.

# Question
+ If you have any problems or find mistakes in this code, please contact with me: 
Yi Dong: dongyi@st.gxu.edu.cn 
