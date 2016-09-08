# NLPCC-2016-WordSim
Code for NLPCC2016 Chinese Word Similarity Task

This is our solution for [NLPCC2016 shared task: Chinese Word Similarity Measurement](http://tcci.ccf.org.cn/conference/2016/pages/page05_CFPTasks.html).

>This task provides a dataset of Chinese word similarity to evaluate and compare different semantic measures of lexical similarity, including 500 word pairs and their similarity scores.

We proposes a novel framework for measuring the Chinese word similarity by combining word embedding and Tongyiic Cilin. We also utilize retrieval techniques to extend the contexts of word pairs and calculate the similarity scores to weakly supervise the selection of a better result. 

The official results show that our solution of the team "DLUT_NLPer" achieves the 2nd place in the Chinese Lexical Similarity Computation (CLSC) shared task with 0.457 Spearman rank correlation coeffcient.

After the submission, we boost the embedding model by merging an English model into the Chinese one and learning
the co-occurrence sequence via LSTM networks. Our final result is 0.541(Spearman rank correlation coeffcient), which outperform the state-of-the-art performance to the best of our knowledge.

**Team members**: Jiahuan Pei, Cong Zhang

Please feel free to send us emails (*zhangcong002@gmail.com*) if you have any problems.

[1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
[2] Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. Neural computation, 12(10), 2451-2471.
[3] Graves, Alex. Supervised sequence labelling with recurrent neural networks. Vol. 385. Springer, 2012.
[4] Bastien, Frédéric, Lamblin, Pascal, Pascanu, Razvan, Bergstra, James, Goodfellow, Ian, Bergeron, Arnaud, Bouchard, Nicolas, and Bengio, Yoshua. Theano: new features and speed improvements. NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2012.
[5] Bergstra, James, Breuleux, Olivier, Bastien, Frédéric, Lamblin, Pascal, Pascanu, Razvan, Desjardins, Guillaume, Turian, Joseph, Warde-Farley, David, and Bengio, Yoshua. Theano: a CPU and GPU math expression compiler. In Proceedings of the Python for Scientific Computing Conference (SciPy), June 2010.