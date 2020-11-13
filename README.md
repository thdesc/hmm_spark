# Hidden Markov Model: Map-Reduce implementation of the Baum-Welch algorithm (EM)

Implementation of the Baum-Welch algorithm (EM) for the estimation of parameters of Hidden Markov Model in a distributed fashion (using PySpark).

Implementation of the transition matrix and emission matrix estimation (forward-backward algorithm) algorithm from the book: Data-Intensive Text Processing with MapReduce(Jimmy Lin and Chris Dyer). It is a map-reduce based approach. Two distinct implementations are provided: one only using python built-in packages and replicating the book pseudo-code and one using the NumPy libraby and some optimizations. See the report for a detailed description.

hmm_python.py: built in python only implementation (no extra package needed). It intents to replicate the map-reduce based implementation from the reference book.

hmm_numpy.py: numpy based, optimized implementation (recusrive forward-backward algorithm).

hmm_report.pdf: report explaining model and implementation. Also contains performance comparison and commentaries on the book point of view.

