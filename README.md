# MATT-Multiple-Attributes-Trajectory-Topic-Modeling
All of the codes are implemented based on the GibbsLDApy. More details see https://github.com/jasperyang/GibbsLDApy.

How to use

python LDA.py -est -alpha 0.5 -beta 0.1 -ntopics 2,2 -niters 100 -savestep 100 -twords 20 -dfile documents.txt -dir run_data/ -model mymodel

'-ntopics 2,2' the parameter means that the MATT model has 2 attributes, and the number of topic is 2 in the first attribute, the second is also 2. if there are 3 attributes, the -ntopics should be '3,4,5'.
