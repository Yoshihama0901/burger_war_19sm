SHELL=bash

.PHONY: all
all: circle-20190813162032-2.csv circle-20190813162032-est.csv model-u.txt model-v.txt model-theta.txt

circle-20190813162032-2.csv: circle-20190813162032.csv
	python circle.py $< > $@

.PHONY: plot-u plot-v plot-theta plot-sin-theta plot2
plot-u plot-v plot-theta plot-sin-theta plot2: circle-20190813162032-2.csv
	python $@.py $<

model-u.txt model-v.txt model-theta.txt model-sin-theta.txt: circle-20190813162032-2.csv
	python $(basename $@).py $< |& tee $@

circle-20190813162032-est.csv: circle-20190813162032-2.csv
	python estimate.py $< > $@

.PHONY: clean
clean:
	rm -f circle-20190813162032-2.csv
	rm -f circle-20190813162032-est.csv
	rm -f model-*.txt
