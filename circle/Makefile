.PHONY: all
all: circle-20190813162032-2.csv circle-20190813162032-est.csv

circle-20190813162032-2.csv: circle-20190813162032.csv
	python circle.py $< > $@

.PHONY: plot-u plot-v model-u model-v
plot-u plot-v model-u model-v: circle-20190813162032-2.csv
	python $@.py $<

circle-20190813162032-est.csv: circle-20190813162032-2.csv
	python estimate.py $< > $@

.PHONY: clean
clean:
	rm -f circle-20190813162032-2.csv
	rm -f circle-20190813162032-est.csv
