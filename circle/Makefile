.PHONY: all
all: circle-20190813162032-2.csv

circle-20190813162032-2.csv: circle-20190813162032.csv
	python circle.py $< > $@

.PHONY: clean
clean:
	rm -f circle-20190813162032-2.csv
