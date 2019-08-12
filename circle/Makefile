.PHONY: all
all: circle-20190812100332.csv

circle-20190812100332.csv: circle-20190812100332.log
	python circle.py $< > $@

.PHONY: clean
clean:
	rm -f circle-20190812100332.csv
