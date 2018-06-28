# wordcept
A python toolkit for machine learning on Chinese words.

## Word Segmentation Tool: `dartfrog.py`

To train the word segmentation tool on a corpus of segmented text, run:

    dartfrog.py --fit TRAIN-DATA-FILE

To process raw text and produce segmented text, run:

    dartfrog.py --transform INPUT-FILE OUTPUT-FILE

## Performance

| Dataset: [SIGHAN Bakeoff 2005](http://sighan.cs.uchicago.edu/bakeoff2005/) | F1 | Recall | OOV Recall |
| --- | --- | --- | --- |
| AS | 0.928 | 0.935 | 0.390 |
| CityU | 0.911 | 0.927 | 0.388 |
| MSRA | 0.946 | 0.963 | 0.205 |
| PKU | 0.924 | 0.932 | 0.499 |
