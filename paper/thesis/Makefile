MAIN=thesis

all: dvi

dvi: $(MAIN).dvi
pdf: $(MAIN).pdf

$(MAIN).dvi: $(MAIN).tex
	platex $<
	platex $<

$(MAIN).pdf: $(MAIN).dvi
	dvipdfmx -d 5 $<

clean:
	del *.aux *.log *.toc *.bmc *.pbm *.dvi *.pdf

$(MAIN).dvi: $(wildcard thesis*.tex)
