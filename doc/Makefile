.PHONY: pdf clean

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build model.tex
	cp build/model.pdf model.pdf

clean:
	rm -rf build
