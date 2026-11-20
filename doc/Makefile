.PHONY: pdf nlte-pdf nlte-equations clean

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build model.tex
	cp build/model.pdf model.pdf

nlte-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build hydrogen_nlte.tex
	cp build/hydrogen_nlte.pdf hydrogen_nlte.pdf

# Run only in the separate authoring environment with MCEG installed.
nlte-equations:
	python -m mceg equations ../mceg/contracts/escape.hydrogen_nlte.yaml --output hydrogen_nlte_equations.tex

clean:
	rm -rf build
