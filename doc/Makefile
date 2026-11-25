.PHONY: pdf parker-pdf parker-equations nlte-pdf nlte-equations coupled-pdf coupled-equations cloudy-pdf cloudy-equations atom-pdf atom-equations spectrum-pdf spectrum-equations radiation-pdf stellar-pdf atmosphere-pdf solar-pdf escape-pdf clean

escape-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build escape_probability.tex
	cp build/escape_probability.pdf escape_probability.pdf

solar-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build solar_atmosphere.tex
	cp build/solar_atmosphere.pdf solar_atmosphere.pdf

atmosphere-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build coupled_atmosphere.tex
	cp build/coupled_atmosphere.pdf coupled_atmosphere.pdf

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build model.tex
	cp build/model.pdf model.pdf

spectrum-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build nlte_spectrum.tex
	cp build/nlte_spectrum.pdf nlte_spectrum.pdf

# Preserve the historical appendix pinned by the original formulation review.
spectrum-equations:
	mkdir -p build
	git show 1f4968f685a3251435331bfc6e791c7c9c838311:mceg/contracts/escape.nlte_spectrum.yaml > build/nlte_spectrum_contract.yaml
	python -m mceg equations build/nlte_spectrum_contract.yaml --output nlte_spectrum_equations.tex

radiation-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build radiation_closure.tex
	cp build/radiation_closure.pdf radiation_closure.pdf

stellar-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build stellar_radiation.tex
	cp build/stellar_radiation.pdf stellar_radiation.pdf

atom-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build atom_convergence.tex
	cp build/atom_convergence.pdf atom_convergence.pdf

atom-equations:
	python -m mceg equations ../mceg/contracts/escape.atom_convergence.yaml --output atom_convergence_equations.tex

parker-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build parker_wind.tex
	cp build/parker_wind.pdf parker_wind.pdf

parker-equations:
	python -m mceg equations ../mceg/contracts/escape.parker_wind.yaml --output parker_wind_equations.tex

nlte-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build hydrogen_nlte.tex
	cp build/hydrogen_nlte.pdf hydrogen_nlte.pdf

# Run only in the separate authoring environment with MCEG installed.
nlte-equations:
	python -m mceg equations ../mceg/contracts/escape.hydrogen_nlte.yaml --output hydrogen_nlte_equations.tex

coupled-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build coupled_model.tex
	cp build/coupled_model.pdf coupled_model.pdf

coupled-equations:
	python -m mceg equations ../mceg/contracts/escape.coupled_hydrogen.yaml --output coupled_model_equations.tex

clean:
	rm -rf build

cloudy-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build cloudy_investigation.tex
	cp build/cloudy_investigation.pdf cloudy_investigation.pdf

cloudy-equations:
	python -m mceg equations ../mceg/contracts/escape.cloudy_investigation.yaml --output cloudy_investigation_equations.tex

# Current branch-center supplement; historical spectrum artifacts stay unchanged.
.PHONY: spectrum-centers-equations spectrum-centers-pdf
spectrum-centers-equations:
	python -c "from pathlib import Path; from mceg.validator import load_contract; from mceg.equations import render_equations; c = load_contract(Path('../mceg/contracts/escape.nlte_spectrum.yaml')); e = [x for x in c.equations if (x.id, x.revision) == ('branch_profiles', '2')]; assert len(e) == 1; Path('observer_branch_centers_equations.tex').write_text(render_equations(c.model_copy(update={'equations': e})))"

spectrum-centers-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build observer_branch_centers.tex
	cp build/observer_branch_centers.pdf observer_branch_centers.pdf

.PHONY: time-longitude-pdf
time-longitude-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build time_longitude.tex
	cp build/time_longitude.pdf time_longitude.pdf

.PHONY: time-gp-pdf
time-gp-pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build time_gp.tex
	cp build/time_gp.pdf time_gp.pdf
