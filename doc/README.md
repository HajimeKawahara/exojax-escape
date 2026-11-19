# Mathematical model definitions

Read [model.pdf](model.pdf); edit [model.tex](model.tex). The English document
separates implemented equations from the proposed multilevel NLTE problem and
choices still needed to close that problem. It includes units, assumptions,
boundary conditions, implicit differentiation and references.

Build with a TeX installation providing `latexmk` and `pdflatex`:

```bash
make -C doc pdf
```

The PDF is committed for reading without LaTeX. Auxiliary files stay in the
ignored `doc/build/` directory; `make -C doc clean` removes only that directory.
TeX and MCEG are external authoring tools, not Python package dependencies.

## Record a model change

1. Update the equations, assumptions, implementation status and document revision
   in the TeX source. Preserve equation labels, or introduce a new label when a
   definition changes materially.
2. Rebuild the PDF and inspect equations, page breaks and references. Check
   `doc/build/model.log` for unresolved references and overflowing content.
3. Review the `population_formulation` Claim and `formulation` Evidence in the
   [MCEG contract](../mceg/contracts/escape.hydrogen_populations.yaml). Record new
   document hashes and a scoped assessment after review. Preserve older records
   in version history; do not relabel planned equations as implemented.
4. If code or physics changes, update the model inventory and run the relevant
   scientific checks. A new document does not renew historical test evidence.

In a separate MCEG environment, run from the repository root:

```bash
python -m mceg check mceg/contracts/escape.hydrogen_populations.yaml \
  --assessment formulation_review --root .
```

MCEG tracks evidence paths, prose locations and hashes. It does not parse TeX,
check equation labels, prove equivalence to code, or generate this PDF. The
initial formulation review covers only POP-1 through POP-5 and their limiting
identities. Other document sections and physical applicability need their own
reviews and evidence.
