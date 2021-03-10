# compchemkit
This is a work in progress package, collecting code snippets, functions, and classes useful for chemoinformatics.

##  The module `filtering`
The class `PainsFilter` can filer SMILES-strings in parallel returning `True` or `False` depending on the presence or absence of PAINS-substructures. Invalid SMILES return
the value `None`.
