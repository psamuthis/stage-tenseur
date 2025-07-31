import ast
from Describer import ASTDescriber
from translator import Translator

with open("simple_program.py") as f:
    tree = ast.parse(f.read())

describer = ASTDescriber()
describer.describe(tree)

translator = Translator(describer.structure)
translator.translate()

with open("dump.txt", "w") as f:
    for line in translator.output:
        f.write(line + "\n")