import ast

class ASTDescriber(ast.NodeVisitor):
    def __init__(self):
        self.structure = []

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            self.structure.append(("assign", target, node.value))
        self.generic_visit(node.value)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            target = node.target.id
            op = type(node.op).__name__
            self.structure.append(("aug_assign", target, op, node.value))
        self.generic_visit(node.value)

    def describe(self, node):
        self.visit(node)
        return self.structure