import ast

class Translator:
    def __init__(self, structure):
        self.structure = structure
        self.output = []

    def translate(self):
        for stmt in self.structure:
            if stmt[0] == "assign":
                var, expr = stmt[1], self.translate_expr(stmt[2])
                self.output.append(f"{expr}")

            elif stmt[0] == "aug_assign":
                var, op, expr_node = stmt[1], stmt[2].lower(), stmt[3]
                expr = self.translate_expr(expr_node)
                self.output.append(f"{op} {var} {expr}")

        return self.output

    def translate_expr(self, node):
        if isinstance(node, ast.BinOp):
            op = type(node.op).__name__.lower()
            left = self.translate_expr(node.left)
            right = self.translate_expr(node.right)
            return f"({op} {left} {right})"
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Call):
            func = self.translate_expr(node.func)
            args = " ".join(self.translate_expr(arg) for arg in node.args)
            return f"({func} {args})"
        else:
            return f"<unsupported {type(node).__name__}>"