import ast

def describe_node(node, output, depth=0, role=""):
    indent = "  " * depth

    if isinstance(node, ast.Assign):
        output.append(f"{indent}Assign to:")
        for t in node.targets:
            describe_node(t, output, depth + 1)
        output.append(f"{indent}Value:")
        describe_node(node.value, output, depth + 1)

    elif isinstance(node, ast.AugAssign):
        output.append(f"{indent}AugAssign: {type(node.op).__name__}")
        output.append(f"{indent}Target:")
        describe_node(node.target, output, depth + 1)
        output.append(f"{indent}Value:")
        describe_node(node.value, output, depth + 1)

    elif isinstance(node, ast.BinOp):
        output.append(f"{indent}BinOp: {type(node.op).__name__}")
        output.append(f"{indent}Left:")
        describe_node(node.left, output, depth + 1)
        output.append(f"{indent}Right:")
        describe_node(node.right, output, depth + 1)

    elif isinstance(node, ast.Name):
        output.append(f"{indent}{role} Name: {node.id}")

    elif isinstance(node, ast.Constant):
        output.append(f"{indent}Constant: {node.value}")

    elif isinstance(node, ast.Expr):
        describe_node(node.value, output, depth)

    elif isinstance(node, ast.Call):
        describe_node(node.func, output, depth + 1, role="Func")
        for arg in node.args:
            describe_node(arg, output, depth + 2, role="Arg")

    else:
        output.append(f"{indent}{type(node).__name__}: {ast.dump(node)}")

with open("simple_program.py") as f:
    tree = ast.parse(f.read())

output = []
for node in ast.walk(tree):
    if isinstance(node, (ast.Assign, ast.AugAssign)):
        describe_node(node, output)

for op in output:
    print(f"{op}")