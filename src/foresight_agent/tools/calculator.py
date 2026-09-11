import ast
import operator
from typing import Any


class CalculatorTool:
    name = "calculator"
    description = "Basic arithmetic with numbers, +, -, *, /, //, %, **, and parentheses."
    _binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
               ast.Mod: operator.mod, ast.Pow: operator.pow}
    _unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expression = str(arguments.get("expression", ""))
        if len(expression) > 200:
            raise ValueError("Expression is too long.")
        value = self._evaluate(ast.parse(expression, mode="eval").body)
        return {"observation": str(value), "success": True, "confidence": 1.0}

    def _evaluate(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._binary:
            left, right = self._evaluate(node.left), self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large.")
            return self._binary[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary:
            return self._unary[type(node.op)](self._evaluate(node.operand))
        raise ValueError("Unsupported operation.")
