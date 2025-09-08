from z3 import *
import pandas as pd

def aggregate_expr(attn_row, values):
    # Берёт из values[j] тот j, где attn_row[j] == True; если none — fallback на values[0]
    expr = values[0]
    # Проходим в обратном порядке, чтобы учесть ранние индексы в случае совпадений
    for j in reversed(range(len(attn_row))):
        expr = If(attn_row[j], values[j], expr)
    return expr

def build_attention_block(solver, keys, queries, predicate_expr, values, name):
    """
    Построение attention-блока:
    - keys: список элементов (Int или String) для predicate_expr
    - queries: список элементов (Int или String), по которым выбираем
    - predicate_expr: функция (q, k) -> BoolRef, задающая условие match
    - values: список элементов (Int или String) для aggregate
    - name: суффикс имени переменных в Z3
    Возвращает list из N выходов (String или Int), по аналогии с `outs[...]`.
    """
    N = len(keys)
    # матрица Bool переменных attn[i][j]
    attn = [[Bool(f"attn_{name}_{i}_{j}") for j in range(N)] for i in range(N)]
    # флаги: для каждой i, есть ли вообще какой-то match среди keys[j]
    any_match = [Bool(f"any_{name}_{i}") for i in range(N)]

    # Считаем any_match[i] == Or(predicate_expr(queries[i], keys[j]) for j in range(N))
    for i in range(N):
        solver.add(any_match[i] == Or([predicate_expr(queries[i], keys[j]) for j in range(N)]))

    # Определяем тип выходов: Int или String, в зависимости от values
    if values and isinstance(values[0], AstRef) and values[0].sort() == IntSort():
        outputs = [Int(f"attn_{name}_output_{i}") for i in range(N)]
    else:
        outputs = [String(f"attn_{name}_output_{i}") for i in range(N)]

    for i in range(N):
        # ровно один True в строке i
        solver.add(Sum([If(attn[i][j], 1, 0) for j in range(N)]) == 1)

        # для каждого j:
        for j in range(N):
            if j == 0:
                # fallback: attn[i][0] может быть True, если нет ни одного match, или если predicate_expr истинно для (i,0)
                solver.add(Implies(attn[i][0], Or(Not(any_match[i]), predicate_expr(queries[i], keys[0]))))
            else:
                # если attn[i][j] == True, то predicate_expr(queries[i], keys[j]) должно быть True
                solver.add(Implies(attn[i][j], predicate_expr(queries[i], keys[j])))

        # условие closest: если attn[i][k] и predicate_expr(queries[i], keys[j]) истинен,
        # то расстояние |i-k| <= |i-j| для всех j.
        for j in range(N):
            for k in range(N):
                solver.add(Implies(
                    And(attn[i][k], predicate_expr(queries[i], keys[j])),
                    Abs(i - k) <= Abs(i - j)
                ))

        # aggregate: выбираем значение из values по вектору attn[i]
        solver.add(outputs[i] == aggregate_expr(attn[i], values))

    return outputs

def build_mlp_block(solver, positions, tokens, mlp_expr_fn, name):
    """
    Построение MLP-блока: для каждой позиции i создаём Int переменную вывода mlp_{name}_output_{i}
    и constraint: output == mlp_expr_fn(position, token_at_position).
    """
    N = len(tokens)
    outputs = [Int(f"mlp_{name}_output_{i}") for i in range(N)]
    for i in range(N):
        solver.add(outputs[i] == mlp_expr_fn(positions[i], tokens[i]))
    return outputs


def predicate_0_0_expr(position, token):
    return Or(
        And(position == IntVal(0), token == StringVal('3')),
        And(Or(position == IntVal(1), position == IntVal(2)), token == StringVal('1')),
        And(Or(position == IntVal(3), position == IntVal(4), position == IntVal(5), position == IntVal(6)), token == StringVal('4')),
        And(position == IntVal(7), token == StringVal('0')),
    )

def predicate_0_1_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(5), q_position == IntVal(7)), k_position == IntVal(6)),
        And(q_position == IntVal(1), k_position == IntVal(2)),
        And(Or(q_position == IntVal(2), q_position == IntVal(3)), k_position == IntVal(4)),
        And(q_position == IntVal(4), k_position == IntVal(5)),
        And(q_position == IntVal(6), k_position == IntVal(1)),
    )

def predicate_0_2_expr(position, token):
    return Or(
        And(Or(position == IntVal(0), position == IntVal(3), position == IntVal(4)), token == StringVal('3')),
        And(Or(position == IntVal(1), position == IntVal(2), position == IntVal(7)), token == StringVal('0')),
        And(Or(position == IntVal(5), position == IntVal(6)), token == StringVal('4')),
    )

def predicate_0_3_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(1), q_position == IntVal(3), q_position == IntVal(6)), k_position == IntVal(2)),
        And(Or(q_position == IntVal(2), q_position == IntVal(5), q_position == IntVal(7)), k_position == IntVal(1)),
        And(q_position == IntVal(4), k_position == IntVal(3)),
    )

def predicate_1_0_expr(q_position, k_position):
    return Or(
        And(q_position == IntVal(0), k_position == IntVal(1)),
        And(q_position == IntVal(1), k_position == IntVal(2)),
        And(Or(q_position == IntVal(2), q_position == IntVal(6)), k_position == IntVal(4)),
        And(q_position == IntVal(3), k_position == IntVal(5)),
        And(q_position == IntVal(4), k_position == IntVal(6)),
        And(q_position == IntVal(5), k_position == IntVal(3)),
        And(q_position == IntVal(7), k_position == IntVal(0)),
    )

def predicate_1_1_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(4)), k_position == IntVal(1)),
        And(Or(q_position == IntVal(1), q_position == IntVal(5), q_position == IntVal(7)), k_position == IntVal(3)),
        And(Or(q_position == IntVal(2), q_position == IntVal(3)), k_position == IntVal(5)),
        And(q_position == IntVal(6), k_position == IntVal(4)),
    )

def predicate_1_2_expr(position, attn_0_1_output):
    return Or(
        And(Or(position == IntVal(0), position == IntVal(1)), attn_0_1_output == StringVal('0')),
        And(Or(position == IntVal(2), position == IntVal(3), position == IntVal(4)), attn_0_1_output == StringVal('</s>')),
        And(Or(position == IntVal(5), position == IntVal(6)), attn_0_1_output == StringVal('3')),
        And(position == IntVal(7), attn_0_1_output == StringVal('4')),
    )

def predicate_1_3_expr(position, token):
    return Or(
        And(Or(position == IntVal(0), position == IntVal(1)), token == StringVal('0')),
        And(Or(position == IntVal(2), position == IntVal(3), position == IntVal(4), position == IntVal(7)), token == StringVal('</s>')),
        And(position == IntVal(5), token == StringVal('<s>')),
        And(position == IntVal(6), token == StringVal('4')),
    )
def mlp_0_0_expr(position, attn_0_1_output):
    conds = [
        ((IntVal(0), StringVal("</s>")), 3),
        ((IntVal(3), StringVal("</s>")), 3),
        ((IntVal(4), StringVal("</s>")), 3),
        ((IntVal(5), StringVal("</s>")), 3),
        ((IntVal(6), StringVal("0")), 3),
        ((IntVal(6), StringVal("1")), 3),
        ((IntVal(6), StringVal("2")), 3),
        ((IntVal(6), StringVal("3")), 3),
        ((IntVal(6), StringVal("4")), 3),
        ((IntVal(6), StringVal("</s>")), 3),
        ((IntVal(6), StringVal("<s>")), 3),
        ((IntVal(7), StringVal("2")), 3),
        ((IntVal(7), StringVal("3")), 3),
        ((IntVal(7), StringVal("4")), 3),
        ((IntVal(7), StringVal("</s>")), 3),
        ((IntVal(7), StringVal("<s>")), 3),
        ((IntVal(1), StringVal("</s>")), 2),
        ((IntVal(2), StringVal("1")), 2),
        ((IntVal(2), StringVal("2")), 2),
        ((IntVal(2), StringVal("3")), 2),
        ((IntVal(2), StringVal("4")), 2),
        ((IntVal(3), StringVal("1")), 2),
        ((IntVal(3), StringVal("2")), 2),
        ((IntVal(3), StringVal("3")), 2),
        ((IntVal(3), StringVal("4")), 2),
        ((IntVal(4), StringVal("0")), 2),
        ((IntVal(5), StringVal("0")), 2),
        ((IntVal(1), StringVal("0")), 0),
        ((IntVal(1), StringVal("1")), 0),
        ((IntVal(1), StringVal("2")), 0),
        ((IntVal(1), StringVal("3")), 0),
        ((IntVal(1), StringVal("<s>")), 0),
        ((IntVal(2), StringVal("0")), 0),
        ((IntVal(3), StringVal("0")), 0),
        ((IntVal(7), StringVal("0")), 0),
        ((IntVal(0), StringVal("0")), 1),
        ((IntVal(1), StringVal("4")), 1),
    ]

    expr = IntVal(6)  # default value
    for (p, t), val in reversed(conds):
        expr = If(And(position == p, attn_0_1_output == t), val, expr)
    return expr


def mlp_1_0_expr(position, attn_1_2_output):
    conds = [
        ((IntVal(0), StringVal("2")), 0),
        ((IntVal(0), StringVal("3")), 0),
        ((IntVal(5), StringVal("0")), 0),
        ((IntVal(5), StringVal("1")), 0),
        ((IntVal(5), StringVal("2")), 0),
        ((IntVal(5), StringVal("3")), 0),
        ((IntVal(5), StringVal("<s>")), 0),
        ((IntVal(7), StringVal("3")), 0),
        ((IntVal(1), StringVal("0")), 2),
        ((IntVal(1), StringVal("1")), 2),
        ((IntVal(1), StringVal("2")), 2),
        ((IntVal(1), StringVal("3")), 2),
        ((IntVal(1), StringVal("4")), 2),
        ((IntVal(1), StringVal("</s>")), 2),
        ((IntVal(1), StringVal("<s>")), 2),
        ((IntVal(2), StringVal("2")), 7),
        ((IntVal(2), StringVal("3")), 7),
        ((IntVal(2), StringVal("4")), 7),
        ((IntVal(2), StringVal("</s>")), 7),
        ((IntVal(3), StringVal("2")), 7),
        ((IntVal(3), StringVal("3")), 7),
        ((IntVal(4), StringVal("2")), 7),
        ((IntVal(4), StringVal("3")), 7),
        ((IntVal(6), StringVal("0")), 4),
        ((IntVal(6), StringVal("1")), 4),
        ((IntVal(6), StringVal("<s>")), 4),
        ((IntVal(7), StringVal("2")), 4),
        ((IntVal(7), StringVal("<s>")), 4),
        ((IntVal(6), StringVal("2")), 1),
        ((IntVal(6), StringVal("3")), 1),
        ((IntVal(6), StringVal("4")), 1),
        ((IntVal(6), StringVal("</s>")), 1),
    ]

    expr = IntVal(3)  # default value
    for (p, t), val in reversed(conds):
        expr = If(And(position == p, attn_1_2_output == t), val, expr)
    return expr


# —————— Читаем веса и настраиваем константы ——————
classifier_weights = pd.read_csv("sort_Z3/sort_weights.csv", index_col=[0, 1], dtype={"feature": str})
classes = classifier_weights.columns.tolist()

def build_pipeline(solver, tokens, position_vars):
    """
    Надёргивает все attention-, MLP-блоки, генерирует логиты и переменные pred[i].
    Возвращает словари outputs_by_name, logits, pred_vars.
    """
    N = len(tokens)
    # === Attention + MLP ===
    outs = {}
    outs["attn_0_0"] = build_attention_block(solver, tokens, position_vars, predicate_0_0_expr, tokens, "0_0")
    outs["attn_0_1"] = build_attention_block(solver, position_vars, position_vars, predicate_0_1_expr, tokens, "0_1")
    outs["attn_0_2"] = build_attention_block(solver, tokens, position_vars, predicate_0_2_expr, tokens, "0_2")
    outs["attn_0_3"] = build_attention_block(solver, position_vars, position_vars, predicate_0_3_expr, tokens, "0_3")
    outs["attn_1_0"] = build_attention_block(solver, position_vars, position_vars, predicate_1_0_expr, outs["attn_0_3"], "1_0")
    outs["attn_1_1"] = build_attention_block(solver, position_vars, position_vars, predicate_1_1_expr, tokens, "1_1")
    outs["attn_1_2"] = build_attention_block(solver, outs["attn_0_1"], position_vars, predicate_1_2_expr, outs["attn_0_1"], "1_2")
    outs["attn_1_3"] = build_attention_block(solver, tokens, position_vars, predicate_1_3_expr, outs["attn_0_0"], "1_3")
    outs["mlp_0_0"] = build_mlp_block(solver, position_vars, outs["attn_0_1"], mlp_0_0_expr, "0_0")
    outs["mlp_1_0"] = build_mlp_block(solver, position_vars, outs["attn_1_2"], mlp_1_0_expr, "1_0")
    # === Логиты ===
    logits = {(i, cls): Real(f"logit_{i}_{cls}") for i in range(N) for cls in classes}
    features = {
        "tokens": tokens,
        "positions": position_vars,
        "ones": [IntVal(1)] * N,
        "attn_0_0_outputs": outs["attn_0_0"],
        "attn_0_1_outputs": outs["attn_0_1"],
        "attn_0_2_outputs": outs["attn_0_2"],
        "attn_0_3_outputs": outs["attn_0_3"],
        "attn_1_0_outputs": outs["attn_1_0"],
        "attn_1_1_outputs": outs["attn_1_1"],
        "attn_1_2_outputs": outs["attn_1_2"],
        "attn_1_3_outputs": outs["attn_1_3"],
        "mlp_0_0_outputs": outs["mlp_0_0"],
        "mlp_1_0_outputs": outs["mlp_1_0"],
    }
    # для каждого (i,cls) одно уравнение logit = Sum(If(...))
    for i in range(N):
        for cls in classes:
            contribs = []
            for feat_name, exprs in features.items():
                feat_var = exprs[i]
                for ((f_name, f_val), weights) in classifier_weights.iterrows():
                    if f_name != feat_name:
                        continue
                    w = RealVal(str(weights[cls]))
                    if feat_name == 'ones':
                        contribs.append(w)
                    else:
                        if isinstance(feat_var, AstRef) and feat_var.sort() == StringSort():
                            const = StringVal(f_val)
                        else:
                            const = IntVal(int(f_val))
                        contribs.append(If(feat_var == const, w, RealVal('0')))
            solver.add(logits[(i, cls)] == Sum(contribs))

    # === Предсказания ===
    pred = [String(f"pred_{i}") for i in range(N)]
    for i in range(N):
        if i == 0:
            solver.add(pred[i] == tokens[0])
        elif i == N-1:
            solver.add(pred[i] == tokens[N-1])
        else:
            for cls in classes:
                cond = And([logits[(i, cls)] >= logits[(i, o)] for o in classes if o != cls])
                solver.add(Implies(cond, pred[i] == StringVal(cls)))

    return outs, logits, pred


def compute_original_predictions(input_tokens):
    N = len(input_tokens)
    s1 = Solver()
    # 1. Переменные и фиксация input_tokens
    tokens = [String(f"token_{i}") for i in range(N)]
    for i, val in enumerate(input_tokens):
        s1.add(tokens[i] == StringVal(val))
    pos = [Int(f"pos_{i}") for i in range(N)]
    for i in range(N):
        s1.add(pos[i] == IntVal(i))
    # 2. Прогон пайплайна
    _, logits, pred_orig_vars = build_pipeline(s1, tokens, pos)
    assert s1.check() == sat
    m = s1.model()

    # 3. Вынимаем конкретные строки
    return [str(m.evaluate(pred_orig_vars[i]).as_string()) for i in range(N)]