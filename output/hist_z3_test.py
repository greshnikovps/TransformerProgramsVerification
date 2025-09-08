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


def predicate_0_0_expr(token, position):
    return Or(
        And(token == StringVal('0'), position == IntVal(2)),
        And(token == StringVal('1'), position == IntVal(5)),
        And(token == StringVal('2'), position == IntVal(3)),
        And(Or(token == StringVal('4'), token == StringVal('3')), position == IntVal(4)),
        And(Or(token == StringVal('<s>'), token == StringVal('5')), position == IntVal(1)),
    )

def predicate_0_1_expr(q_token, k_token):
    return Or(
        And(Or(q_token == StringVal('0'), q_token == StringVal('3')), k_token == StringVal('0')),
        And(q_token == StringVal('1'), k_token == StringVal('1')),
        And(q_token == StringVal('2'), k_token == StringVal('2')),
        And(q_token == StringVal('4'), k_token == StringVal('4')),
        And(q_token == StringVal('5'), k_token == StringVal('5')),
        And(q_token == StringVal('<s>'), k_token == StringVal('3')),
    )

def predicate_0_2_expr(q_token, k_token):
    return Or(
        And(q_token == StringVal('0'), k_token == StringVal('0')),
        And(Or(q_token == StringVal('2'), q_token == StringVal('1'), q_token == StringVal('3')), k_token == StringVal('<s>')),
        And(q_token == StringVal('4'), k_token == StringVal('4')),
        And(q_token == StringVal('5'), k_token == StringVal('5')),
        And(q_token == StringVal('<s>'), k_token == StringVal('1')),
    )

def predicate_0_3_expr(q_token, k_token):
    return Or(
        And(Or(q_token == StringVal('0'), q_token == StringVal('1'), q_token == StringVal('2'), q_token == StringVal('5'), q_token == StringVal('3')), k_token == StringVal('5')),
        And(q_token == StringVal('4'), k_token == StringVal('2')),
        And(q_token == StringVal('<s>'), k_token == StringVal('4')),
    )

def predicate_1_0_expr(token, attn_0_2_output):
    return Or(
        And(Or(token == StringVal('0'), token == StringVal('2'), token == StringVal('<s>'), token == StringVal('5'), token == StringVal('4'), token == StringVal('3')), attn_0_2_output == IntVal(1)),
        And(token == StringVal('1'), attn_0_2_output == IntVal(7)),
    )

def predicate_1_1_expr(q_token, k_token):
    return Or(
        And(q_token == StringVal('0'), k_token == StringVal('0')),
        And(Or(q_token == StringVal('1'), q_token == StringVal('2'), q_token == StringVal('<s>'), q_token == StringVal('5'), q_token == StringVal('4'), q_token == StringVal('3')), k_token == StringVal('<s>')),
    )

def predicate_1_2_expr(token, position):
    return Or(
        And(Or(token == StringVal('0'), token == StringVal('1'), token == StringVal('2'), token == StringVal('5'), token == StringVal('4'), token == StringVal('3')), position == IntVal(6)),
        And(token == StringVal('<s>'), position == IntVal(7)),
    )

def predicate_1_3_expr(position, attn_0_2_output):
    return Or(
        And(position == IntVal(0), attn_0_2_output == IntVal(7)),
        And(Or(position == IntVal(1), position == IntVal(5)), attn_0_2_output == IntVal(4)),
        And(Or(position == IntVal(2), position == IntVal(3), position == IntVal(6)), attn_0_2_output == IntVal(5)),
        And(position == IntVal(4), attn_0_2_output == IntVal(1)),
        And(position == IntVal(7), attn_0_2_output == IntVal(2)),
    )
def mlp_0_0_expr(attn_0_1_output, attn_0_0_output):
    conds = [
        ((StringVal("0"), StringVal("2")), 5),
        ((StringVal("0"), StringVal("3")), 5),
        ((StringVal("0"), StringVal("5")), 5),
        ((StringVal("1"), StringVal("3")), 5),
        ((StringVal("1"), StringVal("5")), 5),
        ((StringVal("2"), StringVal("0")), 5),
        ((StringVal("2"), StringVal("5")), 5),
        ((StringVal("3"), StringVal("0")), 5),
        ((StringVal("3"), StringVal("1")), 5),
        ((StringVal("3"), StringVal("2")), 5),
        ((StringVal("3"), StringVal("4")), 5),
        ((StringVal("3"), StringVal("5")), 5),
        ((StringVal("3"), StringVal("<s>")), 5),
        ((StringVal("4"), StringVal("0")), 5),
        ((StringVal("4"), StringVal("3")), 5),
        ((StringVal("4"), StringVal("5")), 5),
        ((StringVal("5"), StringVal("3")), 5),
        ((StringVal("4"), StringVal("<s>")), 3),
        ((StringVal("5"), StringVal("1")), 3),
        ((StringVal("5"), StringVal("2")), 3),
        ((StringVal("5"), StringVal("5")), 3),
        ((StringVal("5"), StringVal("<s>")), 3),
        ((StringVal("<s>"), StringVal("1")), 3),
        ((StringVal("<s>"), StringVal("2")), 3),
        ((StringVal("<s>"), StringVal("3")), 3),
        ((StringVal("<s>"), StringVal("4")), 3),
        ((StringVal("<s>"), StringVal("5")), 3),
        ((StringVal("<s>"), StringVal("<s>")), 3),
        ((StringVal("2"), StringVal("2")), 4),
        ((StringVal("2"), StringVal("<s>")), 4),
        ((StringVal("1"), StringVal("1")), 0),
    ]

    expr = IntVal(1)  # default value
    for (p, t), val in reversed(conds):
        expr = If(And(attn_0_1_output == p, attn_0_0_output == t), val, expr)
    return expr


def mlp_1_0_expr(attn_0_0_output, attn_0_3_output):
    conds = [
    ]

    expr = IntVal(1)  # default value
    for (p, t), val in reversed(conds):
        expr = If(And(attn_0_0_output == p, attn_0_3_output == t), val, expr)
    return expr


# —————— Читаем веса и настраиваем константы ——————
classifier_weights = pd.read_csv("hist/hist_weights.csv", index_col=[0, 1], dtype={"feature": str})
classes = classifier_weights.columns.tolist()

def build_pipeline(solver, tokens, position_vars):
    """
    Надёргивает все attention-, MLP-блоки, генерирует логиты и переменные pred[i].
    Возвращает словари outputs_by_name, logits, pred_vars.
    """
    N = len(tokens)
    # === Attention + MLP ===
    outs = {}
    outs["attn_0_0"] = build_attention_block(solver, position_vars, tokens, predicate_0_0_expr, tokens, "0_0")
    outs["attn_0_1"] = build_attention_block(solver, tokens, tokens, predicate_0_1_expr, tokens, "0_1")
    outs["attn_0_2"] = build_attention_block(solver, tokens, tokens, predicate_0_2_expr, position_vars, "0_2")
    outs["attn_0_3"] = build_attention_block(solver, tokens, tokens, predicate_0_3_expr, tokens, "0_3")
    outs["mlp_0_0"] = build_mlp_block(solver, outs["attn_0_1"], outs["attn_0_0"], mlp_0_0_expr, "0_0")
    outs["attn_1_0"] = build_attention_block(solver, outs["attn_0_2"], tokens, predicate_1_0_expr, position_vars, "1_0")
    outs["attn_1_1"] = build_attention_block(solver, tokens, tokens, predicate_1_1_expr, outs["mlp_0_0"], "1_1")
    outs["attn_1_2"] = build_attention_block(solver, position_vars, tokens, predicate_1_2_expr, position_vars, "1_2")
    outs["attn_1_3"] = build_attention_block(solver, outs["attn_0_2"], position_vars, predicate_1_3_expr, outs["mlp_0_0"], "1_3")
    outs["mlp_1_0"] = build_mlp_block(solver, outs["attn_0_0"], outs["attn_0_3"], mlp_1_0_expr, "1_0")
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
        "mlp_0_0_outputs": outs["mlp_0_0"],
        "attn_1_0_outputs": outs["attn_1_0"],
        "attn_1_1_outputs": outs["attn_1_1"],
        "attn_1_2_outputs": outs["attn_1_2"],
        "attn_1_3_outputs": outs["attn_1_3"],
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
