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

# ===== Чтение весов для логитов (пример) =====
def read_classifier_weights(csv_path):
    """Читает classifier_weights CSV в pandas.DataFrame с index_col=[0,1]"""
    import pandas as pd
    df = pd.read_csv(csv_path, index_col=[0,1], dtype={"feature": str})
    return df


def predicate_0_0_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(7)), k_position == IntVal(6)),
        And(Or(q_position == IntVal(1), q_position == IntVal(3)), k_position == IntVal(2)),
        And(q_position == IntVal(2), k_position == IntVal(1)),
        And(q_position == IntVal(4), k_position == IntVal(3)),
        And(q_position == IntVal(5), k_position == IntVal(4)),
        And(q_position == IntVal(6), k_position == IntVal(5)),
        And(q_position == IntVal(8), k_position == IntVal(7)),
        And(q_position == IntVal(9), k_position == IntVal(8)),
        And(q_position == IntVal(10), k_position == IntVal(9)),
        And(q_position == IntVal(11), k_position == IntVal(10)),
        And(q_position == IntVal(12), k_position == IntVal(11)),
        And(q_position == IntVal(13), k_position == IntVal(12)),
        And(q_position == IntVal(14), k_position == IntVal(13)),
        And(Or(q_position == IntVal(31), q_position == IntVal(15)), k_position == IntVal(14)),
        And(q_position == IntVal(16), k_position == IntVal(15)),
        And(q_position == IntVal(17), k_position == IntVal(16)),
        And(q_position == IntVal(18), k_position == IntVal(17)),
        And(q_position == IntVal(19), k_position == IntVal(18)),
        And(q_position == IntVal(20), k_position == IntVal(19)),
        And(q_position == IntVal(21), k_position == IntVal(20)),
        And(q_position == IntVal(22), k_position == IntVal(21)),
        And(Or(q_position == IntVal(30), q_position == IntVal(23)), k_position == IntVal(22)),
        And(q_position == IntVal(24), k_position == IntVal(23)),
        And(q_position == IntVal(25), k_position == IntVal(24)),
        And(q_position == IntVal(26), k_position == IntVal(25)),
        And(q_position == IntVal(27), k_position == IntVal(26)),
        And(q_position == IntVal(28), k_position == IntVal(27)),
        And(q_position == IntVal(29), k_position == IntVal(28)),
    )

def predicate_0_1_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(12)), k_position == IntVal(13)),
        And(q_position == IntVal(1), k_position == IntVal(3)),
        And(Or(q_position == IntVal(2), q_position == IntVal(3), q_position == IntVal(31)), k_position == IntVal(4)),
        And(Or(q_position == IntVal(8), q_position == IntVal(4), q_position == IntVal(5)), k_position == IntVal(6)),
        And(Or(q_position == IntVal(6), q_position == IntVal(7), q_position == IntVal(19), q_position == IntVal(23), q_position == IntVal(26)), k_position == IntVal(20)),
        And(Or(q_position == IntVal(9), q_position == IntVal(20), q_position == IntVal(22), q_position == IntVal(30)), k_position == IntVal(24)),
        And(Or(q_position == IntVal(10), q_position == IntVal(14)), k_position == IntVal(11)),
        And(Or(q_position == IntVal(25), q_position == IntVal(11)), k_position == IntVal(26)),
        And(q_position == IntVal(13), k_position == IntVal(14)),
        And(q_position == IntVal(15), k_position == IntVal(16)),
        And(Or(q_position == IntVal(16), q_position == IntVal(21)), k_position == IntVal(17)),
        And(q_position == IntVal(17), k_position == IntVal(18)),
        And(q_position == IntVal(18), k_position == IntVal(15)),
        And(Or(q_position == IntVal(24), q_position == IntVal(27)), k_position == IntVal(25)),
        And(q_position == IntVal(28), k_position == IntVal(29)),
        And(q_position == IntVal(29), k_position == IntVal(28)),
    )

def predicate_0_2_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(3)), k_position == IntVal(2)),
        And(Or(q_position == IntVal(1), q_position == IntVal(5)), k_position == IntVal(4)),
        And(q_position == IntVal(2), k_position == IntVal(1)),
        And(Or(q_position == IntVal(4), q_position == IntVal(7)), k_position == IntVal(3)),
        And(q_position == IntVal(6), k_position == IntVal(5)),
        And(q_position == IntVal(8), k_position == IntVal(7)),
        And(q_position == IntVal(9), k_position == IntVal(8)),
        And(q_position == IntVal(10), k_position == IntVal(9)),
        And(q_position == IntVal(11), k_position == IntVal(10)),
        And(q_position == IntVal(12), k_position == IntVal(11)),
        And(q_position == IntVal(13), k_position == IntVal(12)),
        And(q_position == IntVal(14), k_position == IntVal(13)),
        And(q_position == IntVal(15), k_position == IntVal(14)),
        And(q_position == IntVal(16), k_position == IntVal(15)),
        And(q_position == IntVal(17), k_position == IntVal(16)),
        And(q_position == IntVal(18), k_position == IntVal(17)),
        And(q_position == IntVal(19), k_position == IntVal(18)),
        And(q_position == IntVal(20), k_position == IntVal(19)),
        And(q_position == IntVal(21), k_position == IntVal(20)),
        And(q_position == IntVal(22), k_position == IntVal(21)),
        And(q_position == IntVal(23), k_position == IntVal(22)),
        And(q_position == IntVal(24), k_position == IntVal(23)),
        And(q_position == IntVal(25), k_position == IntVal(24)),
        And(q_position == IntVal(26), k_position == IntVal(25)),
        And(q_position == IntVal(27), k_position == IntVal(26)),
        And(q_position == IntVal(28), k_position == IntVal(27)),
        And(Or(q_position == IntVal(29), q_position == IntVal(30), q_position == IntVal(31)), k_position == IntVal(28)),
    )

def predicate_0_3_expr(position, var0_embedding):
    return Or(
        And(position == IntVal(0), var0_embedding == IntVal(7)),
        And(Or(position == IntVal(1), position == IntVal(26), position == IntVal(4)), var0_embedding == IntVal(3)),
        And(Or(position == IntVal(2), position == IntVal(30)), var0_embedding == IntVal(9)),
        And(position == IntVal(3), var0_embedding == IntVal(12)),
        And(Or(position == IntVal(5), position == IntVal(7), position == IntVal(12), position == IntVal(15), position == IntVal(24), position == IntVal(25)), var0_embedding == IntVal(11)),
        And(Or(position == IntVal(6), position == IntVal(8), position == IntVal(11), position == IntVal(17), position == IntVal(18)), var0_embedding == IntVal(26)),
        And(Or(position == IntVal(9), position == IntVal(20), position == IntVal(29), position == IntVal(22)), var0_embedding == IntVal(24)),
        And(position == IntVal(10), var0_embedding == IntVal(13)),
        And(Or(position == IntVal(19), position == IntVal(13), position == IntVal(14)), var0_embedding == IntVal(20)),
        And(position == IntVal(16), var0_embedding == IntVal(28)),
        And(Or(position == IntVal(21), position == IntVal(23)), var0_embedding == IntVal(30)),
        And(position == IntVal(27), var0_embedding == IntVal(25)),
        And(Or(position == IntVal(28), position == IntVal(31)), var0_embedding == IntVal(14)),
    )

def predicate_1_0_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(3), q_position == IntVal(15)), k_position == IntVal(2)),
        And(q_position == IntVal(1), k_position == IntVal(4)),
        And(q_position == IntVal(2), k_position == IntVal(1)),
        And(q_position == IntVal(4), k_position == IntVal(3)),
        And(Or(q_position == IntVal(10), q_position == IntVal(5), q_position == IntVal(7)), k_position == IntVal(6)),
        And(q_position == IntVal(6), k_position == IntVal(5)),
        And(q_position == IntVal(8), k_position == IntVal(7)),
        And(Or(q_position == IntVal(9), q_position == IntVal(12)), k_position == IntVal(11)),
        And(q_position == IntVal(11), k_position == IntVal(13)),
        And(q_position == IntVal(13), k_position == IntVal(12)),
        And(Or(q_position == IntVal(16), q_position == IntVal(14)), k_position == IntVal(15)),
        And(q_position == IntVal(17), k_position == IntVal(16)),
        And(Or(q_position == IntVal(18), q_position == IntVal(20)), k_position == IntVal(19)),
        And(q_position == IntVal(19), k_position == IntVal(18)),
        And(q_position == IntVal(21), k_position == IntVal(20)),
        And(q_position == IntVal(22), k_position == IntVal(21)),
        And(Or(q_position == IntVal(25), q_position == IntVal(23)), k_position == IntVal(22)),
        And(q_position == IntVal(24), k_position == IntVal(23)),
        And(q_position == IntVal(26), k_position == IntVal(24)),
        And(Or(q_position == IntVal(27), q_position == IntVal(28)), k_position == IntVal(26)),
        And(Or(q_position == IntVal(29), q_position == IntVal(30), q_position == IntVal(31)), k_position == IntVal(28)),
    )

def predicate_1_1_expr(position, var0_embedding):
    return Or(
        And(Or(position == IntVal(0), position == IntVal(25), position == IntVal(29), position == IntVal(31)), var0_embedding == IntVal(4)),
        And(position == IntVal(1), var0_embedding == IntVal(13)),
        And(position == IntVal(2), var0_embedding == IntVal(23)),
        And(Or(position == IntVal(27), position == IntVal(3), position == IntVal(4)), var0_embedding == IntVal(7)),
        And(Or(position == IntVal(20), position == IntVal(5)), var0_embedding == IntVal(9)),
        And(Or(position == IntVal(16), position == IntVal(6)), var0_embedding == IntVal(14)),
        And(Or(position == IntVal(15), position == IntVal(7)), var0_embedding == IntVal(16)),
        And(position == IntVal(8), var0_embedding == IntVal(8)),
        And(position == IntVal(9), var0_embedding == IntVal(12)),
        And(Or(position == IntVal(17), position == IntVal(10), position == IntVal(22)), var0_embedding == IntVal(26)),
        And(Or(position == IntVal(24), position == IntVal(26), position == IntVal(11), position == IntVal(13)), var0_embedding == IntVal(3)),
        And(position == IntVal(12), var0_embedding == IntVal(11)),
        And(Or(position == IntVal(18), position == IntVal(19), position == IntVal(21), position == IntVal(14)), var0_embedding == IntVal(20)),
        And(position == IntVal(23), var0_embedding == IntVal(18)),
        And(position == IntVal(28), var0_embedding == IntVal(6)),
        And(position == IntVal(30), var0_embedding == IntVal(2)),
    )

def predicate_1_2_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(29), q_position == IntVal(30), q_position == IntVal(31)), k_position == IntVal(28)),
        And(Or(q_position == IntVal(1), q_position == IntVal(5)), k_position == IntVal(4)),
        And(q_position == IntVal(2), k_position == IntVal(1)),
        And(q_position == IntVal(3), k_position == IntVal(2)),
        And(q_position == IntVal(4), k_position == IntVal(3)),
        And(q_position == IntVal(6), k_position == IntVal(5)),
        And(q_position == IntVal(7), k_position == IntVal(6)),
        And(q_position == IntVal(8), k_position == IntVal(7)),
        And(q_position == IntVal(9), k_position == IntVal(8)),
        And(q_position == IntVal(10), k_position == IntVal(9)),
        And(q_position == IntVal(11), k_position == IntVal(10)),
        And(q_position == IntVal(12), k_position == IntVal(11)),
        And(q_position == IntVal(13), k_position == IntVal(12)),
        And(q_position == IntVal(14), k_position == IntVal(13)),
        And(q_position == IntVal(15), k_position == IntVal(14)),
        And(q_position == IntVal(16), k_position == IntVal(15)),
        And(q_position == IntVal(17), k_position == IntVal(16)),
        And(q_position == IntVal(18), k_position == IntVal(17)),
        And(q_position == IntVal(19), k_position == IntVal(18)),
        And(q_position == IntVal(20), k_position == IntVal(19)),
        And(q_position == IntVal(21), k_position == IntVal(20)),
        And(q_position == IntVal(22), k_position == IntVal(21)),
        And(q_position == IntVal(23), k_position == IntVal(22)),
        And(q_position == IntVal(24), k_position == IntVal(23)),
        And(q_position == IntVal(25), k_position == IntVal(24)),
        And(q_position == IntVal(26), k_position == IntVal(25)),
        And(q_position == IntVal(27), k_position == IntVal(26)),
        And(q_position == IntVal(28), k_position == IntVal(27)),
    )

def predicate_1_3_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(25), q_position == IntVal(31)), k_position == IntVal(26)),
        And(q_position == IntVal(1), k_position == IntVal(3)),
        And(Or(q_position == IntVal(2), q_position == IntVal(5)), k_position == IntVal(4)),
        And(q_position == IntVal(3), k_position == IntVal(2)),
        And(q_position == IntVal(4), k_position == IntVal(6)),
        And(q_position == IntVal(6), k_position == IntVal(5)),
        And(Or(q_position == IntVal(10), q_position == IntVal(7)), k_position == IntVal(9)),
        And(q_position == IntVal(8), k_position == IntVal(10)),
        And(Or(q_position == IntVal(9), q_position == IntVal(12)), k_position == IntVal(11)),
        And(Or(q_position == IntVal(11), q_position == IntVal(14)), k_position == IntVal(13)),
        And(q_position == IntVal(13), k_position == IntVal(12)),
        And(q_position == IntVal(15), k_position == IntVal(28)),
        And(Or(q_position == IntVal(16), q_position == IntVal(17), q_position == IntVal(30)), k_position == IntVal(15)),
        And(q_position == IntVal(18), k_position == IntVal(20)),
        And(q_position == IntVal(19), k_position == IntVal(18)),
        And(Or(q_position == IntVal(20), q_position == IntVal(23)), k_position == IntVal(22)),
        And(Or(q_position == IntVal(24), q_position == IntVal(21)), k_position == IntVal(23)),
        And(q_position == IntVal(22), k_position == IntVal(21)),
        And(Or(q_position == IntVal(26), q_position == IntVal(27)), k_position == IntVal(25)),
        And(q_position == IntVal(28), k_position == IntVal(29)),
        And(q_position == IntVal(29), k_position == IntVal(27)),
    )
def mlp_0_0_expr(pos, token):
    conds = [
    ]

    expr = IntVal(23)  # default value
    for (p, t), val in reversed(conds):
        expr = If(And(pos == IntVal(p), token == StringVal(t)), val, expr)
    return expr


def mlp_1_0_expr(pos, token):
    conds = [
    ]

    expr = IntVal(11)  # default value
    for (p, t), val in reversed(conds):
        expr = If(And(pos == IntVal(p), token == StringVal(t)), val, expr)
    return expr


# —————— Читаем веса и настраиваем константы ——————
classifier_weights = pd.read_csv("sort_weights.csv", index_col=[0, 1], dtype={"feature": str})
classes = classifier_weights.columns.tolist()

def build_pipeline(solver, tokens, position_vars):
    """
    Надёргивает все attention-, MLP-блоки, генерирует логиты и переменные pred[i].
    Возвращает словари outputs_by_name, logits, pred_vars.
    """
    N = len(tokens)
    # === Инициализация дополнительных фич из run ===
    # Предполагается: задан input_tokens = [...] и classifier_weights загружен, pandas импортирован.
    classifier_weights = pd.read_csv(
        "conll_ner_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]
    embed_df = pd.read_csv("conll_ner_embeddings.csv").set_index("word")
    embeddings = embed_df.loc[tokens]
    var0_embeddings = embeddings["var0_embeddings"].tolist()
    var1_embeddings = embeddings["var1_embeddings"].tolist()
    var2_embeddings = embeddings["var2_embeddings"].tolist()
    var3_embeddings = embeddings["var3_embeddings"].tolist()
    var0_embedding_scores = classifier_weights.loc[
        [("var0_embeddings", str(v)) for v in var0_embeddings]
    ]
    var1_embedding_scores = classifier_weights.loc[
        [("var1_embeddings", str(v)) for v in var1_embeddings]
    ]
    var2_embedding_scores = classifier_weights.loc[
        [("var2_embeddings", str(v)) for v in var2_embeddings]
    ]
    var3_embedding_scores = classifier_weights.loc[
        [("var3_embeddings", str(v)) for v in var3_embeddings]
    ]
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]
    var0_embeddings_z3 = []
    for v in var0_embeddings:
        var0_embeddings_z3.append(StringVal(v))
    var1_embeddings_z3 = []
    for v in var1_embeddings:
        var1_embeddings_z3.append(StringVal(v))
    var2_embeddings_z3 = []
    for v in var2_embeddings:
        var2_embeddings_z3.append(StringVal(v))
    var3_embeddings_z3 = []
    for v in var3_embeddings:
        var3_embeddings_z3.append(StringVal(v))
    # === Attention + MLP ===
    outs = {}
    outs["attn_0_0"] = build_attention_block(solver, position_vars, position_vars, predicate_0_0_expr, var3_embeddings, "0_0")
    outs["attn_0_1"] = build_attention_block(solver, position_vars, position_vars, predicate_0_1_expr, var0_embeddings, "0_1")
    outs["attn_0_2"] = build_attention_block(solver, position_vars, position_vars, predicate_0_2_expr, var0_embeddings, "0_2")
    outs["attn_0_3"] = build_attention_block(solver, var0_embeddings, position_vars, predicate_0_3_expr, var0_embeddings, "0_3")
    outs["attn_1_0"] = build_attention_block(solver, position_vars, position_vars, predicate_1_0_expr, outs["attn_0_3"], "1_0")
    outs["attn_1_1"] = build_attention_block(solver, var0_embeddings, position_vars, predicate_1_1_expr, outs["attn_0_3"], "1_1")
    outs["attn_1_2"] = build_attention_block(solver, position_vars, position_vars, predicate_1_2_expr, var0_embeddings, "1_2")
    outs["attn_1_3"] = build_attention_block(solver, position_vars, position_vars, predicate_1_3_expr, outs["attn_0_2"], "1_3")
    outs["mlp_0_0"] = build_mlp_block(solver, outs["attn_0_2"], outs["attn_0_1"], mlp_0_0_expr, "0_0")
    outs["mlp_1_0"] = build_mlp_block(solver, var1_embeddings, var2_embeddings, mlp_1_0_expr, "1_0")
    # === Логиты ===
    logits = {(i, cls): Real(f"logit_{i}_{cls}") for i in range(N) for cls in classes}
    features = {
        "tokens": tokens,
        "positions": position_vars,
        "ones": [IntVal(1)] * N,
        "var0_embeddings": var0_embeddings_z3,
        "var1_embeddings": var1_embeddings_z3,
        "var2_embeddings": var2_embeddings_z3,
        "var3_embeddings": var3_embeddings_z3,
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
