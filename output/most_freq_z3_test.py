from z3 import *
import pandas as pd

def aggregate_expr(attn_row, values):
    # Takes from values[j] the j where attn_row[j] == True; if none - fallback to values[0]
    expr = values[0]
    # Iterate in reverse order to account for early indices in case of matches
    for j in reversed(range(len(attn_row))):
        expr = If(attn_row[j], values[j], expr)
    return expr

def build_attention_block(solver, keys, queries, predicate_expr, values, name):
    """
    Building an attention block:
    - keys: list of elements (Int or String) for predicate_expr
    - queries: list of elements (Int or String) to select from
    - predicate_expr: function (q, k) -> BoolRef, defining the match condition
    - values: list of elements (Int or String) for aggregate
    - name: suffix for variable names in Z3
    Returns a list of N outputs (String or Int), similar to `outs[...]`.
    """
    N = len(keys)
    # matrix of Bool variables attn[i][j]
    attn = [[Bool(f"attn_{name}_{i}_{j}") for j in range(N)] for i in range(N)]
    # flags: for each i, is there any match among keys[j]
    any_match = [Bool(f"any_{name}_{i}") for i in range(N)]

    # Calculate any_match[i] == Or(predicate_expr(queries[i], keys[j]) for j in range(N))
    for i in range(N):
        solver.add(any_match[i] == Or([predicate_expr(queries[i], keys[j]) for j in range(N)]))

    # Determine output type: Int or String, depending on values
    if values and isinstance(values[0], AstRef) and values[0].sort() == IntSort():
        outputs = [Int(f"attn_{name}_output_{i}") for i in range(N)]
    else:
        outputs = [String(f"attn_{name}_output_{i}") for i in range(N)]

    for i in range(N):
        # exactly one True in row i
        solver.add(Sum([If(attn[i][j], 1, 0) for j in range(N)]) == 1)

        # for each j:
        for j in range(N):
            if j == 0:
                # fallback: attn[i][0] can be True if there's no match, or if predicate_expr is true for (i,0)
                solver.add(Implies(attn[i][0], Or(Not(any_match[i]), predicate_expr(queries[i], keys[0]))))
            else:
                # if attn[i][j] == True, then predicate_expr(queries[i], keys[j]) must be True
                solver.add(Implies(attn[i][j], predicate_expr(queries[i], keys[j])))

        # closest condition: if attn[i][k] and predicate_expr(queries[i], keys[j]) is true,
        # then distance |i-k| <= |i-j| for all j.
        for j in range(N):
            for k in range(N):
                solver.add(Implies(
                    And(attn[i][k], predicate_expr(queries[i], keys[j])),
                    Abs(i - k) <= Abs(i - j)
                ))

        # aggregate: select a value from values based on the attn[i] vector
        solver.add(outputs[i] == aggregate_expr(attn[i], values))

    return outputs

def build_mlp_block(solver, positions, tokens, mlp_expr_fn, name):
    """
    Building an MLP block: for each position i, create an Int output variable mlp_{name}_output_{i}
    and constraint: output == mlp_expr_fn(position, token_at_position).
    """
    N = len(tokens)
    outputs = [Int(f"mlp_{name}_output_{i}") for i in range(N)]
    for i in range(N):
        solver.add(outputs[i] == mlp_expr_fn(positions[i], tokens[i]))
    return outputs


def predicate_0_0_expr(position, token):
    return Or(
        And(position == IntVal(0), token == StringVal('1')),
        And(Or(position == IntVal(1), position == IntVal(2), position == IntVal(3), position == IntVal(4), position == IntVal(5)), token == StringVal('3')),
        And(Or(position == IntVal(6), position == IntVal(7)), token == StringVal('4')),
    )

def predicate_0_1_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(1), q_position == IntVal(7)), k_position == IntVal(1)),
        And(Or(q_position == IntVal(2), q_position == IntVal(3)), k_position == IntVal(0)),
        And(q_position == IntVal(4), k_position == IntVal(5)),
        And(q_position == IntVal(5), k_position == IntVal(6)),
        And(q_position == IntVal(6), k_position == IntVal(7)),
    )

def predicate_0_2_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(4)), k_position == IntVal(6)),
        And(Or(q_position == IntVal(1), q_position == IntVal(2)), k_position == IntVal(2)),
        And(Or(q_position == IntVal(3), q_position == IntVal(7)), k_position == IntVal(4)),
        And(Or(q_position == IntVal(5), q_position == IntVal(6)), k_position == IntVal(7)),
    )

def predicate_0_3_expr(position, token):
    return Or(
        And(Or(position == IntVal(0), position == IntVal(6)), token == StringVal('1')),
        And(Or(position == IntVal(1), position == IntVal(2)), token == StringVal('5')),
        And(Or(position == IntVal(3), position == IntVal(4), position == IntVal(5)), token == StringVal('0')),
        And(position == IntVal(7), token == StringVal('3')),
    )

def predicate_1_0_expr(q_position, k_position):
    return Or(
        And(q_position == IntVal(0), k_position == IntVal(6)),
        And(Or(q_position == IntVal(1), q_position == IntVal(2)), k_position == IntVal(3)),
        And(Or(q_position == IntVal(3), q_position == IntVal(4), q_position == IntVal(5), q_position == IntVal(6)), k_position == IntVal(1)),
        And(q_position == IntVal(7), k_position == IntVal(4)),
    )

def predicate_1_1_expr(q_position, k_position):
    return Or(
        And(q_position == IntVal(0), k_position == IntVal(5)),
        And(q_position == IntVal(1), k_position == IntVal(6)),
        And(Or(q_position == IntVal(2), q_position == IntVal(3), q_position == IntVal(4), q_position == IntVal(5), q_position == IntVal(6)), k_position == IntVal(0)),
        And(q_position == IntVal(7), k_position == IntVal(1)),
    )

def predicate_1_2_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(5)), k_position == IntVal(1)),
        And(q_position == IntVal(1), k_position == IntVal(6)),
        And(Or(q_position == IntVal(2), q_position == IntVal(4)), k_position == IntVal(2)),
        And(q_position == IntVal(3), k_position == IntVal(4)),
        And(q_position == IntVal(6), k_position == IntVal(5)),
        And(q_position == IntVal(7), k_position == IntVal(0)),
    )

def predicate_1_3_expr(q_position, k_position):
    return Or(
        And(Or(q_position == IntVal(0), q_position == IntVal(5)), k_position == IntVal(7)),
        And(q_position == IntVal(1), k_position == IntVal(3)),
        And(q_position == IntVal(2), k_position == IntVal(2)),
        And(q_position == IntVal(3), k_position == IntVal(6)),
        And(Or(q_position == IntVal(4), q_position == IntVal(7)), k_position == IntVal(0)),
        And(q_position == IntVal(6), k_position == IntVal(1)),
    )
def mlp_0_0_expr(position, attn_0_3_output):
    conds = [
        ((IntVal(1), StringVal("1")), 6),
        ((IntVal(5), StringVal("1")), 6),
        ((IntVal(5), StringVal("2")), 6),
        ((IntVal(5), StringVal("3")), 6),
        ((IntVal(5), StringVal("4")), 6),
        ((IntVal(0), StringVal("<s>")), 0),
        ((IntVal(3), StringVal("<s>")), 0),
        ((IntVal(4), StringVal("<s>")), 0),
        ((IntVal(5), StringVal("<s>")), 0),
        ((IntVal(7), StringVal("<s>")), 0),
        ((IntVal(0), StringVal("1")), 4),
        ((IntVal(1), StringVal("<s>")), 4),
        ((IntVal(7), StringVal("1")), 4),
        ((IntVal(6), StringVal("<s>")), 2),
    ]

    expr = IntVal(1)  # default value
    for (p, t), val in reversed(conds):
        expr = If(And(position == p, attn_0_3_output == t), val, expr)
    return expr


def mlp_1_0_expr(position, attn_1_2_output):
    conds = [
        ((IntVal(0), StringVal("1")), 7),
        ((IntVal(1), StringVal("1")), 7),
        ((IntVal(1), StringVal("3")), 7),
        ((IntVal(3), StringVal("0")), 7),
        ((IntVal(3), StringVal("1")), 7),
        ((IntVal(4), StringVal("0")), 7),
        ((IntVal(4), StringVal("1")), 7),
        ((IntVal(5), StringVal("0")), 7),
        ((IntVal(5), StringVal("1")), 7),
        ((IntVal(6), StringVal("0")), 7),
        ((IntVal(6), StringVal("1")), 7),
        ((IntVal(7), StringVal("1")), 7),
        ((IntVal(0), StringVal("2")), 1),
        ((IntVal(1), StringVal("0")), 1),
        ((IntVal(1), StringVal("2")), 1),
        ((IntVal(1), StringVal("5")), 1),
        ((IntVal(3), StringVal("2")), 1),
        ((IntVal(4), StringVal("2")), 1),
        ((IntVal(5), StringVal("2")), 1),
        ((IntVal(6), StringVal("2")), 1),
        ((IntVal(7), StringVal("2")), 1),
        ((IntVal(2), StringVal("0")), 0),
        ((IntVal(2), StringVal("1")), 0),
        ((IntVal(2), StringVal("2")), 0),
        ((IntVal(2), StringVal("5")), 0),
        ((IntVal(3), StringVal("5")), 0),
        ((IntVal(1), StringVal("4")), 2),
        ((IntVal(2), StringVal("3")), 3),
        ((IntVal(3), StringVal("3")), 3),
    ]

    expr = IntVal(5)  # default value
    for (p, t), val in reversed(conds):
        expr = If(And(position == p, attn_1_2_output == t), val, expr)
    return expr


# —————— Read weights and set up constants ——————
classifier_weights = pd.read_csv("most_freq/most_freq_weights.csv", index_col=[0, 1], dtype={"feature": str})
classes = classifier_weights.columns.tolist()

def build_pipeline(solver, tokens, position_vars):
    """
    Pulls all attention-, MLP-blocks, generates logits and pred[i] variables.
    Returns dictionaries outputs_by_name, logits, pred_vars.
    """
    N = len(tokens)
    # === Attention + MLP ===
    outs = {}
    outs["attn_0_0"] = build_attention_block(solver, tokens, position_vars, predicate_0_0_expr, tokens, "0_0")
    outs["attn_0_1"] = build_attention_block(solver, position_vars, position_vars, predicate_0_1_expr, tokens, "0_1")
    outs["attn_0_2"] = build_attention_block(solver, position_vars, position_vars, predicate_0_2_expr, tokens, "0_2")
    outs["attn_0_3"] = build_attention_block(solver, tokens, position_vars, predicate_0_3_expr, tokens, "0_3")
    outs["mlp_0_0"] = build_mlp_block(solver, position_vars, outs["attn_0_3"], mlp_0_0_expr, "0_0")
    outs["attn_1_0"] = build_attention_block(solver, position_vars, position_vars, predicate_1_0_expr, outs["mlp_0_0"], "1_0")
    outs["attn_1_1"] = build_attention_block(solver, position_vars, position_vars, predicate_1_1_expr, outs["mlp_0_0"], "1_1")
    outs["attn_1_2"] = build_attention_block(solver, position_vars, position_vars, predicate_1_2_expr, outs["attn_0_2"], "1_2")
    outs["attn_1_3"] = build_attention_block(solver, position_vars, position_vars, predicate_1_3_expr, outs["attn_0_1"], "1_3")
    outs["mlp_1_0"] = build_mlp_block(solver, position_vars, outs["attn_1_2"], mlp_1_0_expr, "1_0")
    # === Logits ===
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
    # for each (i,cls) one equation logit = Sum(If(...))
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

    # === Predictions ===
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
    # 1. Variables and fixing input_tokens
    tokens = [String(f"token_{i}") for i in range(N)]
    for i, val in enumerate(input_tokens):
        s1.add(tokens[i] == StringVal(val))
    pos = [Int(f"pos_{i}") for i in range(N)]
    for i in range(N):
        s1.add(pos[i] == IntVal(i))
    # 2. Run the pipeline
    _, logits, pred_orig_vars = build_pipeline(s1, tokens, pos)
    assert s1.check() == sat
    m = s1.model()

    # 3. Extract concrete strings
    return [str(m.evaluate(pred_orig_vars[i]).as_string()) for i in range(N)]
