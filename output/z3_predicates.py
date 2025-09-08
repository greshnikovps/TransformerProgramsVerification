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