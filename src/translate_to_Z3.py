from z3 import *

def create_input_variables(
    num_categorical_features: int,
    embedding_dim: int,
    num_numerical_features: int,
    prefix: str = "input"
):
    """
    Создаёт логические переменные Z3 для входов модели трансформера:
    - эмбеддинги категориальных признаков
    - числовые признаки

    Аргументы:
    - num_categorical_features: количество категориальных признаков
    - embedding_dim: размер эмбеддинга для каждого категориального признака
    - num_numerical_features: количество числовых признаков
    - prefix: префикс для имён переменных

    Возвращает:
    - List[z3.Real]: список всех входных переменных
    - Dict[str, List[z3.Real]]: словарь по признаковым группам
    """
    variables = []
    groups = {}

    # Категориальные эмбеддинги
    for i in range(num_categorical_features):
        name = f"{prefix}_cat{i}"
        vec = [Real(f"{name}_dim{j}") for j in range(embedding_dim)]
        variables.extend(vec)
        groups[name] = vec

    # Числовые признаки
    num_vars = [Real(f"{prefix}_num{j}") for j in range(num_numerical_features)]
    variables.extend(num_vars)
    groups["numerical"] = num_vars

    return variables, groups


def linear_layer_z3(x_vars, W, b, prefix="y"):
    """
    Создаёт логическое представление линейного слоя y = x @ W + b в Z3.

    Аргументы:
    - x_vars: список переменных z3.Real — входной вектор x длины d
    - W: матрица весов (список списков float), размерность (d, m)
    - b: вектор сдвига (список float), длина m
    - prefix: префикс для имён выходных переменных y_i

    Возвращает:
    - y_vars: список переменных z3.Real, представляющих выходной вектор y
    - constraints: список утверждений Z3: y_i == sum_j x_j * W[j][i] + b[i]
    """
    d = len(x_vars)
    m = len(b)

    assert len(W) == d, "W должно иметь столько строк, сколько элементов в x"
    assert all(len(W[j]) == m for j in range(d)), "Каждая строка W должна иметь длину m"

    y_vars = [Real(f"{prefix}_{i}") for i in range(m)]
    constraints = []

    for i in range(m):
        linear_comb = Sum([x_vars[j] * W[j][i] for j in range(d)]) + b[i]
        constraints.append(y_vars[i] == linear_comb)

    return y_vars, constraints


def relu_z3(x_vars, prefix="relu"):
    """
    Возвращает Z3-переменные и ограничения, соответствующие ReLU(x)
    """
    relu_vars = []
    constraints = []

    for i, x in enumerate(x_vars):
        y = Real(f"{prefix}_{i}")
        relu_vars.append(y)
        constraints.append(y == If(x > 0, x, 0))

    return relu_vars, constraints


def gelu_z3(x_vars, prefix="gelu"):
    """
    Полиномиальная аппроксимация GELU без exp и tanh:
        GELU(x) ≈ 0.5 * x * (1 + a1 * x + a2 * x^3)
    """
    a1 = 0.797885
    a2 = 0.035677

    gelu_vars = []
    constraints = []

    for i, x in enumerate(x_vars):
        y = Real(f"{prefix}_{i}")
        gelu_vars.append(y)

        x_cubed = x * x * x
        gelu_expr = 0.5 * x * (1 + a1 * x + a2 * x_cubed)

        constraints.append(y == gelu_expr)

    return gelu_vars, constraints


def compute_qkv_z3(x_vars, Wq, bq, Wk, bk, Wv, bv, prefix="attn"):
    """
    Вычисляет Q, K, V: три линейных преобразования из x.
    x_vars: список Z3-переменных
    Wq, Wk, Wv: матрицы весов
    bq, bk, bv: векторы смещений
    Возвращает:
        Q_vars, K_vars, V_vars — списки переменных
        constraints — список всех логических ограничений
    """
    Q_vars, c_q = linear_layer_z3(x_vars, Wq, bq, prefix=f"{prefix}_Q")
    K_vars, c_k = linear_layer_z3(x_vars, Wk, bk, prefix=f"{prefix}_K")
    V_vars, c_v = linear_layer_z3(x_vars, Wv, bv, prefix=f"{prefix}_V")

    constraints = c_q + c_k + c_v
    return Q_vars, K_vars, V_vars, constraints


def attention_z3(Q, K, V, dk, prefix="attn"):
    """
    Z3-реализация упрощённой attention-механики.

    Q, K, V: списки векторов (Q_i, K_j, V_j) — каждый вектор это список z3 Real переменных
    dk: размерность d_k (число)
    prefix: префикс имён для переменных
    Возвращает:
        output_vectors: список выходных векторов (по одному на каждый Q_i)
        constraints: список z3-ограничений
    """
    constraints = []
    scale = 1.0 / (dk ** 0.5)
    num_queries = len(Q)
    num_keys = len(K)
    dim_v = len(V[0])  # размерность выходного вектора

    output_vectors = []

    for i in range(num_queries):
        # Скалярные произведения s_{ij}
        s_ij = []
        for j in range(num_keys):
            dot = Real(f"{prefix}_score_{i}_{j}")
            expr = Sum([Q[i][k] * K[j][k] for k in range(dk)]) * scale
            constraints.append(dot == expr)
            s_ij.append(dot)

        # Нормализованное приближение softmax: alpha_{ij} = s_{ij} / sum(s_{ij})
        sum_scores = Real(f"{prefix}_sum_scores_{i}")
        constraints.append(sum_scores == Sum(s_ij))

        alpha = []
        for j in range(num_keys):
            a_ij = Real(f"{prefix}_alpha_{i}_{j}")
            constraints.append(a_ij == s_ij[j] / sum_scores)
            alpha.append(a_ij)

        # Выход: output_i = sum_j alpha_ij * V_j
        output_i = []
        for d in range(dim_v):
            out_comp = Real(f"{prefix}_out_{i}_{d}")
            expr = Sum([alpha[j] * V[j][d] for j in range(num_keys)])
            constraints.append(out_comp == expr)
            output_i.append(out_comp)

        output_vectors.append(output_i)

    return output_vectors, constraints


def output_projection_z3(heads_outputs, W_out, b_out, prefix="proj"):
    """
    heads_outputs: список списков Z3-переменных (выходы attention-голов)
    W_out: матрица весов output-проекции (список списков чисел)
    b_out: вектор смещений
    prefix: префикс имён переменных
    Возвращает:
        projected: список новых Z3 переменных (результат проекции)
        constraints: список ограничений
    """
    constraints = []

    # Конкатенация выходов голов
    concat_out = []
    for head in heads_outputs:
        concat_out.extend(head)

    input_dim = len(concat_out)
    output_dim = len(b_out)

    assert all(len(row) == input_dim for row in W_out), "Неверная форма W_out"

    # Применение линейного слоя: y_i = sum_j W[i][j] * x_j + b[i]
    projected = []
    for i in range(output_dim):
        y_i = Real(f"{prefix}_y_{i}")
        expr = Sum([RealVal(W_out[i][j]) * concat_out[j] for j in range(input_dim)]) + RealVal(b_out[i])
        constraints.append(y_i == expr)
        projected.append(y_i)

    return projected, constraints


def layernorm_z3(x, gamma, beta, eps=1e-5, prefix="ln"):
    """
    x: список Z3-переменных (вход)
    gamma, beta: списки чисел (параметры масштабирования и сдвига)
    eps: стабильная константа
    prefix: префикс имён переменных
    Возвращает:
        normalized: список новых Z3 переменных (нормализованный выход)
        constraints: список ограничений
    """
    n = len(x)
    constraints = []

    # Вычисляем среднее: mu = sum(x) / n
    mu = Real(f"{prefix}_mean")
    mu_expr = Sum(x) / n
    constraints.append(mu == mu_expr)

    # Вычисляем дисперсию: var = sum((x_i - mu)^2) / n
    var = Real(f"{prefix}_var")
    var_expr = Sum([(x[i] - mu)**2 for i in range(n)]) / n
    constraints.append(var == var_expr)

    # Вычисляем sqrt(var + eps) с отдельной переменной
    sqrt_var_eps = Real(f"{prefix}_sqrt_var_eps")
    constraints.append(sqrt_var_eps * sqrt_var_eps == var + RealVal(eps))
    constraints.append(sqrt_var_eps >= 0)  # sqrt по определению неотрицателен

    # Нормализуем и применяем scale/shift
    normalized = []
    for i in range(n):
        out_i = Real(f"{prefix}_out_{i}")
        norm_expr = ((x[i] - mu) / sqrt_var_eps) * RealVal(gamma[i]) + RealVal(beta[i])
        constraints.append(out_i == norm_expr)
        normalized.append(out_i)

    return normalized, constraints


from z3 import *


def transformer_encoder_block_z3(x, params, prefix="enc"):
    """
    Собирает Z3-представление одного Transformer Encoder блока.

    Аргументы:
        x — список Z3 Real переменных (входной вектор).
        params — словарь с весами:
            {
                "Wq", "bq", "Wk", "bk", "Wv", "bv",         # attention
                "W_out", "b_out",                          # attention output projection
                "W1", "b1", "W2", "b2",                    # MLP
                "gamma1", "beta1",                         # LayerNorm 1
                "gamma2", "beta2",                         # LayerNorm 2
            }
        prefix — префикс для имён выходов.

    Возвращает:
        output — итоговый список Z3-переменных.
        constraints — список всех логических формул.
    """
    constraints = []

    # --- 1. Self-Attention ---
    Q, c1 = linear_layer_z3(x, params["Wq"], params["bq"], prefix=prefix + "_Q")
    K, c2 = linear_layer_z3(x, params["Wk"], params["bk"], prefix=prefix + "_K")
    V, c3 = linear_layer_z3(x, params["Wv"], params["bv"], prefix=prefix + "_V")
    constraints += c1 + c2 + c3

    att_out, c4 = attention_z3(Q, K, V, prefix=prefix + "_att")
    constraints += c4

    proj_out, c5 = linear_layer_z3(att_out, params["W_out"], params["b_out"], prefix=prefix + "_att_proj")
    constraints += c5

    # --- 2. Residual + LayerNorm #1 ---
    res1, c6 = residual_connection_z3(x, proj_out, prefix=prefix + "_res1")
    constraints += c6

    norm1, c7 = layernorm_z3(res1, params["gamma1"], params["beta1"], eps=1e-5, prefix=prefix + "_norm1")
    constraints += c7

    # --- 3. MLP: Linear -> GELU -> Linear ---
    hidden, c8 = linear_layer_z3(norm1, params["W1"], params["b1"], prefix=prefix + "_mlp1")
    constraints += c8

    activated, c9 = gelu_z3(hidden, prefix=prefix + "_gelu")
    constraints += c9

    projected, c10 = linear_layer_z3(activated, params["W2"], params["b2"], prefix=prefix + "_mlp2")
    constraints += c10

    # --- 4. Residual + LayerNorm #2 ---
    res2, c11 = residual_connection_z3(norm1, projected, prefix=prefix + "_res2")
    constraints += c11

    norm2, c12 = layernorm_z3(res2, params["gamma2"], params["beta2"], eps=1e-5, prefix=prefix + "_norm2")
    constraints += c12

    return norm2, constraints
