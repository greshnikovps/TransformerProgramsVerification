import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "conll_ner_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)
    # embed ######################################################
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

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 6
        elif q_position in {1, 3}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {31, 15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {30, 23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 28

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, var3_embeddings)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 12}:
            return k_position == 13
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 3, 31}:
            return k_position == 4
        elif q_position in {8, 4, 5}:
            return k_position == 6
        elif q_position in {6, 7, 19, 23, 26}:
            return k_position == 20
        elif q_position in {9, 20, 22, 30}:
            return k_position == 24
        elif q_position in {10, 14}:
            return k_position == 11
        elif q_position in {25, 11}:
            return k_position == 26
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {15}:
            return k_position == 16
        elif q_position in {16, 21}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 15
        elif q_position in {24, 27}:
            return k_position == 25
        elif q_position in {28}:
            return k_position == 29
        elif q_position in {29}:
            return k_position == 28

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, var0_embeddings)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 2
        elif q_position in {1, 5}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {4, 7}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29, 30, 31}:
            return k_position == 28

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, var0_embeddings)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, var0_embedding):
        if position in {0}:
            return var0_embedding == 7
        elif position in {1, 26, 4}:
            return var0_embedding == 3
        elif position in {2, 30}:
            return var0_embedding == 9
        elif position in {3}:
            return var0_embedding == 12
        elif position in {5, 7, 12, 15, 24, 25}:
            return var0_embedding == 11
        elif position in {6, 8, 11, 17, 18}:
            return var0_embedding == 26
        elif position in {9, 20, 29, 22}:
            return var0_embedding == 24
        elif position in {10}:
            return var0_embedding == 13
        elif position in {19, 13, 14}:
            return var0_embedding == 20
        elif position in {16}:
            return var0_embedding == 28
        elif position in {21, 23}:
            return var0_embedding == 30
        elif position in {27}:
            return var0_embedding == 25
        elif position in {28, 31}:
            return var0_embedding == 14

    attn_0_3_pattern = select_closest(var0_embeddings, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, var0_embeddings)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, attn_0_1_output):
        key = (attn_0_2_output, attn_0_1_output)
        return 23

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 3, 15}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {10, 5, 7}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9, 12}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {16, 14}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18, 20}:
            return k_position == 19
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {25, 23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {26}:
            return k_position == 24
        elif q_position in {27, 28}:
            return k_position == 26
        elif q_position in {29, 30, 31}:
            return k_position == 28

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, var0_embedding):
        if position in {0, 25, 29, 31}:
            return var0_embedding == 4
        elif position in {1}:
            return var0_embedding == 13
        elif position in {2}:
            return var0_embedding == 23
        elif position in {27, 3, 4}:
            return var0_embedding == 7
        elif position in {20, 5}:
            return var0_embedding == 9
        elif position in {16, 6}:
            return var0_embedding == 14
        elif position in {15, 7}:
            return var0_embedding == 16
        elif position in {8}:
            return var0_embedding == 8
        elif position in {9}:
            return var0_embedding == 12
        elif position in {17, 10, 22}:
            return var0_embedding == 26
        elif position in {24, 26, 11, 13}:
            return var0_embedding == 3
        elif position in {12}:
            return var0_embedding == 11
        elif position in {18, 19, 21, 14}:
            return var0_embedding == 20
        elif position in {23}:
            return var0_embedding == 18
        elif position in {28}:
            return var0_embedding == 6
        elif position in {30}:
            return var0_embedding == 2

    attn_1_1_pattern = select_closest(var0_embeddings, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 29, 30, 31}:
            return k_position == 28
        elif q_position in {1, 5}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {28}:
            return k_position == 27

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, var0_embeddings)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 25, 31}:
            return k_position == 26
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 5}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {10, 7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9, 12}:
            return k_position == 11
        elif q_position in {11, 14}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 28
        elif q_position in {16, 17, 30}:
            return k_position == 15
        elif q_position in {18}:
            return k_position == 20
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20, 23}:
            return k_position == 22
        elif q_position in {24, 21}:
            return k_position == 23
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {26, 27}:
            return k_position == 25
        elif q_position in {28}:
            return k_position == 29
        elif q_position in {29}:
            return k_position == 27

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(var1_embedding, var2_embedding):
        key = (var1_embedding, var2_embedding)
        return 11

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(var1_embeddings, var2_embeddings)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                var0_embedding_scores,
                var1_embedding_scores,
                var2_embedding_scores,
                var3_embedding_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                one_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(
    run(
        [
            "<s>",
            "Government",
            "said",
            "that",
            "police",
            "killed",
            #"TOP",
            #"AFTER",
            "<unk>",
            #"VICTORY",
            ".",
            "</s>",
        ]
    )
)
