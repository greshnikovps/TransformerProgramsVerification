"""
z3_code_utils.py

Генерирует *готовый* Z3-скрипт непосредственно из обученной модели,
минуя этап создания промежуточного Python-кода трансформера.

По интерфейсу ориентирован на `code_utils.model_to_code`, но возвращает
строку с Z3-кодом (при желании сразу записывает её в файл).

Основной публичный метод:
    z3_script = model_to_Z3(model, idx_w, idx_t, output_dir=".", name="program")
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional, Union, Sequence

import pandas as pd

# ==== вспомогательные импорты из существующих файлов ====
from src.utils.code_utils import (
    get_unembed_df,
    get_embed_df,
    cat_head_to_code,
    num_head_to_code,
    cat_mlp_to_code,
    num_mlp_to_code,
    get_var_types
)

from src.utils.generate_Z3 import (
    generate_static_z3_code,
    parse_predicate_function,
    generate_z3_predicate_code,
    parse_mlp_function,
    generate_mlp_expr_code,
    generate_build_pipeline_from_run,
    generate_compute_original_predictions_code,
)


# ---------------------------------------------------------------------------
# Основная функция: генерируем полностью рабочий Z3-скрипт
# ---------------------------------------------------------------------------
def model_to_Z3(  # noqa: C901  (функция действительно большая, оставляем как есть)
    model,
    idx_w: Sequence[int],
    idx_t: Sequence[int],
    *,
    embed_csv: bool = False,
    embed_enums: bool = False,
    unembed_csv: bool = True,
    one_hot: bool = False,
    autoregressive: bool = False,
    var_types=None,
    output_dir: Union[str, Path] = ".",
    name: str = "program",
    save: bool = True,
) -> str:
    """
    Превращает `model` в законченный Z3-скрипт.

    Параметры в точности повторяют `code_utils.model_to_code`, плюс:
        save (bool) – если True, скрипт сразу записывается в `<output_dir>/<name>.z3.py`

    Возвращает:
        Полный текст скрипта (**str**).  Можно выполнить через `exec`, записать в файл
        или прокормить Z3 напрямую.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if var_types == True:
        var_types = get_var_types(
            model, idx_w, one_hot=one_hot, enums=embed_enums
        )

    # ------------------------------------------------------------------ #
    # 1.  Сохраняем веса классификатора (и при необходимости – эмбеддинги)
    # ------------------------------------------------------------------ #
    weights_path: Optional[str] = None
    if unembed_csv:
        weights_df = get_unembed_df(
            model,
            idx_t,
            var_types=var_types,
            one_hot=one_hot,
            enums=embed_enums,
            unembed_mask=True,
        )
        weights_path = str(output_dir / f"{name}_weights.csv")
        weights_df.to_csv(weights_path, index=True)

    embed_path: Optional[str] = None
    if embed_csv:
        embed_df = get_embed_df(model.embed, idx_w)
        embed_path = str(output_dir / f"{name}_embeddings.csv")
        embed_df.to_csv(embed_path, index=True)

    # ------------------------------------------------------------------ #
    # 2.  Статические helper-функции (attention, MLP, построение графа…)
    # ------------------------------------------------------------------ #
    static_code: str = generate_static_z3_code()

    # ------------------------------------------------------------------ #
    # 3.  Предикаты attention-голов
    # ------------------------------------------------------------------ #
    predicate_blocks: list[str] = []
    for layer_idx, block in enumerate(model.blocks):
        # --- категориальные головы ---
        for head_idx in range(block.n_heads_cat):
            py_code = cat_head_to_code(
                model,
                layer=layer_idx,
                head=head_idx,
                idx_w=idx_w,
                autoregressive=autoregressive,
            )
            parsed = parse_predicate_function(py_code)
            predicate_blocks.append(generate_z3_predicate_code(parsed))

        # --- числовые головы ---
        for head_idx in range(block.n_heads_num):
            py_code = num_head_to_code(
                model,
                layer=layer_idx,
                head=head_idx,
                autoregressive=autoregressive,
            )
            parsed = parse_predicate_function(py_code)
            predicate_blocks.append(generate_z3_predicate_code(parsed))

    predicates_code: str = "\n\n".join(predicate_blocks)

    # ------------------------------------------------------------------ #
    # 4.  MLP-выражения
    # ------------------------------------------------------------------ #
    mlp_blocks: list[str] = []
    for layer_idx, block in enumerate(model.blocks):
        # --- категориальные MLP ---
        for mlp_idx in range(block.n_cat_mlps):
            py_code = cat_mlp_to_code(model, layer_idx, mlp_idx, idx_w)
            parsed = parse_mlp_function(py_code)
            mlp_blocks.append(generate_mlp_expr_code(parsed))

        # --- числовые MLP ---
        for mlp_idx in range(block.n_num_mlps):
            py_code = num_mlp_to_code(model, layer_idx, mlp_idx)
            parsed = parse_mlp_function(py_code)
            mlp_blocks.append(generate_mlp_expr_code(parsed))

    mlp_code: str = "\n\n".join(mlp_blocks)

    # ------------------------------------------------------------------ #
    # 5.  build_pipeline (на базе оригинального `run`)
    # ------------------------------------------------------------------ #
    try:
        run_src = inspect.getsource(model.run)
    except (OSError, AttributeError):
        raise RuntimeError(
            "Не удалось извлечь исходник функции `run` из модели. "
            "Убедитесь, что она определена, и Python сохранён source-код."
        )

    build_pipeline_code: str = generate_build_pipeline_from_run(run_src)

    # блок, вычисляющий исходные (обученные) предсказания
    compute_pred_code: str = generate_compute_original_predictions_code()

    # ------------------------------------------------------------------ #
    # 6.  Складываем всё воедино
    # ------------------------------------------------------------------ #
    script_parts: list[str] = [
        "# ================================================",
        "#  Auto-generated Z3 model script (DO NOT EDIT)   ",
        "# ================================================",
        "",
        "# --- Static helper functions ---",
        static_code,
        "",
        "# --- Classifier weights & embeddings ---",
    ]

    if weights_path:
        script_parts.append(f"weights = pd.read_csv(r'{weights_path}', index_col=[0,1])")

    if embed_path:
        script_parts.append(f"embeddings = pd.read_csv(r'{embed_path}', index_col=0)")

    script_parts.extend(
        [
            "",
            "# --- Attention predicates ---",
            predicates_code,
            "",
            "# --- MLP expressions ---",
            mlp_code,
            "",
            "# --- Pipeline builder ---",
            build_pipeline_code,
            "",
            "# --- Original model predictions (reference) ---",
            compute_pred_code,
            "",
        ]
    )

    full_script: str = "\n".join(script_parts)

    # ------------------------------------------------------------------ #
    # 7.  (опционально) сохраняем на диск
    # ------------------------------------------------------------------ #
    if save:
        file_path = output_dir / f"{name}.z3.py"
        file_path.write_text(full_script, encoding="utf-8")
        print(f"[z3_code_utils] Z3 script written to: {file_path.resolve()}")

    return full_script


# ---------------------------------------------------------------------------
# Утилита-обёртка «из коробки»: получить текст + сразу записать на диск
# ---------------------------------------------------------------------------
def export_model_to_Z3(
    model,
    idx_w,
    idx_t,
    *,
    output_dir: Union[str, Path] = ".",
    name: str = "program",
    **kwargs,
) -> Path:
    """
    Синтаксический сахар над `model_to_Z3`.
    Возвращает путь к созданному файлу.
    """
    output_dir = Path(output_dir)
    script = model_to_Z3(
        model,
        idx_w,
        idx_t,
        output_dir=output_dir,
        name=name,
        save=True,
        **kwargs,
    )
    return output_dir / f"{name}.z3.py"