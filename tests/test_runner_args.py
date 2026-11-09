from train_feature_map import Args, optimization_plan, select_query_mode, training_phase


def test_runner_provides_all_legacy_vatr_runtime_arguments():
    args = Args({}, num_writers=7, mode="stage1")
    required = {
        "add_noise", "all_chars", "alphabet", "batch_size", "d_lr", "device",
        "english_words_path", "g_lr", "img_height", "is_cycle", "is_kld", "is_seq",
        "no_ocr_loss", "no_writer_loss", "num_examples", "num_words", "num_writers",
        "ocr_lr", "ocr_num_classes", "freeze_ocr", "ocr_warmup_epochs", "query_input", "query_linear", "resolution", "special_alphabet",
        "tn_dec_layers", "tn_dim_feedforward", "tn_dropout", "tn_enc_layers",
        "tn_hidden_dim", "tn_nheads", "vocab_size", "w_lr", "writer_loss_weight",
        "g_steps_per_d_step", "training_schedule",
    }
    assert not sorted(name for name in required if not hasattr(args, name))
    assert args.num_examples == 1
    assert args.num_words == 3
    assert args.is_cycle is False
    assert args.ocr_num_classes == len(args.alphabet) + 1


def test_runner_accepts_writer_loss_ablation_settings():
    args = Args({"no_writer_loss": True, "writer_loss_weight": 0.25}, num_writers=7, mode="stage1")
    assert args.no_writer_loss is True
    assert args.writer_loss_weight == 0.25


def test_default_writer_free_schedule_preserves_original_vatr_cadence():
    assert optimization_plan(True, 0, {}) == ("g_ocr", "d_ocr", "g_writer", "d_writer")
    assert optimization_plan(True, 1, {}) == ("d_ocr", "d_writer")


def test_experimental_writer_free_schedule_uses_configured_g_per_d():
    cfg = {"training_schedule": "g-per-d"}
    assert optimization_plan(True, 0, cfg) == ("d_ocr", "g_ocr", "g_ocr")
    assert optimization_plan(True, 1, cfg) == ("d_ocr", "g_ocr", "g_ocr")
    assert optimization_plan(True, 27, {"training_schedule": "g-per-d", "g_steps_per_d_step": 3}) == (
        "d_ocr", "g_ocr", "g_ocr", "g_ocr"
    )


def test_writer_enabled_schedule_preserves_original_cadence():
    assert optimization_plan(False, 0, {}) == ("g_ocr", "d_ocr", "g_writer", "d_writer")
    assert optimization_plan(False, 1, {}) == ("d_ocr", "d_writer")


def test_no_glyph_cnn_changes_only_query_mode():
    assert select_query_mode("stage1", True) == "reference_baseline"
    assert select_query_mode("auto", True) == "reference_baseline"
    assert select_query_mode("stage1", False) == "stage1"


def test_ocr_warmup_phase_occupies_first_configured_total_epochs():
    assert training_phase(0, 2) == "ocr_warmup"
    assert training_phase(1, 2) == "ocr_warmup"
    assert training_phase(2, 2) == "gan"
    assert training_phase(0, 0) == "gan"
