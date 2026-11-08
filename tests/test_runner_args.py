from train_feature_map import Args, apply_vatr_original_preset, optimization_plan


def test_runner_provides_all_legacy_vatr_runtime_arguments():
    args = Args({}, num_writers=7, mode="stage1")
    required = {
        "add_noise", "all_chars", "alphabet", "batch_size", "d_lr", "device",
        "english_words_path", "g_lr", "img_height", "is_cycle", "is_kld", "is_seq",
        "no_ocr_loss", "no_writer_loss", "num_examples", "num_words", "num_writers",
        "ocr_lr", "freeze_ocr", "query_input", "query_linear", "resolution", "special_alphabet",
        "tn_dec_layers", "tn_dim_feedforward", "tn_dropout", "tn_enc_layers",
        "tn_hidden_dim", "tn_nheads", "vocab_size", "w_lr", "writer_loss_weight",
        "g_steps_per_d_step",
    }
    assert not sorted(name for name in required if not hasattr(args, name))
    assert args.num_examples == 1
    assert args.num_words == 3
    assert args.is_cycle is False


def test_runner_accepts_writer_loss_ablation_settings():
    args = Args({"no_writer_loss": True, "writer_loss_weight": 0.25}, num_writers=7, mode="stage1")
    assert args.no_writer_loss is True
    assert args.writer_loss_weight == 0.25


def test_writer_free_schedule_is_one_discriminator_then_configured_generator_steps():
    assert optimization_plan(True, 0, {}) == ("d_ocr", "g_ocr", "g_ocr")
    assert optimization_plan(True, 1, {}) == ("d_ocr", "g_ocr", "g_ocr")
    assert optimization_plan(True, 27, {"g_steps_per_d_step": 3}) == (
        "d_ocr", "g_ocr", "g_ocr", "g_ocr"
    )


def test_writer_enabled_schedule_preserves_original_cadence():
    assert optimization_plan(False, 0, {}) == ("g_ocr", "d_ocr", "g_writer", "d_writer")
    assert optimization_plan(False, 1, {}) == ("d_ocr", "d_writer")


def test_vatr_original_preset_restores_official_training_defaults():
    original = {"mode": "stage1", "num_examples": 1, "amp": True, "d_lr": 2.5e-5,
                "no_writer_loss": True}
    cfg = apply_vatr_original_preset(original)

    assert original["mode"] == "stage1"
    assert cfg["training_preset"] == "vatr-original"
    assert cfg["mode"] == "reference_baseline"
    assert cfg["num_examples"] == 15
    assert cfg["amp"] is False
    assert cfg["g_lr"] == cfg["d_lr"] == cfg["w_lr"] == cfg["ocr_lr"] == 5e-5
    assert cfg["no_writer_loss"] is False
    assert cfg["freeze_ocr"] is False
    assert cfg["num_critic_gocr_train"] == 2
    assert cfg["num_critic_docr_train"] == 1
