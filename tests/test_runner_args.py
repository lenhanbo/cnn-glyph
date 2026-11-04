from train_feature_map import Args


def test_runner_provides_all_legacy_vatr_runtime_arguments():
    args = Args({}, num_writers=7, mode="stage1")
    required = {
        "add_noise", "all_chars", "alphabet", "batch_size", "d_lr", "device",
        "english_words_path", "g_lr", "img_height", "is_cycle", "is_kld", "is_seq",
        "no_ocr_loss", "no_writer_loss", "num_examples", "num_words", "num_writers",
        "ocr_lr", "query_input", "query_linear", "resolution", "special_alphabet",
        "tn_dec_layers", "tn_dim_feedforward", "tn_dropout", "tn_enc_layers",
        "tn_hidden_dim", "tn_nheads", "vocab_size", "w_lr", "writer_loss_weight",
    }
    assert not sorted(name for name in required if not hasattr(args, name))
    assert args.num_examples == 1
    assert args.num_words == 3
    assert args.is_cycle is False
