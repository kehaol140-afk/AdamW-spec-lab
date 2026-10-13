# V15 源码证据（原始文件行号）
以下由上传压缩包静态抽取，未在本机GPU复现。

## GpuContext仅初始化模型，没有加载父checkpoint
```text
680: class GpuContext:
681:     def __enter__(self):
682:         resource("before_cuda_model_init", admission=True)
683:         if (ROOT / "artifacts" / "run.lock").exists() or (HERE / "run.lock").exists():
684:             raise RuntimeError("RUN_LOCK_PRESENT")
685:         (HERE / "run.lock").write_text(str(os.getpid()), encoding="utf-8")
686:         try:
687:             if not torch.cuda.is_available():
688:                 raise RuntimeError("CUDA_UNAVAILABLE")
689:             free, _ = torch.cuda.mem_get_info(0)
690:             runtime = PCFG["runtime"]
691:             vram_limit = min(runtime["vram_soft_limit_gib"] * 2 ** 30,
692:                              free - VRAM_RESERVE_OVERRIDE_GIB * 2 ** 30)
693:             if vram_limit <= 0:
694:                 raise RuntimeError("VRAM_RESERVE_FAILED")
695:             VRAM[0] = vram_limit
696:             torch.set_num_threads(runtime["cpu_threads"])
697:             torch.backends.cuda.matmul.allow_tf32 = False
698:             torch.backends.cudnn.allow_tf32 = False
699:             from pilot_core import model_and_optimizer
700:             binding = read(ROOT / "bindings.local.json")
701:             self.model, self.opt, self.aliases, _ = model_and_optimizer(PCFG, binding, PEAK, "cuda:0")
702:             self.mem = np.memmap(ROOT / "artifacts" / "train_tokens.u32", dtype=np.uint32, mode="r")
703:             resource("after_model_init", vram_limit=vram_limit)
704:         except BaseException:
705:             VRAM[0] = None
706:             (HERE / "run.lock").unlink(missing_ok=True)
707:             raise
708:         return self
709: 
```

## dev H0在首次load前读取
```text
850:                   "optimizer_step_attempts_total": attempts_total}
851:         write(HERE / "smoke_result.json", result)
852:         remove_new_files(watched, [HERE / "checkpoints", HERE / "raw"])
853:         print(json.dumps(result))
854: 
855: 
856: def phase_dev():
857:     verify_manifest()
858:     if (HERE / "selection_frozen.json").exists():
859:         raise RuntimeError("DEVELOPMENT_ALREADY_FROZEN")
860:     with GpuContext() as ctx:
861:         evaluate_split(ctx.model, dev_views(), "dev", "dev_h0", "H0_step256", 0)
862:         horizons = {h: ["dev"] for h in CFG["eval_horizons"]}
863:         for spec in branch_specs():
864:             run_branch(ctx.model, ctx.opt, ctx.aliases, ctx.mem, spec,
865:                        CFG["main_checkpoint_step"], horizons, one_shot=False, phase="dev")
866: 
867: 
```

## confirm H0与aux H0读取位置
```text
894: def phase_confirm():
895:     manifest = verify_manifest()
896:     frozen = confirm_allowed()
897:     if not frozen.get("confirmation_opened"):
898:         frozen["confirmation_opened"] = True
899:         write(HERE / "selection_frozen.json", frozen)
900:     registry = specs_by_id()
901:     main_ids = list(dict.fromkeys([frozen["selected"]["S"], frozen["selected"]["F"],
902:                                    frozen["selected"]["P"], "S_a1_b1"]))
903:     horizons = {1: ["dev", "confirm"], 16: ["dev", "confirm"], 64: ["dev", "confirm"]}
904:     with GpuContext() as ctx:
905:         evaluate_split(ctx.model, confirm_views(), "confirm", "confirm_h0", "H0_step256", 0)
906:         for cid in main_ids:
907:             load_h1_for_eval(ctx.model, cid)
908:             evaluate_split(ctx.model, confirm_views(), "confirm", "confirm_main_h1", cid, 1)
909:             load_h64_for_eval(ctx.model, ctx.opt, cid, registry[cid], CFG["main_checkpoint_step"])
910:             evaluate_split(ctx.model, confirm_views(), "confirm", "confirm_main_h64", cid, 64)
911:         for family in ("F", "P"):
912:             spec = registry[frozen["selected"][family]]
913:             matched = f"S_a{spec['a']:g}_b{spec['b']:g}"
914:             load_h64_for_eval(ctx.model, ctx.opt, matched, registry[matched], CFG["main_checkpoint_step"])
915:             evaluate_split(ctx.model, confirm_views(), "confirm", "confirm_matched_s",
916:                            matched + "__matchedS", 64)
917:         for family in ("F", "P"):
918:             spec = registry[frozen["selected"][family]]
919:             label = run_branch(ctx.model, ctx.opt, ctx.aliases, ctx.mem, spec,
920:                                CFG["main_checkpoint_step"], horizons, one_shot=True, phase="oneshot")
921:             fresh_h1, os_h1 = eval_rows(spec["id"], 1, "dev"), eval_rows(label, 1, "dev")
922:             if [r["doc_id"] for r in fresh_h1] != [r["doc_id"] for r in os_h1]:
923:                 raise RuntimeError("ONESHOT_H1_PAIRING_MISMATCH")
924:             delta = abs(token_weighted(fresh_h1) - token_weighted(os_h1))
925:             append(HERE / "logs" / "oneshot_h1_reuse.jsonl",
926:                    {"branch": label, "fresh": spec["id"], "h1_dev_nll_delta": delta,
927:                     "reuse_verified": delta <= PCFG["numerics"]["fp32_atol"]})
928:         if manifest["aux_checkpoint_available"]:
929:             evaluate_split(ctx.model, confirm_views(), "confirm", "confirm_h0_aux", "H0_step128", 0)
930:             evaluate_split(ctx.model, dev_views(), "dev", "dev_h0_aux", "H0_step128", 0)
931:             for cid in main_ids:
932:                 run_branch(ctx.model, ctx.opt, ctx.aliases, ctx.mem, registry[cid],
933:                            CFG["aux_checkpoint_step"], horizons, one_shot=False,
934:                            phase="aux", label_prefix="aux_")
935:         else:
936:             write(HERE / "aux_status.json", {"status": "NOT_RUN", "reason": "step128 checkpoint absent"})
937: 
938: 
```

## 本分支A范数匹配与更新公式
```text
415: def apply_step(model, opt, family, a, b, phase, eta, timing, target_name=TARGET):
416:     """One optimizer step: native opt.step() then verified displacement replacement."""
417:     named = dict(model.named_parameters())
418:     beta1, beta2 = PCFG["baseline"]["betas"]
419:     eps = PCFG["baseline"]["eps"]
420:     A_map, meta = {}, {"family_used": family}
421:     for name, param in named.items():
422:         if param.grad is None:
423:             A_map[name] = None
424:             continue
425:         state = opt.state[param]
426:         if "step" in state:
427:             step_plus = int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"]) + 1
428:             m_prev, v_prev = state["exp_avg"], state["exp_avg_sq"]
429:         else:
430:             step_plus = 1
431:             m_prev = torch.zeros_like(param)
432:             v_prev = torch.zeros_like(param)
433:         m_plus = m_prev * beta1 + param.grad * (1 - beta1)
434:         v_plus = v_prev * beta2 + param.grad.square() * (1 - beta2)
435:         A_map[name] = (m_plus / (1 - beta1 ** step_plus)) / (
436:             (v_plus / (1 - beta2 ** step_plus)).sqrt() + eps)
437:         if name == target_name:
438:             meta["target_moments_step_plus"] = step_plus
439:             meta["_target_v_plus"] = v_plus
440:     target_A = A_map[target_name]
441:     if target_A is None:
442:         raise RuntimeError("TARGET_GRADIENT_MISSING")
443:     meta["A_norm"] = float(torch.linalg.vector_norm(target_A.double()))
444:     if family in ("F", "P"):
445:         g_clip = named[target_name].grad.detach()
446:         if family == "P":
447:             step_plus = meta["target_moments_step_plus"]
448:             x = g_clip / ((meta["_target_v_plus"] / (1 - beta2 ** step_plus)).sqrt() + eps)
449:         else:
450:             x = g_clip
451:         D, direction_meta = n_a(x, meta["A_norm"])
452:         meta.update(direction_meta)
453:     else:
454:         D = target_A
455:         meta.update({"zero_norm_fallback": False, "input_norm": meta["A_norm"]})
456:     if not torch.isfinite(D).all() or not torch.isfinite(target_A).all():
457:         raise RuntimeError("NON_FINITE_DIRECTION")
458:     meta.pop("_target_v_plus", None)
459:     resource(f"{phase}_before_optstep", vram_limit=VRAM[0])
460:     begin_optimizer(phase)
461:     started = time.perf_counter()
462:     opt.step()
463:     complete_optimizer()
464:     torch.cuda.synchronize()
465:     timing["optimizer_seconds"] = time.perf_counter() - started
466:     with torch.no_grad():
467:         for name, param in named.items():
468:             A_p = A_map[name]
469:             if A_p is None:
470:                 continue
471:             if name == target_name:
472:                 if not (family == "S" and a == 1.0):
473:                     param.add_(eta * (target_A - a * D))
474:             elif b != 1.0:
475:                 param.add_(eta * (1.0 - b) * A_p)
476:     torch.cuda.synchronize()
477:     meta["D_norm"] = float(torch.linalg.vector_norm(D.double()))
478:     meta["match_multiplier"] = meta["A_norm"] / meta["input_norm"] if meta["input_norm"] else None
479:     return meta
480: 
```

## 真实分支起点加载
```text
569: def run_branch(model, opt, aliases, mem, spec, payload_step, splits_at_horizon, one_shot=False,
570:                phase="dev", steps=None, label_prefix=""):
571:     from pilot_core import load_checkpoint, loss_sum_count, schedule_lr, training_batch
572:     steps = steps or CFG["horizon"]
573:     label = f"{label_prefix}{spec['id']}" + ("_oneshot" if one_shot else "")
574:     checkpoint = ROOT / "artifacts" / "checkpoints" / "seed3407_lr0.003" / f"step{payload_step}.pt"
575:     rolling = HERE / "checkpoints" / f"{label}_rolling.pt"
576:     final = HERE / "checkpoints" / f"{label}_h64.pt"
577:     if final.exists():
578:         return label
579:     if rolling.exists():
580:         state = load_branch_state(rolling, model, opt, label, payload_step)
581:         branch_step = state["branch_step"]
582:         one_shot_applied = state["one_shot_applied"]
583:         cursor = state["cursor"]
584:         total_tokens = state["total_effective_tokens"]
585:     else:
586:         payload = load_checkpoint(checkpoint, model, opt,
587:                                   config_hash=sha(ROOT / "config" / "local_v1_lowmem.json"),
588:                                   data_hash=read(ROOT / "artifacts" / "data_manifest.json")["data_hash"],
589:                                   aliases=aliases)
590:         if payload["successful_optimizer_step"] != payload_step:
591:             raise RuntimeError("CHECKPOINT_STEP_CHANGED")
592:         branch_step, one_shot_applied = 0, False
593:         cursor = payload["data_cursor"]
594:         total_tokens = payload["total_effective_tokens"]
595:     step_log = HERE / "raw" / "steps" / f"{label}.jsonl"
596:     done = ({json.loads(line)["branch_step"] for line in step_log.open(encoding="utf-8")}
597:             if step_log.exists() else set())
598:     seq = PCFG["runtime"]["sequence_length"]
599:     mb = PCFG["runtime"]["micro_batch_size"]
600:     accum = PCFG["runtime"]["accumulation_steps"]
```

## H1与H64终点评估加载
```text
716: def load_h1_for_eval(model, label):
717:     snapshot = torch.load(HERE / "checkpoints" / f"{label}_h1_snapshot.pt",
718:                           map_location="cpu", weights_only=False)
719:     if snapshot["label"] != label or snapshot["manifest_sha256"] != sha(HERE / "run_manifest.json"):
720:         raise RuntimeError("H1_SNAPSHOT_IDENTITY_MISMATCH")
721:     with torch.no_grad():
722:         for name, param in model.named_parameters():
723:             param.copy_(snapshot["params"][name].to(param.device, param.dtype))
724:         for name, buffer in model.named_buffers():
725:             buffer.copy_(snapshot["buffers"][name].to(buffer.device))
726: 
727: 
728: def load_h64_for_eval(model, opt, label, spec, payload_step):
729:     state = load_branch_state(HERE / "checkpoints" / f"{label}_h64.pt", model, opt, label, payload_step)
730:     if state["spec"] != spec or state["branch_step"] != CFG["horizon"]:
731:         raise RuntimeError("H64_ENDPOINT_IDENTITY_MISMATCH")
732: 
733: 
```

## 轻量打包范围
```text
1090:     (HERE / "V15_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
1091: 
1092: 
1093: def package_light():
1094:     names = ["train.py", "test_v15.py", "config.json", "RUN_VERIFIED.ps1", "run_manifest.json",
1095:              "parent_dependencies.json", "budget_ledger.json", "selection_frozen.json",
1096:              "dev_scores.json", "findings.json", "V15_FINDINGS.md", "smoke_result.json",
1097:              "confirm_views.jsonl", "confirm_documents.csv"]
1098:     for folder in ("raw", "logs"):
1099:         base = HERE / folder
1100:         if base.exists():
1101:             names += [str(p.relative_to(HERE)) for p in sorted(base.rglob("*"))
1102:                       if p.is_file() and p.suffix in (".jsonl", ".json", ".log", ".csv", ".txt")]
1103:     names = [n for n in names if (HERE / n).is_file()]
1104:     with zipfile.ZipFile(HERE / "v15_light.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
1105:         for name in names:
1106:             archive.write(HERE / name, name)
1107:     with zipfile.ZipFile(HERE / "v15_light.zip") as archive:
1108:         listed = set(archive.namelist())
1109:     for key in ("budget_ledger.json", "findings.json", "V15_FINDINGS.md", "selection_frozen.json"):
1110:         if key not in listed:
1111:             raise RuntimeError("LIGHT_PACKAGE_INCOMPLETE:" + key)
1112:     return HERE / "v15_light.zip"
1113: 
1114: 
```

## 父pilot_core.py模型初始化（不读取权重）
```text
613: def model_and_optimizer(cfg: dict, binding: dict, peak_lr: float, device: str):
614:     from transformers import AutoTokenizer
615:     repo = Path(binding["repo_root"])
616:     if str(repo) not in sys.path:
617:         sys.path.insert(0, str(repo))
618:     from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
619:     tok = AutoTokenizer.from_pretrained(binding["tokenizer_path"], local_files_only=True)
620:     if max(tok.get_vocab().values()) >= len(tok):
621:         raise RuntimeError("tokenizer ID/vocab mismatch")
622:     c = cfg["model"]
623:     mc = MiniMindConfig(hidden_size=c["hidden_size"], num_hidden_layers=c["num_hidden_layers"],
624:                         intermediate_size=c["intermediate_size"], num_attention_heads=c["num_attention_heads"],
625:                         num_key_value_heads=c["num_key_value_heads"], head_dim=c["head_dim"],
626:                         dropout=c["dropout"], use_moe=False, use_cache=False, vocab_size=len(tok),
627:                         bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
628:                         max_position_embeddings=cfg["runtime"]["sequence_length"], flash_attn=True,
629:                         tie_word_embeddings=True)
630:     model = MiniMindForCausalLM(mc).float()
631:     if model.model.embed_tokens.weight is not model.lm_head.weight:
632:         raise RuntimeError("MiniMind embedding tying unresolved")
633:     count = sum(p.numel() for p in model.parameters())
634:     if count > c["max_parameters"]:
635:         raise RuntimeError(f"MODEL_SIZE_UNRESOLVED: measured parameters={count}")
636:     target = model.model.layers[1].mlp.down_proj.weight
637:     attn = model.model.layers[1].self_attn.o_proj.weight
638:     if tuple(target.shape) != (c["hidden_size"], c["intermediate_size"]) or tuple(attn.shape) != (c["hidden_size"], c["hidden_size"]):
639:         raise RuntimeError("target matrix shape unresolved")
640:     aliases = {}
641:     for name, p in model.named_parameters(remove_duplicate=False):
642:         aliases.setdefault(id(p), []).append(name)
643:     alias_map = {names[0]: names for names in aliases.values()}
644:     groups = [[], []]
645:     for name, p in model.named_parameters():
646:         groups[0 if p.ndim >= 2 else 1].append(p)
647:     bc = cfg["baseline"]
648:     opt = torch.optim.AdamW([{"params": groups[0], "weight_decay": bc["matrix_weight_decay"]},
649:                              {"params": groups[1], "weight_decay": 0.0}],
650:                             lr=peak_lr, betas=tuple(bc["betas"]), eps=bc["eps"],
651:                             amsgrad=False, maximize=False, foreach=False, fused=False)
652:     return model.to(device), opt, alias_map, count
653: 
654: 
```
