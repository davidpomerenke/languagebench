import asyncio
import time
from datetime import timedelta
from os import environ

import pandas as pd
from languages import languages
from models import (
    models,
    AUTO_BLOCKLIST_MIN_ATTEMPTS,
    AUTO_BLOCKLIST_FAIL_PCT_THRESHOLD,
    AUTO_BLOCKLIST_SLOW_SEC_PER_SAMPLE,
    update_blocklist_strikes,
    FatalAPIError,
)
from rich import print
from tasks import tasks
from tqdm.asyncio import tqdm_asyncio
from datasets_.util import load, save, save_local_only, get_valid_task_languages
from tqdm import tqdm

# Canonical scale used in the nightly workflow. Reduced scale (smaller
# N_LANGUAGES or N_MODELS) is OK for local validation, but pushing the
# aggregated `results` dataset back to HF in that mode would truncate the
# published table — see CANONICAL_*_FOR_PUSH and the guard in save() below.
CANONICAL_N_LANGUAGES_FOR_PUSH = 1000
CANONICAL_N_MODELS_FOR_PUSH = 100  # nightly uses 150; bar is "covers the full cohort"

n_sentences = int(environ.get("N_SENTENCES", 10))
n_languages = int(environ.get("N_LANGUAGES", 1000))
n_models = int(environ.get("N_MODELS", 40))

# Optional wall-clock budget (seconds). GitHub-hosted runners hard-cap a job at
# 6h regardless of `timeout-minutes`; a run cancelled at that wall shows as
# "failed" and skips the post-run Space restart. With per-model checkpointing
# we can instead stop GRACEFULLY after the current model once the budget is hit
# — the run exits 0, the Space restarts with whatever finished, and the next
# run resumes the rest (completed models are skipped). 0 = no limit (local runs).
max_runtime_seconds = int(environ.get("MAX_RUNTIME_SECONDS", 0))

# Optional cap on how many NOT-YET-COMPLETE models a single run may work on.
# The weekly CI run is meant for incremental additions ("one new model landed,
# evaluate it"), not for chewing through a large backlog: every checkpoint
# re-pushes the whole results-detailed log, `all_results` is held entirely in
# RAM on a 16GB runner, and a crash re-costs the in-flight model (CI has no
# sample-level joblib cache, that's local-disk only). Large backfills belong on
# a local machine, or in a `workflow_dispatch` run with the cap explicitly
# raised. 0 = unlimited (the default, so local runs are unaffected).
max_new_models_per_run = int(environ.get("MAX_NEW_MODELS_PER_RUN", 0))

# Skip ALL HuggingFace pushes and write snapshots to results/dry-run/ instead.
# For exercising this file end-to-end (e.g. verifying the crash-path checkpoint)
# without a multi-hundred-MB upload per attempt. The separate directory matters:
# results/{results,models,languages}.json are tracked in git, so writing a
# partial-scale aggregate over them would dirty committed files on every test.
# This also suppresses the results-detailed push, which the
# ALLOW_HF_PUSH_RESULTS guard below deliberately does not.
# NOT for real runs: those must push, that is both how results get published
# and how the next run resumes.
DRY_RUN = environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

# When n_languages or n_models is smaller than canonical, the filter in
# `results_agg` below would discard most rows and overwrite the public HF
# aggregate. Detect that and downgrade to local-only writes.
ALLOW_HF_PUSH_RESULTS = (
    n_languages >= CANONICAL_N_LANGUAGES_FOR_PUSH
    and n_models >= CANONICAL_N_MODELS_FOR_PUSH
    and not DRY_RUN
)

def publishable_models(log_df, covered_models):
    """Models safe to include in the published aggregate.

    `covered_models` must already be COVERAGE-COMPLETE (every expected
    task × language × sentence attempted). The caller guarantees this — a
    model is only added once its full matrix is done. Coverage is the load-
    bearing guard: a sparsely-evaluated model (e.g. 5 of 200 languages) would
    otherwise show an inflated mean and jump the leaderboard, which is exactly
    the failure that once put a small model at rank #1.

    On top of coverage we drop models that are *broken* (mostly errors over a
    meaningful sample) rather than merely low-scoring — auto_blocklist removes
    them from the cohort on the next run."""
    if "status" not in log_df.columns:
        return set(covered_models)
    keep = set()
    in_scope = log_df[log_df["model"].isin(covered_models)]
    for mid, grp in in_scope.groupby("model"):
        total = len(grp)
        failed = (grp["status"] == "error").sum()
        if (total >= AUTO_BLOCKLIST_MIN_ATTEMPTS
                and failed / total * 100 >= AUTO_BLOCKLIST_FAIL_PCT_THRESHOLD):
            continue
        keep.add(mid)
    return keep


def checkpoint(log_df, covered_models, cohort_languages, note):
    """Push the immutable log + the aggregate (healthy, fully-covered models
    only) to HuggingFace (or locally if below scale)."""
    if "status" in log_df.columns:
        valid = log_df[log_df["status"].isna() | (log_df["status"] == "ok")]
    else:
        valid = log_df
    keep = publishable_models(log_df, covered_models)
    agg = (
        valid[valid["model"].isin(keep) & valid["bcp_47"].isin(cohort_languages)]
        .groupby(["model", "bcp_47", "task", "metric"])
        .agg({"score": "mean", "origin": "first"})
        .reset_index()
    )
    def write(df, fname, allow_hf):
        # DRY_RUN wins over everything and writes under results/dry-run/, because
        # results/{results,models,languages}.json are TRACKED IN GIT, writing a
        # partial-scale aggregate over them would dirty committed files on every
        # test run.
        if DRY_RUN:
            save_local_only(df, f"dry-run/{fname}")
        elif allow_hf:
            save(df, fname)
        else:
            save_local_only(df, fname)

    # results-detailed is append-merged (immutable log); safe at any scale, so
    # it pushes even from a partial-scale run. DRY_RUN is the only thing that
    # suppresses it, for local verification of this file.
    write(log_df, "results-detailed", allow_hf=True)
    # The aggregated tables are filtered by cohort_models × cohort_languages, so
    # a partial-scale run would truncate the published view — push only from a
    # full-scale run; otherwise local-only.
    write(agg, "results", allow_hf=ALLOW_HF_PUSH_RESULTS)
    write(models, "models", allow_hf=ALLOW_HF_PUSH_RESULTS)
    write(languages, "languages", allow_hf=ALLOW_HF_PUSH_RESULTS)
    if DRY_RUN:
        destination = "DRY_RUN: results/dry-run/ only, nothing pushed"
    elif ALLOW_HF_PUSH_RESULTS:
        destination = "HF"
    else:
        destination = "results-detailed → HF, aggregates local-only"
    print(f"  ✓ checkpoint after {note}: {len(keep)} models published, "
          f"{len(log_df)} detailed rows ({destination})")
    return agg


async def evaluate():
    start_time = time.time()

    # Pre-compute model tasks to avoid O(n²) lookups
    model_tasks = models.set_index("id")["tasks"].to_dict()
    
    # Pre-compute valid languages for each task
    valid_task_langs = {task_name: get_valid_task_languages(task_name) for task_name in tasks}
    
    # get all combinations that need evaluation (filtering invalid lang×task combos)
    combis = [
        (task_name, model, lang.bcp_47, i)
        for i in range(n_sentences)
        for lang in languages.head(n_languages).itertuples()
        for task_name, task in tasks.items()
        for model in models.iloc[:n_models]["id"]
        if task_name in model_tasks[model] and lang.bcp_47 in valid_task_langs[task_name]
    ]
    combis = pd.DataFrame(combis, columns=["task", "model", "bcp_47", "sentence_nr"])

    # Load cached results and filter out completed combinations
    old_results = load("results-detailed")

    # COVERAGE-COMPLETENESS is about what has been ATTEMPTED (any status), not
    # what succeeded, matching `covered.add(model_id)` below, which fires once a
    # model's matrix has been attempted regardless of pass/fail. Computed here
    # against the FULL matrix, before the pending filter strips rows.
    #
    # Why it can't just be "has no pending combis": a failed combo is retried
    # every run (only status=="ok" counts as done), so a model with a handful of
    # permanently-failing combos would never be coverage-complete and would be
    # dropped from the published aggregate, which `checkpoint()` rebuilds from
    # `covered` each time. Uncapped that never showed, because every model was
    # attempted every run. With MAX_NEW_MODELS_PER_RUN it would silently
    # un-publish every model the run didn't reach.
    key_cols = ["task", "model", "bcp_47", "sentence_nr"]
    if not old_results.empty:
        seen = old_results[key_cols].drop_duplicates()
        seen["_attempted"] = True
        merged = combis.merge(seen, on=key_cols, how="left")
        unattempted_per_model = merged[merged["_attempted"].isna()].groupby("model").size()
    else:
        # No prior log: NOTHING has been attempted, so every model is short its
        # entire matrix. Must not be an empty Series, `.get(m, 0)` would then
        # report 0 unattempted for every model and mark the whole cohort
        # coverage-complete, publishing models with no data at all.
        unattempted_per_model = combis.groupby("model").size()

    if not old_results.empty:
        # Only treat status==\"ok\" (or missing status) as completed.
        if "status" in old_results.columns:
            ok_mask = old_results["status"].isna() | (old_results["status"] == "ok")
            completed_df = old_results.loc[ok_mask, key_cols]
        else:
            completed_df = old_results[key_cols]
        completed = set(completed_df.apply(tuple, axis=1))
        combis = combis[~combis.apply(lambda row: tuple(row) in completed, axis=1)]

    print(f"Running {len(combis)} evaluation tasks across {combis['model'].nunique()} models...")

    current_models = set(models.iloc[:n_models]["id"])
    current_languages = set(languages.head(n_languages)["bcp_47"])

    # We evaluate ONE MODEL AT A TIME and checkpoint to HuggingFace after each
    # model finishes its ENTIRE matrix (every task × language × sentence
    # attempted). This buys two things:
    #   1. Progress survives interruption — the GitHub-hosted runner's hard 6h
    #      cap, or a local laptop sleeping. The next run skips models already
    #      fully logged (status=="ok" rows), so a large onboarding completes
    #      across however many runs it takes instead of losing everything.
    #   2. A model only enters the PUBLISHED aggregate once it has full
    #      benchmark coverage, so a half-evaluated model can never show an
    #      inflated score from a sparse sample (the failure mode that once put
    #      a small model at rank #1).
    all_results = old_results.copy() if not old_results.empty else pd.DataFrame(
        columns=["task", "model", "bcp_47", "metric", "sentence_nr", "score", "origin", "status"]
    )
    dedup_keys = ["task", "model", "bcp_47", "metric", "sentence_nr"]
    # Also the crash/budget salvage granularity: progress is preserved to the
    # last COMPLETED batch boundary, so smaller batches lose less on an
    # interruption (at the cost of more frequent Python-side bookkeeping).
    # Overridable mainly so the salvage path can be exercised in a small
    # local DRY_RUN without issuing 2000 real calls.
    batch_size = int(environ.get("BATCH_SIZE", 2000))

    # One entry per model with outstanding work, so each checkpoint boundary is a
    # fully-computed model. Models already fully cached have no pending combis
    # and are skipped. (Re-ordered just below, see the sort.)
    pending_models = [m for m in models.iloc[:n_models]["id"].tolist()
                      if (combis["model"] == m).any()]

    # A model is COVERAGE-COMPLETE once every combo in its matrix has been
    # ATTEMPTED, in this run or any earlier one. Only these may be published,
    # which is what stops a model with a handful of evaluated languages from
    # showing an inflated mean and jumping the leaderboard. Models that failed
    # some combos stay covered (and published); their retries are still queued in
    # pending_models, so a model can legitimately be in BOTH sets.
    covered = {m for m in current_models if int(unattempted_per_model.get(m, 0)) == 0}
    print(f"{len(covered)} models coverage-complete (publishable); "
          f"{len(pending_models)} with outstanding work this run")

    # Ordering matters once MAX_NEW_MODELS_PER_RUN truncates this list.
    #   1. Brand-new models (nothing in the log yet) go FIRST, onboarding them
    #      is the entire point of an incremental run.
    #   2. Then partially-done models, closest-to-complete first, so a model left
    #      half-evaluated by a crash or budget stop is finished (and becomes
    #      publishable) before another is started.
    # Sorting purely by pending-count would invert this: a new model has the
    # WHOLE matrix pending, so it would sort last and be starved behind older
    # models whose errored combos stay pending forever (only status=="ok"
    # counts as done, so failures are re-attempted every run).
    seen_models = set(old_results["model"].unique()) if not old_results.empty else set()
    pending_counts = combis["model"].value_counts()
    pending_models.sort(key=lambda m: (m in seen_models, pending_counts.get(m, 0)))

    # Truncate to the cap. What keeps this safe is that `covered` is derived from
    # `unattempted_per_model` (the full matrix vs the log) and NOT from
    # `pending_models`, so deferring a model cannot reclassify it as
    # coverage-complete. Keep it that way: the earlier
    # `covered = current_models - set(pending_models)` would have published every
    # deferred model on sparse or zero data, the inflation bug the coverage gate
    # exists to prevent.
    deferred_models = set()
    if max_new_models_per_run and len(pending_models) > max_new_models_per_run:
        deferred_models = set(pending_models[max_new_models_per_run:])
        pending_models = pending_models[:max_new_models_per_run]
        # NB: this module does `from rich import print`, which parses "[tag]" as
        # markup and silently eats it. Escape as "\[" so the prefix survives;
        # these are the lines you grep for in a CI log.
        print(rf"\[cap] MAX_NEW_MODELS_PER_RUN={max_new_models_per_run}: running "
              f"{len(pending_models)} model(s) this run, {len(deferred_models)} "
              f"deferred to a later run. Raise the cap (or run locally) to "
              f"clear a backlog faster.")

    def over_budget():
        return max_runtime_seconds and (time.time() - start_time) > max_runtime_seconds

    results_agg = None
    budget_hit = False
    # Models that ran slowly this run (heavily rate-limited). If they're also
    # failing, they're excluded without a grace re-attempt — retrying an
    # expensive failure isn't worth the time/money.
    slow_models = set()
    for mi, model_id in enumerate(pending_models, 1):
        # Don't START a new model once over budget — stop cleanly between models.
        if over_budget():
            print(rf"\[budget] {max_runtime_seconds}s reached before {model_id}; "
                  f"{len(pending_models) - mi + 1} model(s) deferred to next run.")
            break
        model_combis = combis[combis["model"] == model_id]
        print(f"[{mi}/{len(pending_models)}] {model_id}: {len(model_combis)} new samples")
        model_out = []
        model_started = time.time()
        attempted = 0
        try:
            for i in tqdm(range(0, len(model_combis), batch_size),
                          colour="blue", desc=model_id):
                batch = model_combis.iloc[i:i + batch_size]
                attempted += len(batch)
                rows = [(t, m, b, s) for _, (t, m, b, s) in batch.iterrows()]
                # A single combo that throws (e.g. a flaky HF dataset download) must
                # NOT crash a multi-hour run. tqdm_asyncio.gather has no
                # return_exceptions, so wrap each task: catch non-fatal errors and
                # return them as a value; the combo is then skipped (stays pending,
                # re-attempted next run) without counting toward the model's
                # blocklist strikes. FatalAPIError (account-level) still propagates.
                async def _safe(t, m, b, s):
                    try:
                        return await tasks[t](m, b, s)
                    except (FatalAPIError, KeyboardInterrupt, asyncio.CancelledError):
                        # Control-flow signals, not task failures. KeyboardInterrupt
                        # is raised inside whatever coroutine is running when the
                        # signal lands, so a plain BaseException catch swallows it
                        # and Ctrl-C just marks combos "skipped" while the run
                        # carries on. Re-raise so the run actually aborts and the
                        # salvage handler below can checkpoint completed batches.
                        raise
                    except BaseException as e:  # noqa: BLE001 - intentional catch-all
                        return e
                batch_res = await tqdm_asyncio.gather(*[_safe(t, m, b, s) for (t, m, b, s) in rows])
                for (t, m, b, s), res in zip(rows, batch_res):
                    if isinstance(res, BaseException):
                        print(f"  ! {t}/{m}/{b}#{s} skipped: {type(res).__name__}: {str(res)[:120]}")
                    else:
                        model_out.extend(res)
                # Budget can be hit MID-model (a rate-limited model can take >>1h).
                # Stop at this batch boundary so we never blow past the 6h hard cap.
                if over_budget():
                    budget_hit = True
                    break
        except BaseException as crash:
            # Salvage work that has ALREADY BEEN PAID FOR before unwinding.
            # `model_out` is only folded into `all_results` after this loop, so
            # without this handler a FatalAPIError (exhausted key), an OOM, or a
            # local Ctrl-C silently discards every COMPLETED BATCH for this
            # model. That is exactly what cost the 2026-06-15 run ~4.5h of
            # already-billed results.
            #
            # Granularity is the batch, not the sample: `model_out.extend()`
            # only runs once a batch's gather() returns, so an interrupted batch
            # contributes nothing. That's the right trade, the thing that kills
            # a batch is FatalAPIError, which fails every in-flight call in it
            # anyway, but it means a model is salvaged to the last completed
            # batch boundary, not to the last completed sample.
            #
            # BaseException on purpose: also covers KeyboardInterrupt and
            # asyncio.CancelledError. It does NOT cover SIGKILL/SIGTERM from the
            # runner, MAX_RUNTIME_SECONDS is what keeps us clear of that.
            #
            # model_id is deliberately NOT added to `covered` here. A brand-new
            # model therefore stays unpublished until some run attempts its full
            # matrix, which is the sparse-coverage guard. A model already covered
            # by earlier runs stays published, correctly: its aggregate is built
            # from a complete attempt, and this run only added more ok rows.
            #
            # "\\[" so rich's markup parser doesn't eat the "[crash]" prefix.
            print(f"\n\\[crash] {type(crash).__name__} during {model_id}; "
                  f"saving {len(model_out)} completed rows so the next run can "
                  f"resume (do not interrupt again)...")
            try:
                if model_out:
                    all_results = pd.concat(
                        [all_results, pd.DataFrame(model_out)]
                    ).drop_duplicates(subset=dedup_keys, keep="last")
                    checkpoint(all_results, covered, current_languages,
                               f"crash during {model_id}")
            except Exception as save_err:
                # Never let a failing salvage mask the original failure.
                print(rf"\[crash] could not save partial progress: {save_err}")
            raise
        # Flag the model as slow if it took disproportionately long per sample
        # this run (rate-limited). Needs a meaningful sample count to be reliable.
        sec_per_sample = (time.time() - model_started) / max(attempted, 1)
        if attempted >= 100 and sec_per_sample > AUTO_BLOCKLIST_SLOW_SEC_PER_SAMPLE:
            slow_models.add(model_id)
            print(f"  ⏱ {model_id}: {sec_per_sample:.2f}s/sample (slow / rate-limited)")

        model_df = pd.DataFrame(model_out) if model_out else pd.DataFrame(columns=all_results.columns)
        # Persist whatever completed (full model, or partial batches if the
        # budget was hit). Partial work is saved to results-detailed so its
        # done combos are skipped next run — but the model is NOT added to
        # `covered`, so a partial model never enters the published aggregate.
        all_results = pd.concat([all_results, model_df]).drop_duplicates(
            subset=dedup_keys, keep="last"
        )
        if budget_hit:
            print(rf"\[budget] {max_runtime_seconds}s reached mid-{model_id}; saved partial "
                  f"progress, {len(pending_models) - mi} model(s) deferred to next run.")
            results_agg = checkpoint(all_results, covered, current_languages, f"budget stop @ {model_id}")
            break

        if not model_df.empty and "status" in model_df.columns:
            err = (model_df["status"] != "ok").mean()
            if err > 0.8:
                # Logged so auto_blocklist sees it; publish-health filter keeps
                # it out of the aggregate this run.
                print(f"  ⚠ {model_id}: {err:.0%} of new rows errored — logged, not published")

        # This model's full matrix is now attempted → coverage-complete.
        covered.add(model_id)
        results_agg = checkpoint(all_results, covered, current_languages, model_id)

    if results_agg is None:
        # Everything was already cached — still refresh the published tables
        # from the existing log (e.g. cohort/cost metadata may have changed).
        results_agg = checkpoint(all_results, covered, current_languages, "no new work")

    # Update consecutive-bad-run strikes (persisted to HF) so a model is only
    # auto-blocklisted after staying broken across runs — not after a single run
    # where a provider may have rate-limited us. Non-fatal if it fails.
    try:
        update_blocklist_strikes(all_results, slow_models=slow_models,
                                 not_attempted=deferred_models, dry_run=DRY_RUN)
    except Exception as e:
        print(rf"\[main] could not update blocklist strikes: {e}")

    elapsed = time.time() - start_time
    print(f"Evaluation completed in {str(timedelta(seconds=int(elapsed)))}")


if __name__ == "__main__":
    results = asyncio.run(evaluate())
