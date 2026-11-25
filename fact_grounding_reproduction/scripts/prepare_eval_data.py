import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
# Excluded after inspecting the released support sentence and question relation.
MANUAL_EXCLUSIONS = {
    "5a78b386554299148911f949",  # evidence mentions the title but not the asked purchase relation
    "5a7e31c055429934daa2fbee",  # requires unstated geographic comparison
    "5ae3fd2c5542995dadf2428f",  # musician label does not establish the person is not also an actor
    "5ae3790f5542992e3233c415",  # absence of a writing credit does not establish the person was not an author
    "5adf72d35542992d7e9f9369",  # a director/producer biography does not establish the person was not a TV writer
    "5abfa2255542993fe9a41e18",  # the biography does not clearly resolve the vague category 'philosophical writer'
    "5ac311925542995ef918c0de",  # a director/actor biography does not establish the person was not a photographer
    "5ac1a4ed5542994d76dcce90",  # a philosopher/psychologist biography does not establish the person was not a screenwriter
    "5a8230af554299676cceb227",  # one positive fact reports awards won, not that Cocozza was nominated
    "5a851b0d5542997175ce1f7f",  # replacement still leaves the dog type inferable but not explicitly established
    "5a886098554299206df2b248",  # negative says David-Néel wrote 30 books, so it still supports 'writer'
    "5a79c8945542994bb945709e",  # positive evidence does not link Kurosawa to directing Madadayo
    "5add9bfa5542990dbb2f7e85",  # negative still identifies Singha as a beer, partially answering the broad question
    "5ade6acf554299728e26c71a",  # 'more creative titles' is not defined by the evidence or question
    "5a8671e95542994775f60763",  # the negative passage mentions Marvel Comics in the referenced work title/context
    "5add98985542990dbb2f7e82",  # the negative says Harman published the journal, supporting 'publisher'
    "5ae074aa5542993d6555eb54",  # 'artist' is too broad to infer a clean negative from 'romance novelist'
    "5abbc4f955429931dba144ff",  # evidence does not establish the chronology asked about song writing vs. firing
    "5ae0d5335542990adbacf674",  # supporting facts do not clearly link the character to the named show
    "5a7af8fa55429927d897bf25",  # supporting facts do not establish that this is the band's third album reissue
    "5a84b7db5542991dd0999db0",  # evidence does not establish that Götterdämmerung is not an opera
    "5a79c2955542994f819ef097",  # negative evidence states Gandolfini ultimately got the role asked for
    "5a73987c55429978a71e9039",  # first film is described only as a Disney film, not linked to the named company
    "5a78f89055429974737f792b",  # the book title itself exposes the likely answer despite missing support
    "5a8f155e554299458435d54c",  # negative tenure passages support only approximate dates, not gold day precision
    "5a8481ba5542997175ce1ed4",  # evidence does not connect Den Tagayasu's hometown with the named Tokyo raid
    "5a8fab8c5542995b4424208a",  # answer surname leaks through the retained document title
    "5ae056c755429945ae959319",  # positive passages do not establish that Garner acted in The Notebook
    "5ae395d85542992f92d82313",  # question asks for a city but the answer/evidence name a region
    "5ae277575542996483e649b2",  # award-winning film is not linked to the named production company in evidence
    "5ae5df84554299546bf82fa8",  # supporting evidence admits an alternate answer to the constructed question
    "5ac26eed55429951e9e685bf",  # answer alias appears in the retained document title
    "5aba94e355429901930fa845",  # passages do not identify the song as part of the historical musical in the question
    "5ac560235542993e66e82396",  # Sarah Price is called a filmmaker but not specifically a director in the evidence
    "5ac0dcf25542996f0d89cc2c",  # negative evidence exposes only the answer's surname
    "5ae54e005542993aec5ec19a",  # music-composer evidence does not establish a negative professor label
}


def stable_id(prefix, value):
    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def hotpot_pairs(path, question_count=None):
    raw = json.loads(path.read_text(encoding="utf-8"))
    eligible = []
    for item in raw:
        facts = list(dict.fromkeys((str(x[0]), int(x[1])) for x in item.get("supporting_facts", [])))
        if len(facts) < 2:
            continue
        contexts = {title: sentences for title, sentences in item.get("context", [])}
        resolved = []
        valid = True
        for title, sentence_id in facts:
            sentences = contexts.get(title)
            if sentences is None or sentence_id < 0 or sentence_id >= len(sentences):
                valid = False
                break
            resolved.append((title, sentence_id, sentences[sentence_id].strip()))
        if not valid or any(not sentence for _, _, sentence in resolved) or str(item.get("_id")) in MANUAL_EXCLUSIONS:
            continue

        answer = str(item.get("answer", "")).strip()
        answer_norm = normalize(answer)
        if not answer_norm:
            continue
        remove_options = []
        for removed_i in range(len(resolved)):
            retained = [resolved[i] for i in range(2) if i != removed_i]
            retained_text = " ".join(f"{x[0]} {x[2]}" for x in retained)
            # For yes/no questions, the supervision comes from the two required annotated facts.
            answer_absent = answer_norm in {"yes", "no"} or answer_norm not in normalize(retained_text)
            answer_tokens = set(answer_norm.split())
            retained_tokens = set(normalize(retained_text).split())
            if len(answer_tokens) > 1 and len(answer_tokens & retained_tokens) >= 2:
                answer_absent = False
            if answer_absent:
                remove_options.append((removed_i, retained))
        if not remove_options:
            continue

        removed_i, retained = remove_options[0]
        removed_title, removed_sid, removed_sentence = resolved[removed_i]
        support_keys = {(t, sid) for t, sid, _ in resolved}
        candidate_distractors = []
        for title, sentences in item.get("context", []):
            for sid, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence or (title, sid) in support_keys:
                    continue
                if answer_norm not in {"yes", "no"} and answer_norm in normalize(sentence):
                    continue
                same_article = title == removed_title
                candidate_distractors.append((not same_article, abs(len(sentence) - len(removed_sentence)), title, sid, sentence))
        if not candidate_distractors:
            continue
        candidate_distractors.sort()
        _, _, distractor_title, distractor_sid, distractor_sentence = candidate_distractors[0]
        positive_passages = [f"[{t}] {s}" for t, _, s in resolved]
        negative_passages = [f"[{t}] {s}" for t, _, s in retained]
        negative_passages.append(f"[{distractor_title}] {distractor_sentence}")
        # Reject contrasts that become identical after the checkpoint's input
        # serialization cap; otherwise truncation would erase the intervention.
        q = item["question"].strip()
        pos_visible = f"Sub-question: {q}\n\nPassages:\n" + "\n".join(positive_passages)[:1500]
        neg_visible = f"Sub-question: {q}\n\nPassages:\n" + "\n".join(negative_passages)[:1500]
        if pos_visible == neg_visible:
            continue
        negative_text = normalize(" ".join(negative_passages))
        answer_tokens = set(answer_norm.split())
        if answer_norm not in {"yes", "no"} and answer_norm in negative_text:
            continue
        if len(answer_tokens) > 1 and len(answer_tokens & set(negative_text.split())) >= 2:
            continue
        group_id = str(item["_id"])
        common = {
            "question": item["question"].strip(),
            "label_source": "HotpotQA dev distractor supporting_facts annotations; deterministic evidence ablation, not a direct mapping from answer-string or retrieval-relevance labels",
            "answer": answer,
            "paired_group": group_id,
        }
        eligible.append((group_id, common, positive_passages, negative_passages, removed_title, removed_sid, distractor_title, distractor_sid))

    if question_count is not None and len(eligible) < question_count:
        raise RuntimeError(f"Only {len(eligible)} eligible HotpotQA questions; cannot build {question_count} matched pairs.")
    selected = eligible if question_count is None else random.Random(SEED).sample(eligible, question_count)
    examples = []
    for group_id, common, positive, negative, removed_title, removed_sid, distractor_title, distractor_sid in selected:
        facts = next(x.get("supporting_facts", []) for x in raw if str(x.get("_id")) == group_id)
        answer = common["answer"]
        support_ids = [str(title) for title, _ in facts]
        examples.append({
            "id": f"hotpotqa-{group_id}-positive", "dataset": "HotpotQA", "question": common["question"],
            "passages": positive, "label": 1, "label_source": common["label_source"],
            "construction_method": f"All {len(positive)} official sentence-level supporting facts supplied; full annotated support set.",
            "original_document_ids": support_ids, "answer": answer, "paired_group": group_id,
        })
        examples.append({
            "id": f"hotpotqa-{group_id}-negative", "dataset": "HotpotQA", "question": common["question"],
            "passages": negative, "label": 0, "label_source": common["label_source"],
            "construction_method": f"One annotated supporting fact removed (document {removed_title}, sentence {removed_sid}); replaced with a length-matched non-support sentence from the same example's context (document {distractor_title}, sentence {distractor_sid}).",
            "original_document_ids": [str(x.split("]", 1)[0].lstrip("[")) for x in negative],
            "answer": answer, "paired_group": group_id, "removed_support_fact": [removed_title, removed_sid],
            "replacement_distractor": [distractor_title, distractor_sid],
        })
    return examples, len(eligible), len(raw)


def musique_sample(train_dev_csv, sample_per_class=150):
    import pandas as pd
    frame = pd.read_csv(train_dev_csv, dtype={"text": str, "label": int})
    sampled = frame.groupby("label", group_keys=False).sample(n=sample_per_class, random_state=SEED).sort_index()
    result = []
    for row_index, row in sampled.iterrows():
        prefix, passage_blob = row["text"].split("\n\nPassages:\n", 1)
        question = prefix.removeprefix("Sub-question: ").strip()
        result.append({
            "id": stable_id("musique-dev", f"{row_index}\0{row['text']}"), "dataset": "MuSiQue", "question": question,
            "passages": [passage_blob], "label": int(row["label"]),
            "label_source": "Official released MuSiQue dev binary label from the paper repository",
            "construction_method": f"Original MuSiQue dev row; deterministic stratified sample (150 per class), seed={SEED}.",
            "original_document_ids": [], "source_row_index": int(row_index),
        })
    return result


FIQA_PAIRS = [
    ("2568", "139047", "96606", "The passage explicitly says the dealership accepts cashier's checks and gives the buyer time to pay the balance.", "The mortgage-collateral passage does not explain paying cash for a car."),
    ("6867", "443804", "193012", "The passage directly explains that low-liquidity stocks may have no matching buyer or seller at a chosen limit price.", "The moving-average passage discusses technical indicators, not whether a counterparty exists for each trade."),
    ("8271", "415511", "162668", "The passage names dividend-paying blue chips and dividend-focused ETFs as income sources.", "The tax-savings passage discusses liquidity and principal preservation, not income sources in a zero-interest environment."),
    ("6792", "485973", "584295", "The passage states where to obtain deeper bid/ask levels: exchange data feeds or a broker that provides them.", "The broker-versus-market-maker passage does not identify where to view the full order book."),
    ("4047", "417840", "305346", "The passage explains fundamental analysis and how investors use it to estimate stock value.", "The gold-price passage discusses macroeconomic factors, not whether stock research is useful."),
    ("10628", "588116", "537418", "The passage explicitly says the long buyer still owns the shares and illustrates the short seller's obligation.", "The short-selling restrictions passage does not address a failed short or the long buyer's position."),
    ("4999", "314898", "361415", "The passage directly names SEC company filings as a source for public-company financial statements.", "The IRS amended-return anecdote does not provide company financial statements or a retrieval source."),
    ("5255", "75112", "28314", "The passage directly names TSP.Ninja as a charting site for TSP funds.", "The stock-settlement passage does not discuss TSP share-price charts."),
    ("7513", "115372", "105666", "The passage directly names VTI and VEU as a low-cost world-equity fund combination.", "The target-retirement-fund passage does not identify Boglehead world ETFs or index funds."),
    ("10109", "28314", "584295", "The passage explains that settlement takes time for funds and securities to change hands after a sale.", "The broker/market-maker passage discusses execution roles, not the settlement delay after selling."),
    ("10414", "79807", "443804", "The passage gives a concrete high-volume rule of thumb: about two times the 50-day average daily volume.", "The low-liquidity passage mentions small average volume but gives no high/low volume threshold."),
    ("10462", "8266", "105666", "The passage directly assesses starting retirement saving at age 30 and explains why delay is costly.", "The target-date-fund passage discusses fund choice and rebalancing, not whether starting at 30 with no savings is acceptable."),
    ("3357", "209974", "419319", "The passage explains why a client may prefer incorporation: reducing risk of contractor reclassification and associated payroll taxes.", "The wife's-company passage discusses tax-avoidance concerns, not why a client requires incorporation."),
    ("10213", "380942", "23747", "The passage gives concrete short-term savings options and rates, including high-yield savings accounts and CDs.", "The early-retirement IRS passage discusses filing thresholds, not a savings vehicle for seasonal work."),
    ("7700", "273761", "124230", "The passage directly recommends a course of action for the described portfolio reallocation question.", "The ESPP passage gives a single-stock concentration rule, not whether to rebalance this portfolio now or gradually."),
    ("7326", "584295", "485973", "The passage directly distinguishes brokers from market makers and describes their role in finding/completing a trade.", "The order-book passage concerns market-depth data access, not whether brokers execute every exchange trade."),
    ("7467", "193312", "417840", "The passage explicitly identifies NYSE and Nasdaq as secondary markets and distinguishes primary issuance.", "The stock-research passage explains fundamental analysis, not whether stocks trade on secondary markets."),
    ("6803", "193012", "273761", "The passage directly identifies the common Golden Cross averages as 50-day and 200-day.", "The portfolio-reallocation passage discusses selling/rebalancing holdings, not moving-average periods."),
    ("3781", "131959", "23747", "The passage explains AMT depends on income and deductions and includes the cited threshold context.", "The early-retirement passage discusses ordinary filing requirements, not the Alternative Minimum Tax estimate."),
    ("6625", "88892", "482238", "The passage explains that option time value declines with time and connects that sensitivity to theta.", "The delta-hedging passage gives a delta hedge profit formula, not what high theta means."),
    ("5402", "491350", "104988", "The passage gives relevant underwriting conditions for obtaining a home loan with poor credit.", "The financial-advisor mortgage passage discusses advisor/client deals, not poor-credit underwriting after divorce."),
]


def fiqa_sample(base_dir):
    queries = {str(x["_id"]): x["text"] for x in map(json.loads, (base_dir / "queries.jsonl").open(encoding="utf-8"))}
    corpus = {str(x["_id"]): x for x in map(json.loads, (base_dir / "corpus.jsonl").open(encoding="utf-8"))}
    import pandas as pd
    qrels = pd.read_csv(base_dir / "qrels/test.tsv", sep="\t", dtype=str)
    qrel_pairs = set(zip(qrels["query-id"].astype(str), qrels["corpus-id"].astype(str)))
    examples = []
    for qid, pos_id, neg_id, pos_note, neg_note in FIQA_PAIRS:
        if qid not in queries or pos_id not in corpus or neg_id not in corpus:
            raise ValueError(f"FiQA source row missing for curated query {qid}.")
        if (qid, pos_id) not in qrel_pairs:
            raise ValueError(f"Selected positive is not a FiQA test qrel candidate: {qid}/{pos_id}")
        pos_text = f"[FiQA corpus {pos_id}] {corpus[pos_id].get('text','').strip()}"
        neg_text = f"[FiQA corpus {neg_id}] {corpus[neg_id].get('text','').strip()}"
        target_chars = min(1500, len(pos_text), len(neg_text))
        if target_chars < 100:
            raise ValueError(f"Insufficient passage text to create matched pair {qid}.")
        pos_text = pos_text[:target_chars]
        neg_text = neg_text[:target_chars]
        group_id = f"fiqa-{qid}"
        source = "Single-reviewer semantic sufficiency annotation; FiQA qrels are used only to select an on-topic candidate passage and are not mapped to answerability."
        examples.extend([
            {"id": f"{group_id}-positive", "dataset": "FiQA-2018", "question": queries[qid], "passages": [pos_text], "label": 1,
             "label_source": source, "construction_method": "Manually reviewed FiQA question with a candidate passage judged to state an answer; passage length matched to its paired negative.",
             "original_document_ids": [pos_id], "paired_group": group_id, "manual_review_rationale": pos_note, "evidence_chars": target_chars},
            {"id": f"{group_id}-negative", "dataset": "FiQA-2018", "question": queries[qid], "passages": [neg_text], "label": 0,
             "label_source": source, "construction_method": "Same question paired with a manually reviewed same-corpus financial passage that does not provide the requested answer; passage length matched to its positive.",
             "original_document_ids": [neg_id], "paired_group": group_id, "manual_review_rationale": neg_note, "evidence_chars": target_chars},
        ])
    return examples


def main():
    raw_hotpot = ROOT / "datasets/raw/hotpot_dev_distractor_v1.json"
    official_dev = ROOT / "datasets/training_data/deberta_fg_dev.csv"
    hotpot, eligible_n, source_n = hotpot_pairs(raw_hotpot, question_count=None)
    musique = musique_sample(official_dev)
    fiqa = fiqa_sample(ROOT / "datasets/raw/fiqa/fiqa")
    examples = musique + hotpot + fiqa
    out = ROOT / "datasets/prepared_eval_examples.jsonl"
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in examples) + "\n", encoding="utf-8")

    manifest = json.loads((ROOT / "datasets/dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["evaluation_data"].update({
        "seed": SEED,
        "prepared_examples_file": "prepared_eval_examples.jsonl",
        "sample_counts": {"MuSiQue": len(musique), "HotpotQA": len(hotpot), "FiQA-2018": len(fiqa)},
        "hotpotqa_source": "HotpotQA official dev distractor data (retrieved from the Kaggle mirror jeromeblanchet/hotpotqa-question-answering-dataset); official task documentation identifies the file and supporting_facts fields.",
        "hotpotqa_license": "CC BY-SA 4.0 (official HotpotQA site)",
        "hotpotqa_source_questions": source_n,
        "hotpotqa_eligible_questions": eligible_n,
        "hotpotqa_construction": "All eligible questions from the official distractor dev split are paired into answerable and evidence-deficient examples. Positives include the full annotated supporting-fact set (two or more sentences); negatives replace one required support sentence with a nearest-length non-support sentence. Examples with answer leakage, ambiguous annotations, invalid support references, or no visible contrast after the 1,500-character input cap are excluded; labels remain support-fact-derived proxies.",
        "not_evaluated": {
            "SciFact": "SciFact SUPPORT/CONTRADICT/NOT ENOUGH INFO are claim-verification outcomes, not binary answerability labels. No conversion was used.",
        },
        "fiqa_source": "BEIR/MTEB FiQA queries and corpus from the Kaggle mirror hamzaidmoudi/fiqa-2018; test qrels used solely to locate on-topic candidate passages. MTEB metadata says the license is not specified; this curated dataset remains private and is labeled unknown.",
        "fiqa_annotation": "21 paired query/passage comparisons; single reviewer wrote a rationale for each of the 42 binary judgments. No independent second annotator or inter-annotator agreement is available. Paired evidence excerpts have equal character length to reduce length cues.",
    })
    manifest["evaluation_data"]["not_evaluated"] = {
        "SciFact": "SciFact SUPPORT/CONTRADICT/NOT ENOUGH INFO are claim-verification outcomes, not binary answerability labels. No conversion was used."
    }
    (ROOT / "datasets/dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for e in examples:
        counts.setdefault(e["dataset"], {0: 0, 1: 0})[e["label"]] += 1
    print(json.dumps({"counts": counts, "hotpotqa_source_questions": source_n,
                      "hotpotqa_eligible_questions": eligible_n, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
