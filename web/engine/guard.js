// Name guard, mirroring python/glassbox/guard.py. Two checks:
//   * before a token is chosen: logistic regression on the SAE features of the current position
//     (logit = bias + sum of weight * activation); it predicts whether the next token starts a name
//   * once a word is complete: the word check (in the name list, or capitalized mid-sentence and
//     not a listed non-name)

/** Is the word at text[i] the first word of a sentence? (Same rule as guard.py.) */
export function sentenceStart(text, i) {
  if (i > 0 && (text[i - 1] === '"' || text[i - 1] === "'")) return true;
  let j = i - 1;
  while (j >= 0 && ' \t"\''.includes(text[j])) j--;
  return j < 0 || ".!?\n".includes(text[j]);
}

export class NameGuard {
  constructor(json) {
    if (json.format !== "glassbox-guard") throw new Error("not a glassbox guard file");
    this.layer = json.layer;
    this.bias = json.bias;
    this.weights = Float32Array.from(json.weights);
    this.thresholds = { balanced: json.threshold, strict: json.strict_threshold };
    this.names = new Set(json.names);
    this.exclude = new Set(json.exclude);
    this.labels = new Map(Object.entries(json.labels ?? {}).map(([k, v]) => [Number(k), v]));
  }

  /** Classifier logit from the active SAE features {ids, acts} of one position. */
  logit(codes) {
    let s = this.bias;
    for (let i = 0; i < codes.ids.length; i++) s += this.weights[codes.ids[i]] * codes.acts[i];
    return s;
  }

  /** The features that pushed the logit up most, for explaining a block. */
  explain(codes, n = 3) {
    const parts = Array.from(codes.ids, (f, i) => ({ feature: f, contribution: this.weights[f] * codes.acts[i] }));
    return parts.filter((p) => p.contribution > 0).sort((a, b) => b.contribution - a.contribution).slice(0, n)
      .map((p) => ({ ...p, label: this.labels.get(p.feature) ?? null }));
  }

  /** Word check: a known name, or a capitalized word mid-sentence that is not a listed non-name. */
  wordFlag(word, atSentenceStart) {
    if (!word) return false;
    if (this.names.has(word)) return true;
    return /^[A-Z]/.test(word) && !atSentenceStart && !this.exclude.has(word);
  }
}
