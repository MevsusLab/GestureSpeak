const HOLD_MS = 500;
const RELEASE_MS = 290;
const MAX_GAP_MS = 400;

export class LetterGate {
  constructor(threshold) {
    this.threshold = threshold;
    this.reset();
  }

  reset() {
    this.lastInserted = "";
    this.lastHandAt = null;
    this.lastFrameAt = null;
    this.restartCandidate();
  }

  restartCandidate() {
    this.candidate = "";
    this.candidateAt = 0;
    this.consecutive = 0;
    this.readyLetter = "";
    this.lastScore = 0;
  }

  confirm() {
    const letter = this.readyLetter;
    if (letter) {
      this.lastInserted = letter;
      this.readyLetter = "";
    }
    return letter;
  }

  update({ points, letter, score }, now, auto) {
    if (this.lastFrameAt !== null && now - this.lastFrameAt > MAX_GAP_MS) this.restartCandidate();
    this.lastFrameAt = now;
    if (!points.length) {
      if (this.lastHandAt !== null && now - this.lastHandAt >= RELEASE_MS) this.lastInserted = "";
      this.restartCandidate();
      return { mode: "empty", progress: 0 };
    }
    this.lastHandAt = now;
    if (!letter || score < this.threshold) {
      this.restartCandidate();
      return { mode: letter ? "low" : "unknown", progress: 0 };
    }
    if (letter === this.lastInserted) {
      this.restartCandidate();
      return { mode: "fixed", progress: 1 };
    }
    if (letter !== this.candidate) {
      this.restartCandidate();
      this.candidate = letter;
      this.candidateAt = now;
    }
    this.lastScore = score;
    this.consecutive += 1;
    const progress = Math.min(1, (now - this.candidateAt) / HOLD_MS);
    if (progress === 1 && this.consecutive >= 4) {
      this.readyLetter = letter;
      if (auto) return { mode: "fixed", progress: 1, insert: this.confirm() };
      return { mode: "ready", progress: 1 };
    }
    return { mode: "holding", progress };
  }
}
