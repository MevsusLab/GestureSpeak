import assert from "node:assert/strict";
import test from "node:test";
import { LetterGate } from "../static/letter-gate.js";

const hand = (letter, score = 0.98) => ({ points: [[0, 0]], letter, score });
const empty = { points: [], letter: "", score: 0 };

function hold(gate, frame, start, auto = true) {
  const states = [];
  for (let offset = 0; offset <= 800; offset += 100) {
    states.push(gate.update(frame, start + offset, auto));
  }
  return states;
}

test("one insertion per held letter then re-entry after releasing", () => {
  const gate = new LetterGate(0.65);
  const states = hold(gate, hand("М"), 0);
  assert.equal(states[4].mode, "holding");
  assert.equal(states[5].insert, "М");
  assert.deepEqual(states.flatMap((state) => state.insert || []), ["М"]);
  hold(gate, hand(""), 900);
  assert.ok(hold(gate, hand("М"), 1800).every((state) => !state.insert));
  gate.update(empty, 2750, true);
  gate.update(empty, 2930, true);
  assert.equal(hold(gate, hand("М"), 3000)[5].insert, "М");
});

test("recognizes a different letter with the hand still in the frame", () => {
  const gate = new LetterGate(0.65);
  hold(gate, hand("М"), 0);
  assert.equal(hold(gate, hand("И"), 900)[5].insert, "И");
});

test("uncertain frames and missing samples do not accumulate a hold", () => {
  const gate = new LetterGate(0.65);
  gate.update(hand("А"), 0, true);
  gate.update(hand("А"), 100, true);
  assert.equal(gate.update(hand("А"), 800, true).progress, 0);
  assert.ok(hold(gate, hand("А", 0.55), 900).every((state) => state.mode === "low"));
});

test("manual input requires confirmation and subsequent stable gesture", () => {
  const gate = new LetterGate(0.65);
  assert.equal(hold(gate, hand("Р"), 0, false).at(-1).mode, "ready");
  assert.equal(gate.confirm(), "Р");
  assert.equal(gate.confirm(), "");
  assert.equal(hold(gate, hand("Р"), 900, false).at(-1).mode, "fixed");
});
