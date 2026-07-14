// UI state machine: model registry/picker, install-all flow, config popover,
// validation toasts (bottom-left), download progress (bottom-right), and the
// token-streaming story renderer.

import { EOS_ID, loadTokenizer } from "./tokenizer.js";
import { fetchModel, isCached, loadManifest, parsePack } from "./pack.js";
import { Model, greedy, mulberry32, sampleToken } from "./model.js";
import { fixMojibake, mojibakeFilter } from "./display.js";
import {
  RegExpMatcher,
  englishDataset,
  englishRecommendedTransformers,
} from "../vendor/obscenity.mjs";

const $ = (id) => document.getElementById(id);
const el = {
  installBtn: $("install-btn"),
  hero: $("hero"),
  heroBytes: $("hero-bytes"),
  transcript: $("transcript"),
  main: $("main"),
  composer: $("composer"),
  input: $("input"),
  sendBtn: $("send-btn"),
  modelBtn: $("model-btn"),
  modelBtnLabel: $("model-btn-label"),
  modelMenu: $("model-menu"),
  configBtn: $("config-btn"),
  configMenu: $("config-menu"),
  toastRegion: $("toast-region"),
  progressCard: $("progress-card"),
  progressLabel: $("progress-label"),
  progressFill: $("progress-fill"),
  progressOverall: $("progress-overall"),
};

const state = {
  entries: [], // manifest entries + { available }
  currentId: null,
  model: null,
  tokenizer: null,
  matcher: new RegExpMatcher({
    ...englishDataset.build(),
    ...englishRecommendedTransformers,
  }),
  generating: false,
  installing: false,
  stopRequested: false,
  config: { maxTokens: 150, temperature: 0.8, topK: 40, seed: 0 },
};

const fmtMB = (b) => `${(b / 1e6).toFixed(1)} MB`;
const tick = () => new Promise((r) => setTimeout(r, 0));

// ---- toasts (bottom-left) --------------------------------------------------

function toast(message) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = message;
  el.toastRegion.append(t);
  setTimeout(() => t.remove(), 6000);
}

// ---- model registry / picker ------------------------------------------------

function currentEntry() {
  return state.entries.find((e) => e.id === state.currentId);
}

function renderModelMenu() {
  el.modelMenu.replaceChildren();
  for (const e of state.entries) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "menu-item" + (e.id === state.currentId ? " selected" : "");
    item.disabled = !e.available;
    const label = document.createElement("span");
    label.textContent = e.label;
    const size = document.createElement("span");
    size.className = "size";
    size.textContent = fmtMB(e.bytes);
    const row = document.createElement("span");
    row.className = "row";
    row.append(label, size);
    item.append(row);
    if (e.judge != null && e.ppl != null) {
      const stats = document.createElement("span");
      stats.className = "stats";
      stats.textContent = `coherence ${e.judge.toFixed(2)} / 5 · val ppl ${e.ppl.toFixed(2)}`;
      item.append(stats);
    }
    item.addEventListener("click", () => {
      closeMenus();
      if (e.id !== state.currentId) switchModel(e.id);
    });
    if (e.available) {
      el.modelMenu.append(item);
    } else {
      // A disabled button swallows hover in some browsers, so the tooltip lives
      // on an enabled wrapper around it.
      const row = document.createElement("span");
      row.className = "menu-row locked";
      const tip = document.createElement("span");
      tip.className = "menu-tip";
      tip.textContent = "Not installed yet — press the install button (top left) to download it.";
      row.append(item, tip);
      el.modelMenu.append(row);
    }
  }
}

function updateChrome() {
  const e = currentEntry();
  el.modelBtnLabel.textContent = e
    ? `${e.label} · ${fmtMB(e.bytes)}`
    : "loading…";
  if (e) el.heroBytes.textContent = fmtMB(e.bytes);
  const allInstalled = state.entries.every((x) => x.available);
  el.installBtn.disabled = state.installing || allInstalled;
  el.installBtn.classList.toggle("done", allInstalled);
  if (allInstalled) el.installBtn.title = "All models installed";
  el.sendBtn.disabled = !state.model && !state.generating;
  renderModelMenu();
}

async function switchModel(id) {
  const entry = state.entries.find((e) => e.id === id);
  if (!entry || state.generating) return;
  el.modelBtnLabel.textContent = `loading ${entry.label}…`;
  el.sendBtn.disabled = true;
  try {
    const buffer = await fetchModel(entry.url);
    state.model = new Model(parsePack(buffer));
    state.currentId = id;
  } catch (err) {
    toast(`Couldn't load ${entry.label} — ${err.message}`);
  }
  updateChrome();
}

// ---- install-all flow (progress bottom-right) --------------------------------

async function installAll() {
  if (state.installing) return;
  const missing = state.entries.filter((e) => !e.available);
  if (!missing.length) return;
  state.installing = true;
  updateChrome();
  el.progressCard.hidden = false;

  let done = 0;
  for (const entry of missing) {
    el.progressLabel.textContent = `${entry.label} — ${fmtMB(entry.bytes)}`;
    el.progressOverall.textContent = `${done + 1} of ${missing.length}`;
    el.progressFill.style.width = "0";
    try {
      await fetchModel(entry.url, (loaded, total) => {
        const pct = Math.min(100, (loaded / (total ?? entry.bytes)) * 100);
        el.progressFill.style.width = `${pct}%`;
      });
      entry.available = true;
      renderModelMenu();
    } catch {
      toast(`${entry.label} isn't available yet.`);
    }
    done++;
  }

  el.progressCard.hidden = true;
  state.installing = false;
  updateChrome();
}

// ---- config popover -----------------------------------------------------------

const cfgInputs = {
  maxTokens: { el: $("cfg-max-tokens"), min: 1, max: 400, int: true },
  temperature: { el: $("cfg-temperature"), min: 0, max: 2, int: false },
  topK: { el: $("cfg-top-k"), min: 0, max: 4096, int: true },
  seed: { el: $("cfg-seed"), min: 0, max: 2 ** 31 - 1, int: true },
};

for (const [key, c] of Object.entries(cfgInputs)) {
  c.el.addEventListener("change", () => {
    let v = Number(c.el.value);
    if (!Number.isFinite(v)) v = state.config[key];
    v = Math.min(c.max, Math.max(c.min, c.int ? Math.round(v) : v));
    c.el.value = v;
    state.config[key] = v;
  });
}

// ---- menus ----------------------------------------------------------------------

function closeMenus() {
  for (const [btn, menu] of [
    [el.modelBtn, el.modelMenu],
    [el.configBtn, el.configMenu],
  ]) {
    menu.hidden = true;
    btn.setAttribute("aria-expanded", "false");
  }
}

function toggleMenu(btn, menu) {
  const open = menu.hidden;
  closeMenus();
  if (open) {
    menu.hidden = false;
    btn.setAttribute("aria-expanded", "true");
  }
}

el.modelBtn.addEventListener("click", () => toggleMenu(el.modelBtn, el.modelMenu));
el.configBtn.addEventListener("click", () => toggleMenu(el.configBtn, el.configMenu));
document.addEventListener("click", (ev) => {
  if (!ev.target.closest(".menu-anchor")) closeMenus();
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") closeMenus();
});

// ---- transcript rendering --------------------------------------------------------

function addUserLine(text) {
  const label = document.createElement("p");
  label.className = "entry-user";
  label.textContent = "You began";
  el.transcript.append(label);

  const story = document.createElement("p");
  story.className = "story streaming";
  const first = text.match(/\S/);
  if (first) {
    const cap = document.createElement("span");
    cap.className = "dropcap";
    cap.textContent = text.slice(0, first.index + 1).trimStart();
    story.append(cap, document.createTextNode(text.slice(first.index + 1)));
  } else {
    story.append(document.createTextNode(text));
  }
  el.transcript.append(story);
  return story;
}

function appendText(story, text) {
  if (!text) return;
  story.lastChild.nodeType === Node.TEXT_NODE
    ? (story.lastChild.data += text)
    : story.append(document.createTextNode(text));
  el.main.scrollTop = el.main.scrollHeight;
}

function finishStory(story, note) {
  story.classList.remove("streaming");
  if (note) {
    const n = document.createElement("p");
    n.className = "story-note";
    n.textContent = note;
    story.after(n);
  }
  const fin = document.createElement("p");
  fin.className = "fin";
  fin.textContent = "❦";
  el.transcript.append(fin);
}

// ---- generation -------------------------------------------------------------------

function setGenerating(on) {
  state.generating = on;
  document.body.classList.toggle("generating", on);
  el.modelBtn.disabled = on;
  el.configBtn.disabled = on;
  el.input.disabled = on;
  el.sendBtn.setAttribute("aria-label", on ? "Stop writing" : "Continue the story");
  if (on) closeMenus();
}

// Attempts to phish keys/secrets out of the storyteller, or a pasted key itself.
const KEY_HUNT_RE =
  /\b(?:api|secret|access|private|auth(?:orization)?)[\s_-]*(?:key|token|credential)s?\b|\bsk-[A-Za-z0-9]{8,}\b/i;

function validate(text, ids) {
  if (!text.trim()) return "Write an opening line first.";
  if (KEY_HUNT_RE.test(text))
    return "Bar'd has no secrets here to tell :D";
  if (state.matcher.hasMatch(text))
    return "Guys, please don't swear.";
  const ctx = state.model.cfg.ctx;
  if (ids.length + state.config.maxTokens > ctx)
    return `That opening is too long at ${ids.length} tokens of opening in addition to the ${state.config.maxTokens} max tokens exceeding the model's ${ctx}-token memory.`;
  return null;
}

async function runStory(text) {
  const ids = state.tokenizer.encode(text);
  const problem = validate(text, ids);
  if (problem) {
    toast(problem);
    return;
  }

  document.body.classList.add("storytime");
  setGenerating(true);
  state.stopRequested = false;
  const story = addUserLine(fixMojibake(text));
  const model = state.model;
  const { maxTokens, temperature, topK, seed } = state.config;

  try {
    model.reset();
    let logits = null;
    for (let i = 0; i < ids.length; i++) {
      logits = model.step(ids[i]);
      if (i % 16 === 15) await tick();
    }

    const rng = mulberry32(seed);
    const decoder = state.tokenizer.streamDecoder();
    const display = mojibakeFilter();
    let note = null;
    for (let n = 0; n < maxTokens; n++) {
      if (state.stopRequested) break;
      const id = sampleToken(logits, { temperature, topK }, rng);
      if (id === EOS_ID) break;
      appendText(story, display.push(decoder.push(id)));
      if (model.pos >= model.cfg.ctx) {
        note = "The model reached its context limit and had to stop.";
        break;
      }
      logits = model.step(id);
      await tick();
    }
    appendText(story, display.push(decoder.flush()));
    appendText(story, display.flush());
    finishStory(story, note);
  } catch (err) {
    finishStory(story, `Something went wrong while writing: ${err.message}`);
  }
  setGenerating(false);
}

// ---- composer events -----------------------------------------------------------

el.composer.addEventListener("submit", (ev) => {
  ev.preventDefault();
  if (state.generating) {
    state.stopRequested = true;
    return;
  }
  if (!state.model) return;
  const text = el.input.value;
  el.input.value = "";
  el.input.style.height = "auto";
  runStory(text);
});

el.input.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    el.composer.requestSubmit();
  }
});

el.input.addEventListener("input", () => {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, 144)}px`;
});

el.installBtn.addEventListener("click", installAll);

// ---- startup -------------------------------------------------------------------

async function main() {
  try {
    const [manifest, tokenizer] = await Promise.all([
      loadManifest(),
      loadTokenizer(),
    ]);
    state.tokenizer = tokenizer;
    state.entries = await Promise.all(
      manifest.models.map(async (e) => ({
        ...e,
        available: e.baked || (await isCached(e.url)),
      }))
    );

    // Default: smallest gate-passing model that's available; fallback = baked.
    const candidates = state.entries
      .filter((e) => e.available && e.gate === true)
      .sort((a, b) => a.bytes - b.bytes);
    const first = candidates[0] ?? state.entries.find((e) => e.baked);
    updateChrome();
    await switchModel(first.id);
  } catch (err) {
    toast(`The storyteller couldn't start: ${err.message}`);
  }
}

// Dev/parity hook (docs/demo_site.md): greedy token ids for a prompt.
window.__greedy = (prompt, n) =>
  greedy(state.model, state.tokenizer.encode(prompt), n);

main();
