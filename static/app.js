import { LetterGate } from "./letter-gate.js";

const byId = (id) => document.getElementById(id);
const cameraButton = byId("camera");
const startButton = byId("start");
const demoButton = byId("demo");
const openVideoButton = byId("open-video");
const videoFile = byId("video-file");
const device = byId("device");
const video = byId("video");
const viewer = byId("viewer");
const placeholder = byId("placeholder");
const overlay = byId("overlay");
const ink = overlay.getContext("2d");
const capture = document.createElement("canvas");
const captureInk = capture.getContext("2d", { alpha: false });
const transcript = byId("transcript");
const auto = byId("auto");
const insert = byId("insert");
const speak = byId("speak");
const tiles = [...document.querySelectorAll(".tile")];
const guide = byId("guide");
const guideVideo = byId("guide-video");
const gate = new LetterGate(Number(document.body.dataset.threshold));
const fingers = [
  [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 10, 11, 12],
  [9, 13, 14, 15, 16], [13, 17, 18, 19, 20], [17, 0],
];

let active = null;
let videoUrl = null;
let voice = null;
let requestId = 0;
let lastRateCount = 0;
let lastRateTime = 0;

function status(text, error = false) {
  const element = byId("status");
  if (element.textContent !== text) element.textContent = text;
  element.classList.toggle("error", error);
}

function voiceStatus(text, error = false) {
  const element = byId("speech-status");
  element.textContent = text;
  element.classList.toggle("error", error);
}

function syncText() {
  const hasText = transcript.value.length > 0;
  byId("clear").disabled = !hasText;
  byId("delete").disabled = !hasText;
  byId("copy").disabled = !hasText;
  byId("space").disabled = transcript.value.length >= transcript.maxLength;
  insert.disabled = !gate.readyLetter || transcript.value.length >= transcript.maxLength;
  speak.disabled = !voice && !transcript.value.trim();
}

function append(text, atCaret = false) {
  const start = atCaret ? transcript.selectionStart : transcript.value.length;
  const end = atCaret ? transcript.selectionEnd : start;
  if (transcript.value.length + text.length - (end - start) > transcript.maxLength) return;
  transcript.setRangeText(text, start, end, "end");
  transcript.scrollTop = transcript.scrollHeight;
  syncText();
  if (text.trim()) byId("announcer").textContent = `Добавлена буква ${text}. ${transcript.value}`;
}

function draw(points, width, height) {
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
  ink.clearRect(0, 0, width, height);
  if (!points.length) return;
  ink.lineWidth = Math.max(2, width / 240);
  ink.lineJoin = "round";
  ink.lineCap = "round";
  ink.strokeStyle = "#6db1ff";
  for (const finger of fingers) {
    ink.beginPath();
    for (const [index, joint] of finger.entries()) {
      const [x, y] = points[joint];
      if (index === 0) ink.moveTo(x * width, y * height);
      else ink.lineTo(x * width, y * height);
    }
    ink.stroke();
  }
  ink.fillStyle = "#f2f7ff";
  for (const [x, y] of points) {
    ink.beginPath();
    ink.arc(x * width, y * height, Math.max(2.2, width / 220), 0, Math.PI * 2);
    ink.fill();
  }
}

function showResult(letter = "", score = 0, progress = 0, message = "Ожидание жеста") {
  byId("letter").textContent = letter || "—";
  byId("score").textContent = letter ? `${Math.round(score * 100)}%` : "";
  byId("hold").value = progress;
  byId("reading").textContent = message;
  insert.disabled = !gate.readyLetter || transcript.value.length >= transcript.maxLength;
  for (const tile of tiles) tile.classList.toggle("active", tile.dataset.letter === letter);
}

function endSource(message = "Камера выключена", error = false) {
  requestId += 1;
  if (active) {
    active.controller.abort();
    clearTimeout(active.timer);
  }
  active = null;
  video.pause();
  video.srcObject?.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
  video.removeAttribute("src");
  video.load();
  if (videoUrl) {
    URL.revokeObjectURL(videoUrl);
    videoUrl = null;
  }
  viewer.hidden = true;
  placeholder.hidden = false;
  byId("live").hidden = true;
  cameraButton.hidden = true;
  device.hidden = true;
  startButton.disabled = false;
  byId("fps").textContent = "";
  gate.reset();
  ink.clearRect(0, 0, overlay.width, overlay.height);
  showResult();
  status(message, error);
}

async function listCameras(id, selected) {
  try {
    const cameras = (await navigator.mediaDevices.enumerateDevices()).filter((source) => source.kind === "videoinput");
    if (requestId !== id) return;
    device.replaceChildren(...cameras.map((camera, index) =>
      new Option(camera.label || `Камера ${index + 1}`, camera.deviceId, false, camera.deviceId === selected)));
    device.hidden = cameras.length < 2;
  } catch {
    if (requestId === id) device.hidden = true;
  }
}

async function startSource(kind, file = null) {
  endSource();
  const id = requestId;
  startButton.disabled = true;
  status({
    camera: "Подключаем камеру…",
    demo: "Загружаем демо…",
    file: "Открываем видео…",
  }[kind]);
  try {
    if (kind === "camera") {
      if (!navigator.mediaDevices?.getUserMedia) {
        endSource("Браузер открывает камеру на localhost или по доверенному HTTPS.", true);
        return;
      }
      const camera = device.value ? { deviceId: { exact: device.value } } : { facingMode: "user" };
      const settings = { width: { ideal: 960 }, height: { ideal: 720 }, ...camera };
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: settings });
      } catch (error) {
        if (error.name !== "OverconstrainedError" || !device.value || requestId !== id) throw error;
        stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: "user" } });
      }
      if (requestId !== id) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      video.srcObject = stream;
      for (const track of stream.getVideoTracks()) {
        track.addEventListener("ended", () => {
          if (requestId === id) endSource("Камера отключилась.", true);
        });
      }
    } else if (file) {
      videoUrl = URL.createObjectURL(file);
      video.src = videoUrl;
    } else {
      video.src = "/static/demo.mp4";
    }
    await video.play();
    if (requestId !== id) return;
    viewer.classList.toggle("mirror", kind === "camera");
    viewer.hidden = false;
    placeholder.hidden = true;
    byId("live").hidden = false;
    byId("source-name").textContent = {
      camera: "Камера включена",
      demo: "Видео из Bukva",
      file: "Видеофайл",
    }[kind];
    cameraButton.hidden = false;
    active = { kind, controller: new AbortController(), timer: 0, lastVideoTime: -1 };
    if (kind === "camera") listCameras(id, video.srcObject.getVideoTracks()[0].getSettings().deviceId);
    lastRateCount = 0;
    lastRateTime = performance.now();
    status("Ищем руку в кадре");
    processFrame(active);
  } catch (error) {
    if (requestId !== id) return;
    const cameraErrors = {
      NotAllowedError: "Доступ к камере закрыт. Разрешите его в браузере и настройках устройства.",
      NotFoundError: "Камера не найдена. Подключите устройство и повторите.",
      NotReadableError: "Камера занята или недоступна. Закройте другие приложения с камерой.",
    };
    const errors = {
      camera: cameraErrors[error.name] || "Не удалось подключить камеру.",
      demo: "Не удалось открыть демовидео.",
      file: "Браузер не смог открыть файл. Попробуйте видео в формате MP4 или WebM.",
    };
    endSource(errors[kind], true);
  }
}

function applyFrame(frame) {
  const hasRoom = transcript.value.length < transcript.maxLength;
  const state = gate.update(frame, performance.now(), auto.checked && hasRoom);
  if (state.insert) append(state.insert);
  const message = {
    empty: "Ожидание жеста",
    unknown: "Жест не распознан",
    low: "Неуверенный результат",
    holding: "Удерживайте жест",
    ready: hasRoom ? "Готово к вставке" : "Текст заполнен",
    fixed: "Буква добавлена",
  }[state.mode];
  showResult(frame.letter, frame.score, state.progress, message);
  status(frame.points.length ? "Рука в кадре" : "Ищем руку в кадре");
}

async function processFrame(source) {
  if (active !== source) return;
  const cycleStarted = performance.now();
  try {
    if (video.readyState >= 2 && video.videoWidth && video.currentTime !== source.lastVideoTime) {
      source.lastVideoTime = video.currentTime;
      const factor = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
      const width = Math.round(video.videoWidth * factor);
      const height = Math.round(video.videoHeight * factor);
      if (capture.width !== width || capture.height !== height) {
        capture.width = width;
        capture.height = height;
      }
      captureInk.drawImage(video, 0, 0, width, height);
      const blob = await new Promise((resolve) => capture.toBlob(resolve, "image/jpeg", 0.82));
      if (active !== source) return;
      if (!blob) throw new Error("Cannot encode frame");
      const response = await fetch("/api/frame", {
        method: "POST", body: blob, signal: source.controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const frame = await response.json();
      if (active !== source) return;
      draw(frame.points, width, height);
      applyFrame(frame);
      lastRateCount += 1;
      if (performance.now() - lastRateTime >= 1000) {
        byId("fps").textContent = `${(lastRateCount * 1000 / (performance.now() - lastRateTime)).toFixed(0)} кадр/с`;
        lastRateCount = 0;
        lastRateTime = performance.now();
      }
    }
  } catch {
    if (active === source) endSource("Не удалось обработать кадр. Запустите источник ещё раз.", true);
    return;
  }
  if (active === source) {
    source.timer = setTimeout(() => processFrame(source), Math.max(4, 95 - (performance.now() - cycleStarted)));
  }
}

function stopVoice() {
  if (voice) {
    voice.controller.abort();
    voice.audio.pause();
    if (voice.url) URL.revokeObjectURL(voice.url);
  }
  voice = null;
  byId("speak-label").textContent = "Озвучить текст";
  syncText();
}

async function playVoice() {
  if (voice) {
    stopVoice();
    voiceStatus("");
    return;
  }
  const text = transcript.value.trim();
  if (!text) return;
  const run = { controller: new AbortController(), audio: new Audio(), url: null };
  voice = run;
  byId("speak-label").textContent = "Остановить озвучку";
  voiceStatus("Готовим озвучку…");
  try {
    const response = await fetch("/api/speech", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: run.controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const audio = await response.blob();
    if (voice !== run) return;
    run.url = URL.createObjectURL(audio);
    run.audio.src = run.url;
    run.audio.addEventListener("ended", () => {
      if (voice === run) {
        stopVoice();
        voiceStatus("Озвучено");
      }
    });
    run.audio.addEventListener("error", () => {
      if (voice === run) {
        stopVoice();
        voiceStatus("Не удалось воспроизвести звук.", true);
      }
    });
    await run.audio.play();
    if (voice === run) voiceStatus("Воспроизводим…");
  } catch {
    if (voice === run) {
      stopVoice();
      voiceStatus("Не удалось озвучить текст. Проверьте звук браузера и повторите.", true);
    }
  }
}

startButton.addEventListener("click", () => startSource("camera"));
demoButton.addEventListener("click", () => startSource("demo"));
openVideoButton.addEventListener("click", () => videoFile.click());
videoFile.addEventListener("change", () => {
  const file = videoFile.files[0];
  videoFile.value = "";
  if (file) startSource("file", file);
});
cameraButton.addEventListener("click", () => endSource());
device.addEventListener("change", () => startSource("camera"));
video.addEventListener("ended", () => {
  if (active?.kind === "demo") endSource("Демо завершено");
  else if (active?.kind === "file") endSource("Видео обработано");
});
video.addEventListener("error", () => {
  if (active?.kind === "file") endSource("Видео не удалось прочитать до конца.", true);
});
auto.addEventListener("change", () => {
  insert.hidden = auto.checked;
  gate.restartCandidate();
  showResult();
});
insert.addEventListener("click", () => {
  const letter = gate.confirm();
  if (!letter) return;
  append(letter);
  showResult(letter, gate.lastScore, 1, "Буква добавлена");
});
transcript.addEventListener("input", syncText);
byId("space").addEventListener("click", () => append(" ", true));
byId("delete").addEventListener("click", () => {
  const end = transcript.selectionEnd;
  let start = transcript.selectionStart;
  if (start === end && start > 0) start -= [...transcript.value.slice(0, start)].at(-1).length;
  transcript.setRangeText("", start, end, "end");
  syncText();
});
byId("clear").addEventListener("click", () => {
  transcript.value = "";
  voiceStatus("");
  syncText();
});
byId("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(transcript.value);
    voiceStatus("Текст скопирован");
  } catch {
    transcript.focus();
    transcript.select();
    voiceStatus("Текст выделен для копирования.");
  }
});
speak.addEventListener("click", playVoice);
for (const tile of tiles) {
  tile.addEventListener("click", () => {
    const letter = tile.dataset.letter;
    byId("guide-title").textContent = `Буква ${letter}`;
    guideVideo.src = `/static/signs/${encodeURIComponent(letter)}.mp4`;
    guide.showModal();
    guideVideo.play().catch(() => {});
  });
}
guide.addEventListener("close", () => {
  guideVideo.pause();
  guideVideo.removeAttribute("src");
  guideVideo.load();
});
window.addEventListener("pagehide", () => {
  endSource();
  stopVoice();
});
syncText();
