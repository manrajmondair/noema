import { behavior, decode, worldStep } from "./model.mjs";

const $ = (id) => document.getElementById(id);
const toggle = $("toggle");
const reset = $("reset");
const status = $("status");
const pad = $("pad");
const knob = $("knob");
const vector = $("vector");
const readout = $("readout");
const align = $("align");
const raster = $("raster");
const pathCanvas = $("path");
const rasterContext = raster.getContext("2d");
const pathContext = pathCanvas.getContext("2d");

const CAP = 45;
const HISTORY = 90;
const RAMP = [
  [250, 247, 240], [225, 231, 238], [200, 209, 222], [175, 185, 202],
  [147, 159, 181], [118, 130, 157], [87, 99, 129], [57, 67, 99], [29, 36, 62],
];

let model;
let loadPromise;
let running = false;
let lastStep = 0;
let latents = [];
let actions = [];
let action = [0, 0];
let position = [0, 0];
let head = [0, 0];
let alignment;
let dragging = false;
const history = [];
const path = [];

function setStatus(message, state = "") {
  status.textContent = message;
  status.dataset.state = state;
}

function validModel(value) {
  return value
    && Number.isInteger(value.dim)
    && Array.isArray(value.seed?.z)
    && Array.isArray(value.seed?.actions)
    && Array.isArray(value.world?.blocks)
    && Array.isArray(value.readout);
}

async function loadModel() {
  if (model) return model;
  if (!loadPromise) {
    toggle.disabled = true;
    setStatus("Loading 2.4 MB model…");
    loadPromise = fetch(new URL("./model-data.json", import.meta.url))
      .then((response) => {
        if (!response.ok) throw new Error(`model request returned ${response.status}`);
        return response.json();
      })
      .then((value) => {
        if (!validModel(value)) throw new Error("model artifact is incomplete");
        model = value;
        initialise();
        reset.disabled = false;
        setStatus("Ready · model runs locally");
        return model;
      })
      .catch((error) => {
        loadPromise = undefined;
        setStatus(`Could not load model: ${error.message}`, "error");
        throw error;
      })
      .finally(() => {
        toggle.disabled = false;
      });
  }
  return loadPromise;
}

function initialise() {
  latents = model.seed.z.map((row) => row.slice());
  actions = model.seed.actions.map((row) => row.slice());
  position = [0, 0];
  head = [0, 0];
  alignment = undefined;
  history.length = 0;
  path.length = 0;
  setVector(0, 0);
  drawRaster();
  drawPath();
}

async function start() {
  try {
    await loadModel();
  } catch {
    return;
  }
  running = true;
  lastStep = performance.now();
  toggle.textContent = "Pause";
  setStatus("Running · model runs locally");
}

function pause(message = "Paused") {
  running = false;
  toggle.textContent = model ? "Run" : "Load instrument";
  setStatus(message);
}

function stepModel() {
  const next = worldStep(latents, actions, model);
  latents.push(next);
  actions.push(action.slice());
  if (latents.length > CAP) {
    latents.shift();
    actions.shift();
  }

  const rates = decode(next, model);
  const velocity = behavior(next, model);
  history.push(rates);
  if (history.length > HISTORY) history.shift();

  position = [
    position[0] * .95 + velocity[0] * .15,
    position[1] * .95 + velocity[1] * .15,
  ];
  path.push(position.slice());
  if (path.length > 260) path.shift();
}

function ramp(value, output) {
  const t = Math.max(0, Math.min(1, value));
  const scaled = t * 8;
  const index = Math.floor(scaled);
  const mix = scaled - index;
  const first = RAMP[index];
  const second = RAMP[Math.min(index + 1, 8)];
  output[0] = Math.floor(first[0] + (second[0] - first[0]) * mix);
  output[1] = Math.floor(first[1] + (second[1] - first[1]) * mix);
  output[2] = Math.floor(first[2] + (second[2] - first[2]) * mix);
}

function drawRaster() {
  const units = model?.readout.length || 32;
  raster.width = HISTORY;
  raster.height = units;
  rasterContext.imageSmoothingEnabled = false;
  const image = rasterContext.createImageData(HISTORY, units);
  const color = [0, 0, 0];

  for (let x = 0; x < history.length; x += 1) {
    for (let y = 0; y < units; y += 1) {
      ramp(Math.log1p(Math.min(history[x][y], 24) / 2) / 2.5649, color);
      const column = HISTORY - history.length + x;
      const offset = (y * HISTORY + column) * 4;
      image.data[offset] = color[0];
      image.data[offset + 1] = color[1];
      image.data[offset + 2] = color[2];
      image.data[offset + 3] = 255;
    }
  }
  rasterContext.putImageData(image, 0, 0);
}

function drawPath() {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = pathCanvas.clientWidth;
  const height = 200;
  pathCanvas.width = width * ratio;
  pathCanvas.height = height * ratio;
  pathContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  pathContext.clearRect(0, 0, width, height);

  const centerX = width / 2;
  const centerY = height / 2;
  const scale = 6;
  const count = path.length;
  pathContext.strokeStyle = "#293040";
  pathContext.lineWidth = 1;
  pathContext.beginPath();
  pathContext.moveTo(centerX - 6, centerY);
  pathContext.lineTo(centerX + 6, centerY);
  pathContext.moveTo(centerX, centerY - 6);
  pathContext.lineTo(centerX, centerY + 6);
  pathContext.stroke();
  pathContext.strokeStyle = "#1c2230";
  pathContext.beginPath();
  pathContext.arc(centerX, centerY, Math.min(width, height) * .42, 0, Math.PI * 2);
  pathContext.stroke();

  const actionMagnitude = Math.hypot(action[0], action[1]);
  if (actionMagnitude > .01) {
    pathContext.strokeStyle = "rgb(88 183 212 / 38%)";
    pathContext.setLineDash([3, 3]);
    pathContext.beginPath();
    pathContext.moveTo(centerX, centerY);
    pathContext.lineTo(
      centerX + (action[0] / actionMagnitude) * 46,
      centerY - (action[1] / actionMagnitude) * 46,
    );
    pathContext.stroke();
    pathContext.setLineDash([]);
  }

  pathContext.lineCap = "round";
  pathContext.lineJoin = "round";
  for (let index = 1; index < count; index += 1) {
    const time = index / (count - 1);
    const first = path[index - 1];
    const second = path[index];
    pathContext.strokeStyle = `rgb(88 183 212 / ${(time * time * .9).toFixed(3)})`;
    pathContext.lineWidth = .6 + time * 1.6;
    pathContext.beginPath();
    pathContext.moveTo(centerX + first[0] * scale, centerY - first[1] * scale);
    pathContext.lineTo(centerX + second[0] * scale, centerY - second[1] * scale);
    pathContext.stroke();
  }

  pathContext.strokeStyle = "#293040";
  pathContext.lineWidth = 1;
  pathContext.beginPath();
  pathContext.arc(centerX, centerY, 3, 0, Math.PI * 2);
  pathContext.stroke();

  const x = centerX + head[0] * scale;
  const y = centerY - head[1] * scale;
  pathContext.fillStyle = "#1D243E";
  pathContext.beginPath();
  pathContext.arc(x, y, 2.6, 0, Math.PI * 2);
  pathContext.fill();
  pathContext.strokeStyle = "rgb(88 183 212 / 55%)";
  pathContext.beginPath();
  pathContext.arc(x, y, 5, 0, Math.PI * 2);
  pathContext.stroke();

  if (count > 14 && actionMagnitude > .05) {
    const dx = path[count - 1][0] - path[count - 14][0];
    const dy = path[count - 1][1] - path[count - 14][1];
    const decodedMagnitude = Math.hypot(dx, dy);
    if (decodedMagnitude > 1e-3) {
      const cosine = (dx * action[0] + dy * action[1]) / (decodedMagnitude * actionMagnitude);
      alignment = alignment === undefined ? cosine : alignment * .9 + cosine * .1;
    }
  }
  align.value = `align ${alignment === undefined ? "—" : Math.max(-1, Math.min(1, alignment)).toFixed(2)}`;
  align.style.color = alignment === undefined ? "var(--ink-3)" : "var(--ink-2)";
}

function setVector(dx, dy) {
  const magnitude = Math.min(1, Math.hypot(dx, dy));
  const degrees = (Math.round(Math.atan2(-dy, dx) * 180 / Math.PI) + 360) % 360;
  knob.style.left = `${92 + dx * 92}px`;
  knob.style.top = `${92 + dy * 92}px`;
  vector.style.width = `${magnitude * 92}px`;
  vector.style.transform = `rotate(${Math.atan2(dy, dx)}rad)`;
  readout.classList.toggle("idle", magnitude <= .01);
  readout.value = `θ ${magnitude > .01 ? degrees : "—"}° · ‖v‖ ${magnitude > .01 ? magnitude.toFixed(2) : "—"}`;
  action = [dx * 1.3, -dy * 1.3];
}

function setPointerAction(event) {
  const bounds = pad.getBoundingClientRect();
  let dx = (event.clientX - bounds.left - bounds.width / 2) / (bounds.width / 2);
  let dy = (event.clientY - bounds.top - bounds.height / 2) / (bounds.height / 2);
  const magnitude = Math.hypot(dx, dy);
  if (magnitude > 1) {
    dx /= magnitude;
    dy /= magnitude;
  }
  setVector(dx, dy);
}

toggle.addEventListener("click", () => {
  if (running) pause();
  else start();
});

reset.addEventListener("click", initialise);

pad.addEventListener("pointerdown", async (event) => {
  dragging = true;
  pad.classList.add("grab");
  pad.setPointerCapture(event.pointerId);
  setPointerAction(event);
  await start();
});

pad.addEventListener("pointermove", (event) => {
  if (dragging) setPointerAction(event);
});

function releasePointer() {
  dragging = false;
  pad.classList.remove("grab");
}

pad.addEventListener("pointerup", releasePointer);
pad.addEventListener("pointercancel", releasePointer);

pad.addEventListener("keydown", async (event) => {
  const key = event.key.toLowerCase();
  if (key === "escape") {
    setVector(0, 0);
    return;
  }
  const directions = {
    arrowup: [0, -1],
    w: [0, -1],
    arrowdown: [0, 1],
    s: [0, 1],
    arrowleft: [-1, 0],
    a: [-1, 0],
    arrowright: [1, 0],
    d: [1, 0],
  };
  if (!directions[key]) return;
  event.preventDefault();
  await start();
  setVector(...directions[key]);
});

window.addEventListener("blur", () => {
  setVector(0, 0);
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden && running) pause("Paused while tab is hidden");
});

window.addEventListener("resize", drawPath);

setInterval(() => {
  if (!running || !model) return;
  const now = performance.now();
  if (now - lastStep >= 110) {
    stepModel();
    drawRaster();
    lastStep = now;
  }
  if (path.length) {
    const latest = path[path.length - 1];
    head = [
      head[0] + (latest[0] - head[0]) * .35,
      head[1] + (latest[1] - head[1]) * .35,
    ];
  }
  drawPath();
}, 40);

drawRaster();
drawPath();
