const byId = (id) => document.getElementById(id);

function signed(value, digits = 3) {
  if (!Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function position(value, domain) {
  const bounded = Math.max(domain[0], Math.min(domain[1], value));
  return ((bounded - domain[0]) / (domain[1] - domain[0])) * 100;
}

function validEstimate(estimate) {
  return Number.isFinite(estimate?.value)
    && Array.isArray(estimate?.interval)
    && estimate.interval.length === 2
    && estimate.interval.every(Number.isFinite);
}

function valid(data) {
  const statuses = new Set(["supported", "not_supported", "stopped_at_selection"]);
  return data?.schema_version === 2
    && Array.isArray(data.benchmarks)
    && Array.isArray(data.studies)
    && data.studies.every((study) =>
      statuses.has(study.status)
      && Array.isArray(study.domain)
      && study.domain.length === 2
      && study.domain[0] < study.domain[1]
      && Array.isArray(study.comparisons)
      && study.comparisons.every(validEstimate));
}

function renderBenchmarks(benchmarks) {
  const ledger = byId("benchmark-ledger");
  ledger.className = "benchmark-ledger";
  ledger.append(element("h3", "", "Reference benchmarks"));
  for (const benchmark of benchmarks) {
    const row = element("div", "benchmark-row");
    row.append(
      element("strong", "", benchmark.name),
      element("b", "", benchmark.value),
      element("span", "", benchmark.scope),
      element("small", "", benchmark.boundary),
    );
    ledger.append(row);
  }
}

function estimateRow(estimate, domain) {
  const row = element("div", "forest-row");
  const track = element("div", "forest-track");
  const zero = position(0, domain);
  const low = position(estimate.interval[0], domain);
  const high = position(estimate.interval[1], domain);
  const point = position(estimate.value, domain);
  track.style.setProperty("--zero", `${zero}%`);
  track.style.setProperty("--low", `${low}%`);
  track.style.setProperty("--width", `${high - low}%`);
  track.style.setProperty("--point", `${point}%`);
  track.append(element("i"), element("b"));
  row.append(
    element("span", "", estimate.label),
    track,
    element(
      "output",
      "",
      `${signed(estimate.value)} [${signed(estimate.interval[0])}, ${signed(estimate.interval[1])}]`,
    ),
  );
  return row;
}

function renderStudy(study) {
  const plate = element("article", "study-plate");
  const head = element("div", "study-head");
  const title = element("div");
  title.append(
    element("h3", "", study.title),
    element("p", "", study.scope),
  );
  const decision = element("p", "decision", study.decision);
  decision.dataset.status = study.status;
  head.append(title, decision);

  const route = element("div", "route");
  route.setAttribute("aria-label", "Recorded source and evaluated targets");
  route.append(
    element("strong", "", study.route.source),
    element("span", "", "→"),
    element("span", "", study.route.targets.join(" · ")),
    element("small", "", `${study.horizons_ms.join("/")} ms`),
  );

  const figure = element("figure", "forest");
  figure.append(element("figcaption", "", "Paired future bits/spike · model minus comparator"));
  const axis = element("div", "forest-axis");
  axis.setAttribute("aria-hidden", "true");
  const axisTrack = element("div", "forest-axis-track");
  axisTrack.style.setProperty("--zero", `${position(0, study.domain)}%`);
  axisTrack.append(
    element("span", "", signed(study.domain[0], 2)),
    element("span", "", "0"),
    element("span", "", signed(study.domain[1], 2)),
  );
  axis.append(axisTrack);
  figure.append(axis, ...study.comparisons.map((item) => estimateRow(item, study.domain)));

  plate.append(head, route, figure);
  if (study.notes.length) {
    const notes = element("div", "study-notes");
    for (const note of study.notes) {
      const row = element("div", "study-note-row");
      row.append(
        element("span", "", note.label),
        element("b", "", `${signed(note.value)} [${signed(note.interval[0])}, ${signed(note.interval[1])}]`),
        element("small", "", note.boundary),
      );
      notes.append(row);
    }
    plate.append(notes);
  }
  plate.append(element("p", "study-boundary", study.boundary));
  return plate;
}

function render(data) {
  renderBenchmarks(data.benchmarks);
  byId("study-list").className = "study-list";
  byId("study-list").append(...data.studies.map(renderStudy));
}

fetch(new URL("./study-data.json", import.meta.url))
  .then((response) => {
    if (!response.ok) throw new Error(`study request returned ${response.status}`);
    return response.json();
  })
  .then((data) => {
    if (!valid(data)) throw new Error("study artifact is incomplete");
    render(data);
  })
  .catch((error) => {
    byId("study-error").textContent = `Reviewed evidence unavailable: ${error.message}`;
  });
