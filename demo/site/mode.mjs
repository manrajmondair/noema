// The palette itself is chosen in CSS, so the page is already correct before this
// module parses. All this does is let the reader override the system preference and
// keep the control's label honest about what pressing it will do.
const root = document.documentElement;
const button = document.getElementById("mode");
const system = matchMedia("(prefers-color-scheme: dark)");

const isDark = () =>
  root.dataset.mode ? root.dataset.mode === "dark" : system.matches;

function sync() {
  const dark = isDark();
  button.textContent = dark ? "Light" : "Dark";
  button.setAttribute("aria-pressed", String(dark));
}

button.addEventListener("click", () => {
  root.dataset.mode = isDark() ? "light" : "dark";
  sync();
});
system.addEventListener("change", sync);
sync();
