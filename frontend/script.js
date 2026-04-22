"use strict";

const API_BASE = "http://127.0.0.1:8000";
const API_URL  = `${API_BASE}/predict`;

// ── DOM refs ────────────────────────────────────────────────────────────────
const $  = (id) => document.getElementById(id);

const els = {
  form:        $("predictForm"),
  expYears:    $("expYears"),
  expMonths:   $("expMonths"),
  jobRole:     $("jobRole"),
  location:    $("location"),
  skillInput:  $("skillInput"),
  chipTrack:   $("chipTrack"),
  skillBox:    $("skillBox"),
  predictBtn:  $("predictBtn"),
  formError:   $("formError"),
  resultPanel: $("resultPanel"),
  resultMeta:  $("resultMeta"),
  annualLpa:   $("annualLpa"),
  annualInr:   $("annualInr"),
  monthlyInr:  $("monthlyInr"),
  justification:$("justification"),
  resetBtn:    $("resetBtn"),
  apiStatus:   $("apiStatus"), // optional now
};

// ─────────────────────────────────────────────
// Skill Dropdown
// ─────────────────────────────────────────────
const input = els.skillInput;
const chipTrack = els.chipTrack;
const skillBox = els.skillBox;
const dropdown = $("skillDropdown");

// Skills from datalist
const skills = Array.from(
  document.querySelectorAll("#skillSuggestions option")
).map(opt => opt.value);

let selected = [];

// ─────────────────────────────────────────────
function renderDropdown(filter = "") {
  if (!dropdown) return;

  dropdown.innerHTML = "";

  const filtered = skills.filter(skill =>
    skill.toLowerCase().includes(filter.toLowerCase()) &&
    !selected.includes(skill)
  );

  if (!filtered.length) {
    dropdown.classList.add("hidden");
    return;
  }

  filtered.forEach(skill => {
    const item = document.createElement("div");
    item.className = "dropdown-item";
    item.textContent = skill;
    item.onclick = () => addSkill(skill);
    dropdown.appendChild(item);
  });

  dropdown.classList.remove("hidden");
}

// ─────────────────────────────────────────────
function addSkill(skill) {
  skill = skill.trim();
  if (!skill || selected.includes(skill)) return;

  selected.push(skill);
  renderChips();

  if (input) input.value = "";
  if (dropdown) dropdown.classList.add("hidden");
}

// ─────────────────────────────────────────────
function removeSkill(skill) {
  selected = selected.filter(s => s !== skill);
  renderChips();
}

// ─────────────────────────────────────────────
function renderChips() {
  if (!chipTrack) return;

  chipTrack.innerHTML = "";

  selected.forEach(skill => {
    const chip = document.createElement("div");
    chip.className = "chip";

    const text = document.createElement("span");
    text.textContent = skill;

    const btn = document.createElement("button");
    btn.textContent = "×";
    btn.onclick = () => removeSkill(skill);

    chip.appendChild(text);
    chip.appendChild(btn);

    chipTrack.appendChild(chip);
  });
}

// ─────────────────────────────────────────────
// Events
// ─────────────────────────────────────────────
if (input) {
  input.addEventListener("input", () => renderDropdown(input.value));
  input.addEventListener("focus", () => renderDropdown(input.value));

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addSkill(input.value);
    }
  });
}

document.addEventListener("click", (e) => {
  if (!skillBox || !dropdown) return;
  if (!skillBox.contains(e.target)) {
    dropdown.classList.add("hidden");
  }
});

// ── Formatters ─────────────────────────────────────────────
function fmtINR(n) {
  return `₹${Math.round(n).toLocaleString("en-IN")}`;
}

function fmtLPA(n) {
  return `${(n / 100000).toFixed(2)} LPA`;
}

// ── API health (SAFE)
async function checkHealth() {
  if (!els.apiStatus) return; // ← prevents crash

  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    els.apiStatus.className = `api-status ${res.ok ? "online" : "offline"}`;
  } catch {
    els.apiStatus.className = "api-status offline";
  }
}

checkHealth();
setInterval(checkHealth, 15000);

// ── Form submit ─────────────────────────────────────────────
if (els.form) {
  els.form.addEventListener("submit", async (e) => {
    e.preventDefault();
    showError("");

    const years   = Number(els.expYears.value);
    const months  = Number(els.expMonths.value);
    const jobRole = els.jobRole.value.trim();
    const location= els.location.value.trim();
    const skillsCSV = getSkillsCSV();

    if (!Number.isFinite(years) || years < 0 || years > 50)
      return showError("Enter valid experience (0–50 years).");

    if (!Number.isFinite(months) || months < 0 || months > 11)
      return showError("Months must be 0–11.");

    if (!jobRole) return showError("Select job role.");
    if (!location) return showError("Select location.");
    if (!skillsCSV) return showError("Add at least one skill.");

    const experience_years = years + months / 12;

    const payload = {
      experience_years,
      job_role: jobRole,
      skills: skillsCSV,
      location,
    };

    setLoading(true);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        let msg = "API error";
        try {
          msg = (await res.json()).detail || msg;
        } catch {}
        throw new Error(msg);
      }

      const data = await res.json();
      renderResult(data, { jobRole, location, years, months });

    } catch (err) {
      showError(err.message);
    } finally {
      setLoading(false);
    }
  });
}

// ── Render result ─────────────────────────────
function renderResult(data, { jobRole, location, years, months }) {
  const salary  = Number(data.predicted_salary);
  const monthly = salary / 12;

  els.annualLpa.textContent  = fmtLPA(salary);
  els.annualInr.textContent  = `${fmtINR(salary)} / year`;
  els.monthlyInr.textContent = `${fmtINR(monthly)} / month`;
  els.justification.textContent = data.justification;

  const expStr = months > 0
    ? `${years}y ${months}mo`
    : `${years} yr`;

  els.resultMeta.textContent =
    `${jobRole} · ${location} · ${expStr} experience`;

  els.resultPanel.classList.remove("hidden");
}

// ── Reset ─────────────────────────────────────
if (els.resetBtn) {
  els.resetBtn.addEventListener("click", () => {
    els.resultPanel.classList.add("hidden");
    els.form.scrollIntoView({ behavior: "smooth" });
    showError("");
  });
}

// ── REQUIRED FUNCTIONS (missing earlier)
function showError(msg) {
  if (els.formError) els.formError.textContent = msg || "";
}

function getSkillsCSV() {
  return selected.join(",");
}

function setLoading(state) {
  if (!els.predictBtn) return;

  els.predictBtn.disabled = state;

  if (state) {
    els.predictBtn.classList.add("loading");
  } else {
    els.predictBtn.classList.remove("loading");
  }
}