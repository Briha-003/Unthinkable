// -----------------------------
// static/script.js
// -----------------------------

// -------- Upload flow --------
document.getElementById("uploadBtn").onclick = async () => {
  const input = document.getElementById("fileInput");
  if (!input.files.length) return alert("Select files to upload.");
  const form = new FormData();
  for (let f of input.files) form.append("files", f);
  document.getElementById("uploadStatus").textContent =
    "Uploading & indexing (this may take a while)...";
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const j = await res.json();
    if (j.error) {
      document.getElementById("uploadStatus").textContent = "Error: " + j.error;
    } else {
      document.getElementById("uploadStatus").textContent =
        "Uploaded: " + j.saved.join(", ");
      document.getElementById("docList").textContent =
        "Documents: " + j.saved.join(", ");
    }
  } catch (e) {
    document.getElementById("uploadStatus").textContent =
      "Upload failed: " + e;
  }
};

// -------- Ask flow --------
async function ask(question, topk = 4, refine_instruction = null, previous_answer = null) {
  const payload = { question, top_k: topk };
  if (refine_instruction) payload["refine_instruction"] = refine_instruction;
  if (previous_answer) payload["previous_answer"] = previous_answer;

  const resp = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return await resp.json();
}

document.getElementById("askBtn").onclick = async () => {
  const q = document.getElementById("question").value.trim();
  if (!q) return alert("Enter a question.");
  document.getElementById("answerCard").classList.remove("hidden");
  document.getElementById("answerText").textContent = "Thinking...";
  const topk = parseInt(document.getElementById("topk").value);
  try {
    const res = await ask(q, topk);
    if (res.error) {
      document.getElementById("answerText").textContent =
        "Error: " + res.error;
      return;
    }
    showResult(q, res);
  } catch (e) {
    document.getElementById("answerText").textContent =
      "Request failed: " + e;
  }
};

// -------- Display + Refine + Feedback --------
function showResult(question, res) {
  // --- Meta info ---
  document.getElementById("confidence").textContent =
    "Confidence: " + (res.confidence ? res.confidence.toFixed(3) : "-");
  document.getElementById("latency").textContent =
    res.time ? res.time.toFixed(2) + "s" : "";
  document.getElementById("tldr").textContent = res.tldr || "";
  document.getElementById("answerText").textContent = res.answer || "";

  // --- Context list ---
  const ul = document.getElementById("contexts");
  ul.innerHTML = "";
  (res.contexts || []).forEach((c) => {
    const li = document.createElement("li");
    li.style.display = "flex";
    li.style.justifyContent = "space-between";
    li.style.alignItems = "flex-start";
    li.style.padding = "8px";
    li.style.borderBottom = "1px solid #ccc";

    const left = document.createElement("div");
    left.style.flex = "1";
    const title = document.createElement("div");
    title.textContent = `${c.doc} — page ${c.page} (chunk ${c.chunk_id})`;
    title.style.fontWeight = "600";
    const snippet = document.createElement("div");
    snippet.textContent =
      c.text.slice(0, 320) + (c.text.length > 320 ? "..." : "");
    snippet.style.marginTop = "6px";
    snippet.style.cursor = "pointer";
    snippet.title = "Click to copy snippet";
    left.appendChild(title);
    left.appendChild(snippet);

    const right = document.createElement("div");
    right.style.textAlign = "right";
    const open = document.createElement("a");
    open.href = c._url;
    open.target = "_blank";
    open.textContent = "Open";
    const sim = document.createElement("div");
    sim.textContent =
      "sim: " + (c._similarity ? c._similarity.toFixed(3) : "-");
    sim.style.marginTop = "6px";
    right.appendChild(open);
    right.appendChild(sim);

    li.appendChild(left);
    li.appendChild(right);
    ul.appendChild(li);

    snippet.onclick = () => {
      navigator.clipboard
        .writeText(c.text)
        .then(() => alert("Snippet copied!"));
    };
  });

  // --- Refine section ---
  const refineBox = document.getElementById("refineBox");
  refineBox.style.display = "block";
  const applyRefine = document.getElementById("applyRefine");

  // Remove old handlers (avoid stacking)
  const newApplyRefine = applyRefine.cloneNode(true);
  applyRefine.parentNode.replaceChild(newApplyRefine, applyRefine);

  newApplyRefine.onclick = async () => {
    const instr = document.getElementById("refineInstruction").value.trim();
    if (!instr) {
      alert("Please enter a refinement instruction (e.g., 'Be concise').");
      return;
    }
    const prevAnswer = document.getElementById("answerText").textContent;
    const topk = parseInt(document.getElementById("topk").value) || 3;
    document.getElementById("answerText").textContent =
      "⏳ Refining answer...";

    try {
      const res2 = await ask(question, topk, instr, prevAnswer);
      if (res2.error) {
        document.getElementById("answerText").textContent =
          "❌ Error: " + res2.error;
        return;
      }
      showResult(question, res2);
    } catch (err) {
      console.error(err);
      document.getElementById("answerText").textContent =
        "❌ Failed to fetch refined answer.";
    }
  };

  // --- Feedback section ---
  const qbox = document.getElementById("question").value.trim();
  document.getElementById("thumbUp").onclick = () =>
    sendFeedback(qbox, res.answer, "up", res.contexts);
  document.getElementById("thumbDown").onclick = () =>
    sendFeedback(qbox, res.answer, "down", res.contexts);
  document.getElementById("copyBtn").onclick = () => {
    navigator.clipboard.writeText(res.answer);
    alert("Answer copied!");
  };

  // Toggle refine input visibility
  document.getElementById("refineBtn").onclick = () => {
    document.getElementById("refineBox").classList.toggle("hidden");
  };
}

// -------- Feedback flow --------
async function sendFeedback(question, answer, feedback, contexts) {
  const comment = document.getElementById("feedbackComment").value.trim();
  const payload = { question, answer, feedback, comment, contexts };
  await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  alert("Feedback sent. Thank you!");
}
