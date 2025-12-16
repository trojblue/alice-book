---
---

{% include_relative README.md %}

<script>
(function() {
  if (typeof window === "undefined") return;
  if (!/github\\.io$/.test(window.location.hostname)) return;
  const key = "alice-book-notes-popup-dismissed";
  if (localStorage.getItem(key) === "true") return;

  const box = document.createElement("div");
  box.style.position = "fixed";
  box.style.right = "18px";
  box.style.bottom = "18px";
  box.style.maxWidth = "320px";
  box.style.background = "#0f172a";
  box.style.color = "#e2e8f0";
  box.style.borderRadius = "12px";
  box.style.padding = "14px 16px 12px";
  box.style.boxShadow = "0 18px 36px rgba(15,23,42,0.25)";
  box.style.zIndex = "9999";
  box.style.border = "1px solid rgba(255,255,255,0.12)";
  box.style.fontFamily = "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";

  const text = document.createElement("div");
  text.style.fontSize = "14px";
  text.style.lineHeight = "1.35";
  text.innerHTML = 'View the rendered notes at <a href="https://trojblue.github.io/alice-book/extracted_notes/" style="color:#93c5fd; font-weight:600;">https://trojblue.github.io/alice-book/extracted_notes/</a>.';

  const title = document.createElement("div");
  title.textContent = "Rendered notes";
  title.style.fontWeight = "700";
  title.style.letterSpacing = "-0.01em";
  title.style.marginBottom = "4px";
  title.style.fontSize = "15px";

  const close = document.createElement("button");
  close.setAttribute("aria-label", "Dismiss");
  close.textContent = "×";
  close.style.background = "transparent";
  close.style.border = "none";
  close.style.color = "#cbd5e1";
  close.style.fontSize = "16px";
  close.style.cursor = "pointer";
  close.style.lineHeight = "1";
  close.style.marginLeft = "12px";

  const row = document.createElement("div");
  row.style.display = "flex";
  row.style.justifyContent = "space-between";
  row.style.alignItems = "flex-start";
  row.style.gap = "12px";

  const col = document.createElement("div");
  col.appendChild(title);
  col.appendChild(text);

  row.appendChild(col);
  row.appendChild(close);
  box.appendChild(row);

  close.addEventListener("click", () => {
    box.remove();
    localStorage.setItem(key, "true");
  });

  document.body.appendChild(box);
})();
</script>
