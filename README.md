# AliceBook — Markdown Extraction, Notes, and Labs

This repository contains a **Markdown extraction** of the book  
*Alice’s Adventures in a Differentiable Wonderland*, together with **summarized / distilled notes** and **practical lab materials** derived from the original work.

<div id="notes-popup" style="display:none; position:fixed; right:18px; bottom:18px; max-width:320px; background:#0f172a; color:#e2e8f0; border-radius:12px; padding:14px 16px 12px; box-shadow:0 18px 36px rgba(15,23,42,0.25); z-index:9999; border:1px solid rgba(255,255,255,0.12); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
    <div>
      <div style="font-weight:700; letter-spacing:-0.01em; margin-bottom:4px;">View the rendered notes</div>
      <div style="font-size:14px; line-height:1.35;">Open the HTML exports at <a href="https://trojblue.github.io/alice-book/extracted_notes/" style="color:#93c5fd; font-weight:600;">https://trojblue.github.io/alice-book/extracted_notes/</a>.</div>
    </div>
    <button id="notes-popup-close" aria-label="Dismiss" style="background:transparent; border:none; color:#cbd5e1; font-size:16px; cursor:pointer; line-height:1;">×</button>
  </div>
</div>

<script>
(function() {
  const popup = document.getElementById("notes-popup");
  const closeBtn = document.getElementById("notes-popup-close");
  const key = "alice-book-notes-popup-dismissed";
  if (!popup || !closeBtn) return;
  if (localStorage.getItem(key) === "true") return;
  popup.style.display = "block";
  closeBtn.addEventListener("click", () => {
    popup.remove();
    localStorage.setItem(key, "true");
  });
})();
</script>

The goal of this project is to make the material easier to navigate, annotate, and study in a plain-text, version-controlled format.

---

## Source Material

**Title:** *Alice’s Adventures in a Differentiable Wonderland*  
**Author:** Simone Scardapane  
**Copyright:** © 2025 Simone Scardapane  

The original book introduces differentiable programming and neural networks, covering topics such as automatic differentiation, convolutional, attentional, recurrent, graph-based, and transformer architectures, with practical references to PyTorch and JAX.

Original sources include:
- Author’s website
- arXiv preprint: arXiv:2404.17625
- Commercial editions published independently by the author

All rights to the original text belong to the original author.

---

## License and Attribution

The original book is released under the  
**Creative Commons Attribution–ShareAlike 4.0 International License (CC BY-SA 4.0)**.

Accordingly:

- This repository is also licensed under **CC BY-SA 4.0**
- You are free to **share, adapt, and build upon** the contents of this repository
- **Attribution must be given** to the original author
- **Derivative works must be licensed under the same terms**

A copy of the CC BY-SA 4.0 license is provided in the `LICENSE` file.

---

## Nature of the Modifications

This repository includes:

- A **reformatted Markdown version** of the original text
- **Summaries, distilled notes, and study-oriented restructurings**
- **Lab-style materials** aligned with topics in the book

The content has been converted from its original format and reorganized for clarity and study.  
No claim is made to original authorship of the underlying material.

Where changes have been made, they are limited to:
- Formatting and structural adjustments
- Summarization and condensation
- Educational annotations and exercises

---

## Disclaimer

This project is **not affiliated with or endorsed by** the original author or publisher.

If you are looking for the authoritative version of the book, or wish to support the author, please consult the original sources.

---

## License

This repository is licensed under the  
**Creative Commons Attribution–ShareAlike 4.0 International License (CC BY-SA 4.0)**.

See the `LICENSE` file for details.
