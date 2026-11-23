"use strict";

document.documentElement.classList.add("js");

const walkthrough = document.querySelector("#construction-walkthrough");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
walkthrough.open = !reducedMotion.matches;
reducedMotion.addEventListener("change", (event) => {
  if (event.matches) walkthrough.open = false;
});

const menuToggle = document.querySelector(".menu-toggle");
const navigation = document.querySelector("#nav-links");
function closeMenu() {
  menuToggle.setAttribute("aria-expanded", "false");
  navigation.classList.remove("is-open");
}
menuToggle.addEventListener("click", () => {
  const open = menuToggle.getAttribute("aria-expanded") !== "true";
  menuToggle.setAttribute("aria-expanded", String(open));
  navigation.classList.toggle("is-open", open);
});
navigation.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && menuToggle.getAttribute("aria-expanded") === "true") {
    closeMenu();
    menuToggle.focus();
  }
});

if ("IntersectionObserver" in window) {
  const visibleSections = new Set();
  const sectionLinks = [...navigation.querySelectorAll("a")];
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) visibleSections.add(entry.target.id);
      else visibleSections.delete(entry.target.id);
    });
    const current = sectionLinks.find((link) => visibleSections.has(link.hash.slice(1)));
    sectionLinks.forEach((link) => {
      if (link === current) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }, { rootMargin: "-15% 0px -65% 0px" });
  sectionLinks.forEach((link) => {
    const section = document.querySelector(link.hash);
    if (section) observer.observe(section);
  });
}

const dialog = document.querySelector("#figure-dialog");
const expandedImage = document.querySelector("#expanded-figure");
document.querySelectorAll("[data-figure]").forEach((link) => {
  link.addEventListener("click", (event) => {
    if (typeof dialog.showModal !== "function" || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const image = link.querySelector("img");
    expandedImage.src = link.href;
    expandedImage.alt = image.alt;
    document.querySelector("#figure-title").textContent = link.dataset.title;
    document.querySelector("#full-figure-link").href = link.href;
    dialog.showModal();
    document.body.classList.add("dialog-open");
  });
});
document.querySelector("#close-figure").addEventListener("click", () => dialog.close());
dialog.addEventListener("close", () => document.body.classList.remove("dialog-open"));
dialog.addEventListener("click", (event) => {
  const bounds = dialog.getBoundingClientRect();
  if (event.target === dialog && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) dialog.close();
});

const copyButton = document.querySelector("#copy-citation");
const citation = document.querySelector("#bibtex");
const copyStatus = document.querySelector("#copy-status");
copyButton.addEventListener("click", async () => {
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
    await navigator.clipboard.writeText(citation.textContent);
    copyStatus.textContent = "Citation copied to clipboard.";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citation);
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = "Citation selected. Press Ctrl+C (Windows/Linux) or Command+C (Mac) to copy.";
  }
});
